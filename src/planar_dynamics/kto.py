import numpy as np
from pydrake.math import BsplineBasis, KnotVectorType
from pydrake.trajectories import BsplineTrajectory, PiecewisePolynomial

from planar_physics.certificate import CertifierConfig
from planar_physics.drake_model import get_model

ORDER = 4
N_CP = 10
N_COLL = 40
CUSHION_M = 5e-3
ROUNDS = 6
MAJOR_ITERS = 100
T0_FACTOR = .75
TOPPRA_RUNGS = ((200, 1e-5),)
SEED_SPACING = .2
SEED_SAMPLES = 300
KIN_SCALE = .995
INFLUENCE_M = .02
LIMIT_MARGIN = 2e-3
MAX_NEW_CLUSTERS = 6


class KTOUnavailable(RuntimeError):
    pass


def _basis_matrix(basis, s):
    n = basis.num_basis_functions()
    return np.array([[basis.EvaluateBasisFunctionI(i, float(x)) for i in range(n)] for x in s])


def sample_by_arclength(curve_fn, m=400, probe=4001):
    u = np.linspace(0., 1., probe)
    q = curve_fn(u)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    arc /= arc[-1]
    target = np.linspace(0., 1., m)
    return np.column_stack([np.interp(target, arc, q[:, j]) for j in range(q.shape[1])])


def fit_seed(samples, n_cp):
    m, n = samples.shape
    basis = BsplineBasis(ORDER, n_cp, KnotVectorType.kClampedUniform, 0., 1.)
    A = _basis_matrix(basis, np.linspace(0., 1., m))
    fixed = {0: samples[0], 1: samples[0], n_cp-2: samples[-1], n_cp-1: samples[-1]}
    free = [i for i in range(n_cp) if i not in fixed]
    rhs = samples - sum(np.outer(A[:, i], v) for i, v in fixed.items())
    P = np.zeros((n_cp, n))
    for i, v in fixed.items():
        P[i] = v
    P[free] = np.linalg.lstsq(A[:, free], rhs, rcond=None)[0]
    return P


def to_cubic(traj, start=None):
    T = traj.end_time() - traj.start_time()
    knots = np.unique(np.asarray(traj.basis().knots()))
    q = traj.vector_values(knots)
    v = traj.MakeDerivative().vector_values(knots)
    if start is not None:
        q[:, 0] = start
    return PiecewisePolynomial.CubicHermite(knots/T, q, v*T)


def failing_clusters(s, free, max_new=MAX_NEW_CLUSTERS):
    bad = np.flatnonzero(~free)
    if not len(bad):
        return []
    groups = np.split(bad, np.flatnonzero(np.diff(bad) > 1)+1)
    groups.sort(key=len, reverse=True)
    picks = []
    for g in groups[:max_new]:
        picks += [s[g[0]], s[g[len(g)//2]], s[g[-1]]]
    return sorted(set(float(x) for x in picks))


class KTOProblem:
    def __init__(self, robot, scene, start, goal, n_cp):
        from pydrake.planning import KinematicTrajectoryOptimization
        plant = get_model(robot, scene).plant
        self.plant = plant
        self.root_ctx = get_model(robot, scene).diagram.CreateDefaultContext()
        self.plant_ctx = plant.GetMyMutableContextFromRoot(self.root_ctx)
        self.kto = k = KinematicTrajectoryOptimization(robot.n_joints, n_cp, ORDER)
        k.AddDurationCost(1.)
        lo = np.asarray(robot.lo_rad)+LIMIT_MARGIN
        hi = np.asarray(robot.hi_rad)-LIMIT_MARGIN
        start = np.asarray(start, dtype=float)
        self.start_outside = bool(np.any(start < lo) or np.any(start > hi))
        if self.start_outside:
            prog = k.get_mutable_prog()
            wide = (np.minimum(lo, start), np.maximum(hi, start))
            for index in range(n_cp):
                bounds = wide if index < 2 else (lo, hi)
                prog.AddBoundingBoxConstraint(bounds[0], bounds[1], k.control_points()[:, index])
        else:
            k.AddPositionBounds(lo, hi)
        v = np.asarray(plant.GetVelocityUpperLimits())*KIN_SCALE
        a = np.asarray(plant.GetAccelerationUpperLimits())*KIN_SCALE
        k.AddVelocityBounds(-v, v)
        k.AddAccelerationBounds(-a, a)
        k.AddDurationConstraint(.05, 30.)
        z = np.zeros(robot.n_joints)
        k.AddPathPositionConstraint(start, start, 0.)
        goal = np.clip(np.asarray(goal, dtype=float), lo, hi)
        k.AddPathPositionConstraint(goal, goal, 1.)
        k.AddPathVelocityConstraint(z, z, 0.)
        k.AddPathVelocityConstraint(z, z, 1.)

    def add_distance(self, s_values, bound):
        from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint
        c = MinimumDistanceLowerBoundConstraint(self.plant, float(bound), self.plant_ctx, None,
                                                INFLUENCE_M)
        for s in np.atleast_1d(s_values):
            self.kto.AddPathPositionConstraint(c, float(s))

    def solve(self, guess=None, warm=None):
        from pydrake.solvers import SnoptSolver, SolverOptions
        prog = self.kto.get_mutable_prog()
        if guess is not None:
            self.kto.SetInitialGuess(guess)
        solver = SnoptSolver()
        if not (solver.available() and solver.enabled()):
            raise KTOUnavailable('snopt_unavailable')
        opts = SolverOptions()
        opts.SetOption(solver.id(), 'Major iterations limit', int(MAJOR_ITERS))
        result = solver.Solve(prog, warm if warm is not None else prog.initial_guess(), opts)
        return self.kto.ReconstructTrajectory(result), result


def kto_candidate(robot, scene, points, *, margin_m, baseline_s, ledger=None, expired, record):
    from .optimize import _spline, _time_path, certify_curve
    if expired():
        record['stop'] = 'expired'
        return None
    pad = CertifierConfig(margin_m=margin_m).padding(robot)
    seed = _spline(robot, points, SEED_SPACING)
    P0 = fit_seed(sample_by_arclength(lambda u: seed.vector_values(u).T, m=SEED_SAMPLES), N_CP)
    problem = KTOProblem(robot, scene, points[0], points[-1], N_CP)
    problem.add_distance(np.linspace(0., 1., N_COLL+2)[1:-1], pad+CUSHION_M)
    T0 = float(baseline_s)*T0_FACTOR
    guess = BsplineTrajectory(BsplineBasis(ORDER, N_CP, KnotVectorType.kClampedUniform, 0., T0),
                              [p.reshape(-1, 1) for p in P0])
    warm = None
    for round_ in range(ROUNDS+1):
        if expired():
            record['stop'] = 'expired'
            return None
        try:
            traj, result = problem.solve(guess if warm is None else None, warm=warm)
        except KTOUnavailable:
            raise
        except RuntimeError as error:
            record['last_error'] = f'{type(error).__name__}: {str(error)[:200]}'
            record['stop'] = 'solver_error'
            return None
        record['solves'] += 1
        warm = result.GetSolution()
        if (not all(np.isfinite(c).all() for c in traj.control_points())
                or not np.isfinite(traj.end_time())):
            record['stop'] = 'nonfinite'
            return None
        candidate = to_cubic(traj, points[0] if problem.start_outside else None)
        trace = {}
        ok, _, _ = certify_curve(robot, scene, candidate, margin_m=margin_m, ledger=ledger,
                                 trace=trace)
        record['rounds'] += 1
        if ok:
            record['solver_success'] = bool(result.is_success())
            timing = {}
            timed, code = _time_path(robot, candidate, lambda: False, rungs=TOPPRA_RUNGS,
                                     trace=timing)
            if timed is None:
                record['toppra'] = code
                record['stop'] = 'toppra'
                return None
            record['toppra_rung'] = list(timing['rung'])
            return timed, code
        if expired():
            record['stop'] = 'expired'
            return None
        if round_ >= ROUNDS:
            record['stop'] = 'repair_exhausted'
            return None
        new = failing_clusters(trace['samples'], trace['free']) if 'free' in trace else []
        if not new:
            record['stop'] = 'no_clusters'
            return None
        problem.add_distance(new, pad+CUSHION_M)
    return None
