from time import perf_counter, process_time, thread_time
import numpy as np
from pydrake.common import Parallelism
from pydrake.multibody.optimization import CalcGridPointsOptions, Toppra
from pydrake.trajectories import BezierCurve, PathParameterizedTrajectory, PiecewisePolynomial
from planar_physics.certificate import CertifierConfig
from planar_physics.clearance import exact_clearance
from planar_physics.budget import BudgetExhausted
from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.drake_model import get_checker, get_model

from . import kto as K


def max_degree(curve):
    degree = getattr(curve, 'getSegmentPolynomialDegree', None)
    if degree is None:
        return None
    return max(degree(index, row, 0) for index in range(curve.get_number_of_segments())
               for row in range(curve.rows()))


def _leaves_joint_range(controls, lo_rad, hi_rad):
    controls = np.asarray(controls, dtype=float).T
    first, second, third = (controls[:, 1]-controls[:, 0], controls[:, 2]-controls[:, 1],
                            controls[:, 3]-controls[:, 2])
    quad, lin, const = first-2.*second+third, 2.*(second-first), first
    for joint in range(len(controls)):
        candidates = [0., 1.]
        a2, b2, c2 = quad[joint], lin[joint], const[joint]
        if a2 == 0.:
            if b2 != 0.:
                candidates.append(-c2/b2)
        else:
            discriminant = b2*b2 - 4.*a2*c2
            if discriminant >= 0.:
                q = -.5*(b2 + np.copysign(np.sqrt(discriminant), b2))
                candidates.append(q/a2)
                if q != 0.:
                    candidates.append(c2/q)
        for s in candidates:
            if not 0. <= s <= 1.:
                continue
            rest = 1.-s
            value = (rest**3*controls[joint, 0] + 3.*rest*rest*s*controls[joint, 1]
                     + 3.*rest*s*s*controls[joint, 2] + s**3*controls[joint, 3])
            if value < lo_rad[joint] or value > hi_rad[joint]:
                return True
    return False


def certify_curve(robot, scene, curve, *, margin_m, max_samples=20000, ledger=None, trace=None):
    degree = max_degree(curve)
    if degree is None or degree > 3:
        return False, f'unsupported_degree:{degree}', 0
    cfg = CertifierConfig(margin_m=margin_m)
    knots = list(curve.get_segment_times())
    pieces = [curve.slice(index, 1) for index in range(curve.get_number_of_segments())]
    samples = []
    for index, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        piece = pieces[index]
        q, v = piece.vector_values([a, b]), piece.MakeDerivative().vector_values([a, b])
        if index+1 < len(pieces) and np.max(np.abs(q[:, 1]-pieces[index+1].value(b).ravel())) > 1e-9:
            return False, 'discontinuous', 0
        bezier = BezierCurve(a, b, np.column_stack((q[:, 0], q[:, 0]+v[:, 0]*(b-a)/3,
                                                   q[:, 1]-v[:, 1]*(b-a)/3, q[:, 1])))
        controls = bezier.control_points()
        if ((np.any(controls.T < robot.lo_rad) or np.any(controls.T > robot.hi_rad))
                and _leaves_joint_range(controls.T, robot.lo_rad, robot.hi_rad)):
            return False, 'joint_limits', 0
        speed = np.linalg.norm(bezier.MakeDerivative().control_points(), axis=0).max()
        count = max(1, int(np.ceil((b-a)*speed/cfg.edge_step_rad)))
        samples.extend(np.linspace(a, b, count+1)[:-1])
    samples.append(knots[-1])
    if len(samples) > max_samples:
        return False, 'budget', len(samples)
    if ledger is not None:
        ledger.charge('collision_queries', len(samples))
    checker = get_checker(robot, scene, padding=cfg.padding(robot), step=cfg.edge_step_rad)
    try:
        free = np.asarray(checker.CheckConfigsCollisionFree(curve.vector_values(samples).T,
                                                            parallelize=Parallelism(1)), dtype=bool)
    except RuntimeError:
        free = None
    if trace is not None:
        trace.update(samples=np.asarray(samples),
                     free=np.zeros(len(samples), bool) if free is None else free)
    if free is None:
        return False, 'indeterminate', len(samples)
    if free.all():
        free &= exact_clearance(robot, scene, curve.vector_values(samples).T) >= cfg.padding(robot) - cfg.numerical_slack_m
    return bool(free.all()), '' if free.all() else 'collision', len(samples)


LIMIT_EPS = 2e-3


def _spline(robot, points, spacing=None):
    dense = points if spacing is None else np.vstack([
        np.linspace(a, b, max(1, int(np.ceil(np.linalg.norm(b-a)/spacing))), endpoint=False)
        for a, b in zip(points[:-1], points[1:])] + [points[-1:]])
    knots = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    knots /= knots[-1]
    return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        knots, dense.T, np.zeros((robot.n_joints, 1)), np.zeros((robot.n_joints, 1)))


def _time_path(robot, path, expired, rungs=K.TOPPRA_RUNGS, trace=None):
    plant = get_model(robot).plant
    caps = np.asarray(robot.max_joint_torque_nm)
    knots = np.asarray(path.get_segment_times())
    for density, relaxation in rungs:
        if expired():
            return None, 'budget'
        grid = np.unique(np.r_[knots, Toppra.CalcGridPoints(path, CalcGridPointsOptions(min_points=density))])
        grid = np.r_[grid[np.r_[True, np.diff(grid) > 1e-7]][:-1], knots[-1]]
        timing = Toppra(path, plant, grid)
        timing.set_constraint_relaxation(relaxation)
        timing.AddJointVelocityLimit(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
        timing.AddJointAccelerationLimit(plant.GetAccelerationLowerLimits(), plant.GetAccelerationUpperLimits())
        timing.AddJointTorqueLimit(-caps, caps)
        scaling = timing.SolvePathParameterization(0., 0.)
        if scaling is not None and _scaling_in_domain(scaling, path):
            if trace is not None:
                trace['rung'] = (density, relaxation)
            return PathParameterizedTrajectory(path, scaling), 'optimized_relaxed' if relaxation else 'optimized'
    return None, 'solver'


def _scaling_in_domain(scaling, path, tolerance=1e-9):
    s = np.asarray(scaling.vector_values(scaling.get_segment_times()), dtype=float).ravel()
    return bool(np.isfinite(s).all() and np.isfinite(scaling.end_time())
                and s.min() >= path.start_time()-tolerance and s.max() <= path.end_time()+tolerance)


def _support_points(robot, baseline):
    points = baseline.vector_values(baseline.get_segment_times()).T
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]]
    if len(points) > 2:
        points = np.vstack([points[:1],
                            np.clip(points[1:-1], robot.lo_rad+LIMIT_EPS, robot.hi_rad-LIMIT_EPS),
                            points[-1:]])
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]]
    return points


def _kto_refusal(record):
    if record['deadline']:
        return 'deadline'
    for key in ('stationary', 'nonfinite'):
        if record.get('stop') == key:
            return key
    if 'error' in record:
        return str(record['error']).split(':')[0]
    if record.get('stop') == 'toppra':
        return 'toppra'
    if record.get('stop') == 'solver_error':
        return str(record.get('last_error', 'RuntimeError')).split(':')[0]
    return 'no_certified_curve'


def optimize_trajectory(robot, scene, baseline, *, margin_m, wall_time_s=None, ledger=None,
                        details=None):
    started, started_cpu, started_thread = perf_counter(), process_time(), thread_time()
    record = {'rounds': 0, 'solves': 0, 'deadline': False, 'toppra_rung': None}
    unlimited = wall_time_s is None or wall_time_s == float('inf')

    def expired():
        passed = ((not unlimited and perf_counter()-started >= wall_time_s)
                  or (ledger is not None and ledger.deadline_exceeded()))
        record['deadline'] = record['deadline'] or passed
        return record['deadline']

    result = None
    if not unlimited and not (np.isfinite(wall_time_s) and wall_time_s > 0):
        record['deadline'] = True
    elif not expired():
        points = _support_points(robot, baseline)
        if len(points) < 2 or baseline.end_time() <= DEFAULT_CONTRACT.dt_s:
            record['stop'] = 'stationary'
        else:
            try:
                result = K.kto_candidate(robot, scene, points, margin_m=margin_m,
                                         baseline_s=baseline.end_time(), ledger=ledger,
                                         expired=expired, record=record)
            except BudgetExhausted as error:
                record['error'] = f'budget:{getattr(error, "field", error)}'
            except (RuntimeError, ValueError, OverflowError) as error:
                record['error'] = f'{type(error).__name__}: {str(error)[:200]}'
    record['kto_wall_s'] = perf_counter()-started
    record['kto_cpu_s'] = thread_time()-started_thread
    curve = None
    if result is not None:
        curve, code = result
        if not np.isfinite(curve.end_time()):
            record['stop'], curve = 'nonfinite', None
    if details is not None:
        details['search'] = {'winner': 'kto' if curve is not None else None, 'kto': record,
                             'elapsed_s': perf_counter()-started, 'cpu_s': process_time()-started_cpu}
    if curve is None:
        return None, f'rejected:kto:{_kto_refusal(record)}'
    return curve, f'{code}:kto'
