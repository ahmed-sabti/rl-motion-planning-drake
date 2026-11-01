from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import SnoptSolver, SolverOptions
from .budget import BudgetExhausted, Ledger
from .clearance import is_free
from .certificate import CertifierConfig
from .drake_model import get_model
from .kinematics import tcp, within_limits
from .plan import IKConfig, SolveStatus
from .robot import Robot
from .scene import Scene

@dataclass(frozen=True, eq=False)
class IKResult:
    status: SolveStatus
    configs: NDArray[np.float64]
    reason: str = ""

def solve_ik(robot: Robot, scene: Scene, target_xy: ArrayLike, *,
             rng: np.random.Generator, seed_q: ArrayLike | None = None,
             cfg: IKConfig = IKConfig(), certifier: CertifierConfig = CertifierConfig(),
             ledger: Ledger | None = None) -> IKResult:
    target = np.asarray(target_xy, dtype=float)
    n = robot.n_joints
    target_tol = cfg.tol_m / 2
    margin = max(cfg.margin_m, certifier.margin_m)
    goal_margin = margin + robot.clearance_lipschitz_m * certifier.edge_step_rad / 2
    anchor = np.zeros(n) if seed_q is None else np.asarray(seed_q, dtype=float)
    def failure(reason, status=SolveStatus.UNKNOWN):
        return IKResult(status, np.empty((0, n)), reason)
    if target.shape != (2,) or not np.isfinite(target).all():
        return failure("invalid:target", SolveStatus.INVALID_INPUT)
    if anchor.shape != (n,) or not np.isfinite(anchor).all():
        return failure("invalid:seed", SolveStatus.INVALID_INPUT)
    expired = lambda: ledger is not None and ledger.deadline_exceeded()
    try:
        if expired():
            return failure("deadline")
        if (seed_q is not None and within_limits(robot, anchor)
                and float(np.linalg.norm(tcp(robot, anchor) - target)) <= target_tol
                and is_free(robot, scene, anchor, margin_m=margin,
                            numerical_slack_m=certifier.numerical_slack_m, ledger=ledger)):
            return failure("deadline") if expired() else IKResult(SolveStatus.SOLVED, anchor[None, :].copy())
        model = get_model(robot, scene)
        context = model.diagram.CreateDefaultContext()
        ik = InverseKinematics(model.plant, model.plant.GetMyMutableContextFromRoot(context))
        ik.AddPointToPointDistanceConstraint(model.tip_frame, np.zeros(3),
            model.plant.world_frame(), [*target, 0.], 0., target_tol)
        ik.AddMinimumDistanceLowerBoundConstraint(goal_margin + certifier.numerical_slack_m + 1e-6)
        cost = ik.prog().AddQuadraticErrorCost(np.eye(n), anchor, ik.q()).evaluator()
        solver, options = SnoptSolver(), SolverOptions()
        if not solver.available() or not solver.enabled():
            return failure("ik:solver_unavailable")
        options.SetOption(solver.solver_id(), "Major iterations limit", 100)
        options.SetOption(solver.solver_id(), "Iterations limit", 500)
        options.SetOption(solver.solver_id(), "Major feasibility tolerance", 1e-7)
        candidates, stalled, threw = [], 0, 0
        for attempt in range(cfg.samples):
            if stalled >= 4 and len(candidates) >= 2 and attempt > n:
                break
            if expired():
                return failure("deadline")
            seed = np.clip(anchor, robot.lo_rad, robot.hi_rad)
            if 0 < attempt <= n:
                seed[attempt-1] = robot.lo_rad[attempt-1] + robot.hi_rad[attempt-1] - seed[attempt-1]
            elif attempt > n:
                seed = rng.uniform(robot.lo_rad, robot.hi_rad)
            ik.prog().SetInitialGuess(ik.q(), seed)
            cost.UpdateCoefficients(2 * np.eye(n), -2 * seed)
            stalled += 1
            try:
                result = solver.Solve(ik.prog(), None, options)
            except RuntimeError:
                threw += 1
                continue
            if expired():
                return failure("deadline")
            if not result.is_success():
                continue
            q = result.GetSolution(ik.q())
            if not np.isfinite(q).all() or not within_limits(robot, q):
                continue
            residual = float(np.linalg.norm(tcp(robot, q) - target))
            if not np.isfinite(residual) or residual > cfg.tol_m or not is_free(
                    robot, scene, q, margin_m=goal_margin,
                    numerical_slack_m=certifier.numerical_slack_m, ledger=ledger):
                continue
            if all(np.abs(q - previous).max() > cfg.branch_separation_rad for previous in candidates):
                candidates.append(q.copy())
                stalled = 0
                if len(candidates) >= cfg.max_branches:
                    break
        if expired():
            return failure("deadline")
        if candidates:
            order = np.argsort(np.abs(np.asarray(candidates) - anchor).sum(-1), kind="stable")
            return IKResult(SolveStatus.SOLVED, np.asarray(candidates)[order])
    except BudgetExhausted as exhausted:
        return failure(f"budget:{exhausted.field}")
    except RuntimeError:
        return failure("ik:native_failure")
    return failure("ik:native_failure" if threw else "ik:no_candidate")
