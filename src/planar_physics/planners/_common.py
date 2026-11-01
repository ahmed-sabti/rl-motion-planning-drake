import numpy as np
from numpy.typing import NDArray

from ..budget import BudgetExhausted, Ledger
from ..certificate import EdgeStatus, certify_path
from ..clearance import clearance
from ..ik import solve_ik
from ..kinematics import within_limits
from ..plan import PlanRequest, PlanResult, SolveStatus

def validate_start(request: PlanRequest, ledger: Ledger) -> str:
    q = np.asarray(request.start_q_rad, dtype=np.float64)
    if q.shape != (request.robot.n_joints,) or not np.isfinite(q).all():
        return "invalid:start_shape"
    if not bool(within_limits(request.robot, q)):
        return "invalid:start_outside_limits"
    gap = float(clearance(request.robot, request.scene, q, ledger))
    if np.isnan(gap):
        return 'revalidation:start_geometry_unknown'
    if gap < request.certifier.margin_m - request.certifier.numerical_slack_m:
        return 'invalid:start_in_collision'
    if gap < request.certifier.padding(request.robot):
        return 'revalidation:start_geometry_uncertain'
    return ""


def resolve_goals(request: PlanRequest, ledger: Ledger) -> tuple[NDArray[np.float64], str]:
    n = request.robot.n_joints
    result = solve_ik(request.robot, request.scene, np.asarray(request.goal, dtype=np.float64),
                      rng=request.seed.stream("ik.samples"), seed_q=request.start_q_rad,
                      cfg=request.ik, certifier=request.certifier, ledger=ledger)
    if result.status is SolveStatus.SOLVED:
        return result.configs, ""
    return np.zeros((0, n)), result.reason or "ik:no_candidate"


def goals_for(request, ledger, goals):
    return resolve_goals(request, ledger) if goals is None else goals


def finish(request: PlanRequest, ledger: Ledger, algorithm: str, *, status: SolveStatus,
           reason: str = "", waypoints: NDArray | None = None, graph=None) -> PlanResult:
    if waypoints is not None:
        try:
            certificate = certify_path(request.robot, request.scene, waypoints,
                                       cfg=request.certifier, ledger=ledger)
            if certificate.status is not EdgeStatus.FREE:
                reason = ('budget:wall_clock_s' if ledger.deadline_exceeded()
                          else f"revalidation:{certificate.status.name.lower()}")
                status = SolveStatus.UNKNOWN
                waypoints = None
        except BudgetExhausted as exhausted:
            status, reason, waypoints = SolveStatus.UNKNOWN, f"budget:{exhausted.field}", None
    elapsed = ledger.elapsed_s()
    cap = ledger.budget.wall_clock_s
    return PlanResult(status, algorithm, reason=reason, waypoints_rad=waypoints, graph=graph,
                      ops=ledger.snapshot(), wall_clock_s=elapsed,
                      wall_clock_overrun_s=max(0., elapsed-cap) if cap is not None else 0.)
