import time
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray
from pydrake.trajectories import Trajectory

from planar_physics.budget import Budget, BudgetExhausted, Ledger
from planar_physics.certificate import CertifierConfig, certify_path
from planar_physics.plan import IKConfig, PlanRequest, SolveStatus
from planar_physics.rng import SeedSpec
from planar_physics.robot import Robot
from planar_physics.scene import Scene
from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.outcomes import MissionOutcome

from .limits import legality_scale
from .rollout import ExecutionReport, execute_trajectory, tick_duration
from . import optimize
from .trajectory import waypoint_trajectory, rescale_trajectory

__all__ = ["MissionRequest", "MissionReport", "run_mission", "prepare", "plan_request",
           "START_TOL_RAD", "KTO_PROFILE"]

START_TOL_RAD = 1e-9
KTO_PROFILE = 'prune-kto-v1'
PRUNED_POLYLINE = 'pruned_polyline'
PLANNER_POLYLINE = 'planner_polyline'


@dataclass(frozen=True, slots=True)
class MissionRequest:
    start_q_rad: NDArray[np.float64]
    goal_xy_m: tuple[float, float]

    seed: SeedSpec
    deadline_s: float = DEFAULT_CONTRACT.deadline_s
    clearance_margin_m: float = DEFAULT_CONTRACT.margin_m
    goal_tol_m: float = DEFAULT_CONTRACT.goal_tol_m
    hold_seconds: float = DEFAULT_CONTRACT.dwell_s

    max_settling_s: float = DEFAULT_CONTRACT.settle_s
    prune_waypoints: bool = True

    budget: Budget = field(default_factory=lambda: Budget(wall_clock_s=None))

    def __post_init__(self) -> None:
        values = [self.deadline_s, self.goal_tol_m, self.clearance_margin_m, self.hold_seconds,
                  self.max_settling_s, *np.ravel(self.start_q_rad), *np.ravel(self.goal_xy_m)]
        if (np.shape(self.goal_xy_m) != (2,) or not np.isfinite(values).all()
                or min(values[2:5]) < 0 or min(self.deadline_s, self.goal_tol_m) <= 0):
            raise ValueError('invalid mission request')


@dataclass(frozen=True, slots=True)
class MissionReport:
    outcome: MissionOutcome

    reason: str
    planning_s: float
    preparation_s: float
    execution_wall_s: float

    motion_and_hold_s: float

    request_to_completion_s: float

    accepted_trajectory: bool

    execution: ExecutionReport | None = None
    reference_trajectory: Trajectory | None = None
    optimization: str = "off"
    limits_scale: float = 1.0
    prepared_kind: str | None = None
    prepared_path_rad: NDArray[np.float64] | None = None
    preparation_ops: dict[str, int] = field(default_factory=dict)
    trajectory_search: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "outcome", MissionOutcome(self.outcome))

    @property
    def succeeded(self) -> bool:
        return self.outcome == MissionOutcome.HELD


def plan_request(robot, scene, request):
    budget = replace(request.budget, wall_clock_s=min(
        request.deadline_s, request.deadline_s if request.budget.wall_clock_s is None else request.budget.wall_clock_s))
    return PlanRequest(robot, scene, request.start_q_rad, request.goal_xy_m, request.seed,
                       budget=budget, certifier=CertifierConfig(margin_m=request.clearance_margin_m),
                       ik=IKConfig(tol_m=request.goal_tol_m, margin_m=request.clearance_margin_m))


def solve_request(robot, scene, planner, request, goals=None):
    started = time.perf_counter()
    plan = planner.solve(plan_request(robot, scene, request), goals=goals)
    diagnostic = (plan.search_stats or {}).get('graph_capture_s', 0.)
    return plan, max(time.perf_counter() - started - diagnostic, plan.wall_clock_s)


@dataclass
class Prepared:
    outcome: MissionOutcome | None = None
    reason: str = ''
    path: NDArray[np.float64] | None = None
    kind: str | None = None
    reference: Trajectory | None = None
    optimization: str = 'off'
    limits_scale: float = 1.
    search: dict = field(default_factory=dict)


def prepare(robot, scene, path, *, ledger, margin_m, dt, wall_time_s=None, prune=True, seed_entropy=0):
    started = time.perf_counter()
    cfg = CertifierConfig(margin_m=margin_m)
    done = Prepared()

    def refuse(outcome, reason):
        done.outcome, done.reason = outcome, reason
        return done
    try:
        if prune:
            from planar_physics.planners.ompl_planners import simplify_path
            simplified = simplify_path(robot, scene, path, cfg=cfg, ledger=ledger,
                                       seed_entropy=seed_entropy)
            if simplified is not None and certify_path(robot, scene, simplified, cfg=cfg,
                                                       ledger=ledger).certified:
                done.path, done.kind = simplified, PRUNED_POLYLINE
        elif certify_path(robot, scene, path, cfg=cfg, ledger=ledger).certified:
            done.path, done.kind = np.array(path, dtype=float, copy=True), PLANNER_POLYLINE
    except BudgetExhausted as error:
        return refuse(MissionOutcome.DEADLINE if ledger.deadline_exceeded()
                      else MissionOutcome.TRAJECTORY_REJECTED, f'prepare:budget:{error.field}')
    except ValueError as error:
        return refuse(MissionOutcome.TRAJECTORY_REJECTED, f'path:{error}')
    if ledger.deadline_exceeded():
        return refuse(MissionOutcome.DEADLINE, 'Path preparation exhausted the mission deadline.')
    if done.path is None:
        return refuse(MissionOutcome.TRAJECTORY_REJECTED, 'path:unknown')
    remaining = None if wall_time_s is None else max(0., wall_time_s-(time.perf_counter()-started))
    details = {}
    curve, done.optimization = optimize.optimize_trajectory(
        robot, scene, waypoint_trajectory(robot, done.path, 1.), margin_m=margin_m,
        wall_time_s=remaining, ledger=ledger, details=details)
    done.search = details.get('search', {})
    if curve is None:
        return refuse(MissionOutcome.DEADLINE if ledger.deadline_exceeded()
                      else MissionOutcome.TRAJECTORY_REJECTED, done.optimization)
    ok, why, _ = optimize.certify_curve(robot, scene, curve.path(), margin_m=margin_m)
    if not ok:
        return refuse(MissionOutcome.TRAJECTORY_REJECTED, f'curve:{why}')
    scale = float(legality_scale(robot, curve, dt))
    if not np.isfinite(scale):
        return refuse(MissionOutcome.TRAJECTORY_REJECTED, 'limits:exceeded')
    if ledger.deadline_exceeded() or (wall_time_s is not None
                                      and time.perf_counter()-started > wall_time_s):
        return refuse(MissionOutcome.DEADLINE, 'Path preparation exhausted the mission deadline.')
    done.limits_scale = scale
    done.reference = rescale_trajectory(curve, scale) if scale > 1. else curve
    return done


def run_mission(robot: Robot, scene: Scene, plant, planner, request: MissionRequest,
                goals=None) -> MissionReport:
    dt = plant.config.timestep_s
    done, execution = Prepared(), None
    plan, planning_s = solve_request(robot, scene, planner, request, goals)
    preparation_ledger = None

    def finish(outcome, reason, preparation_s=0.0, execution_wall_s=0.0, accepted=False):
        motion = execution.run.duration_s if execution is not None else 0.0
        return MissionReport(outcome=outcome, reason=reason, planning_s=planning_s,
                             preparation_s=preparation_s, execution_wall_s=execution_wall_s,
                             motion_and_hold_s=motion,
                             request_to_completion_s=planning_s+preparation_s+motion,
                             accepted_trajectory=accepted, limits_scale=done.limits_scale,
                             execution=execution,
                             reference_trajectory=done.reference, optimization=done.optimization,
                             trajectory_search=done.search, prepared_kind=done.kind,
                             prepared_path_rad=done.path,
                             preparation_ops={key: value-plan.ops.get(key, 0) for key, value in
                                 preparation_ledger.counts.items()} if preparation_ledger else {})

    if planning_s >= request.deadline_s:
        return finish(MissionOutcome.DEADLINE,
                      f"planning alone took {planning_s:.2f}s of {request.deadline_s:.2f}s")
    if plan.status is not SolveStatus.SOLVED:
        return finish(MissionOutcome.PLAN_FAILED, plan.reason or plan.status.value)
    start = np.asarray(request.start_q_rad, dtype=float)
    path = plan.waypoints_rad
    if path is None or len(path) == 0:
        return finish(MissionOutcome.PLAN_FAILED, 'no_waypoints')
    if np.max(np.abs(np.asarray(path[0], dtype=float) - start)) > START_TOL_RAD:
        return finish(MissionOutcome.PLAN_FAILED, 'start_mismatch')
    started = time.perf_counter()
    preparation_ledger = Ledger(replace(request.budget, wall_clock_s=request.deadline_s-planning_s),
                                counts=dict(plan.ops))
    done = prepare(robot, scene, path, ledger=preparation_ledger, margin_m=request.clearance_margin_m,
                   dt=dt, wall_time_s=request.deadline_s-planning_s, prune=request.prune_waypoints,
                   seed_entropy=request.seed.entropy)
    preparation_s = time.perf_counter() - started
    if done.outcome is not None:
        return finish(done.outcome, done.reason, preparation_s)
    reference = done.reference
    if np.max(np.abs(np.asarray(reference.value(0.)).ravel()-start)) > START_TOL_RAD:
        return finish(MissionOutcome.TRAJECTORY_REJECTED, 'reference:start_mismatch', preparation_s)

    computation = planning_s + preparation_s
    motion_duration = tick_duration(reference.end_time(), dt)
    dwell_duration = tick_duration(request.hold_seconds, dt)
    predicted = computation + motion_duration + dwell_duration
    if predicted > request.deadline_s:
        return finish(MissionOutcome.DEADLINE,
                      f"would need {predicted:.2f}s of {request.deadline_s:.2f}s; "
                      "refused before starting the motion",
                      preparation_s, accepted=True)

    started = time.perf_counter()
    spare = request.deadline_s - computation - motion_duration
    window = tick_duration(min(spare, dwell_duration + request.max_settling_s), dt, round_up=False)
    execution = execute_trajectory(plant, scene, reference, goal_xy_m=request.goal_xy_m,
                                   goal_tol_m=request.goal_tol_m, hold_seconds=window,
                                   required_dwell_s=request.hold_seconds)
    execution_wall_s = time.perf_counter() - started
    total = computation + execution.run.duration_s
    if total > request.deadline_s:
        return finish(MissionOutcome.DEADLINE,
                      f"overran during execution: {total:.2f}s against "
                      f"{request.deadline_s:.2f}s", preparation_s, execution_wall_s, True)
    outcome = MissionOutcome.HELD if execution.held else MissionOutcome.NOT_HELD
    return finish(outcome, execution.outcome, preparation_s, execution_wall_s, True)
