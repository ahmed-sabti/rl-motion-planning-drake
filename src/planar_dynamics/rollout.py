from dataclasses import dataclass
import numpy as np
from pydrake.systems.framework import EventStatus
from planar_physics.clearance import clearance, exact_clearance
from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.drake_model import get_checker, get_model
from planar_physics.kinematics import tcp, within_limits
from .metrics import motion_series

@dataclass(frozen=True)
class Trace:
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau_applied: np.ndarray
    entry_torque_nm: np.ndarray

    @property
    def sample_torque(self):
        return np.vstack((self.entry_torque_nm, self.tau_applied))

    @property
    def duration_s(self):
        return float(self.t[-1] - self.t[0])

    @staticmethod
    def from_log(plant, start):
        log = plant.log()
        times, data = log.sample_times(), log.data().T
        indices = np.flatnonzero(times >= start - 1e-10)
        indices = indices[np.r_[True, np.diff(times[indices]) > 1e-10]]
        times, data = times[indices].copy(), data[indices].copy()
        n = plant.robot.n_joints
        return Trace(times, data[:, :n], data[:, n:2*n], data[1:, 2*n:3*n], data[0, 2*n:3*n])

@dataclass(frozen=True)
class RunMetrics:
    min_clearance_m: float
    collided: bool
    left_joint_limits: bool
    peak_jerk_rad_s3: float
    jerk_rms_rad_s3: float
    distance_m: float
    peak_joint_speed_rad_s: np.ndarray
    speed_exceeded: bool


@dataclass(frozen=True)
class RunResult:
    trace: Trace
    metrics: RunMetrics
    termination: str
    series: dict

    @property
    def duration_s(self):
        return self.trace.duration_s


def rollout_padding(robot, timestep_s):
    v_max = np.asarray(get_model(robot).plant.GetVelocityUpperLimits(), dtype=float)
    return robot.clearance_lipschitz_m*float(np.linalg.norm(v_max))*timestep_s/2.


def result(plant, scene, trace, termination="horizon"):
    robot = plant.robot
    native = np.asarray(clearance(robot, scene, trace.q))
    gaps = np.minimum(native, exact_clearance(robot, scene, trace.q))
    series = motion_series(robot, trace.t, trace.q, trace.dq)
    peak_joint = np.max(np.abs(trace.dq), axis=0) if len(trace.dq) else np.zeros(robot.n_joints)
    speed_limit = np.asarray(robot.max_joint_speed_rad_s, dtype=float)
    metrics = RunMetrics(float(gaps.min()), bool(np.any(native < 0)),
                         not bool(within_limits(robot, trace.q).all()),
                         float(np.max(series["jerk"])), float(series["jrms"][-1]),
                         float(series["distance"][-1]), peak_joint, bool(np.any(peak_joint > speed_limit)))
    if metrics.left_joint_limits:
        termination = "joint_limits"
    elif metrics.collided:
        termination = "collision"
    elif np.isnan(metrics.min_clearance_m):
        termination = "geometry_unknown"
    return RunResult(trace, metrics, termination, series)


def tick_duration(seconds, dt, *, round_up=True):
    ticks = seconds / dt
    return float((np.ceil(ticks - 1e-10) if round_up else np.floor(ticks + 1e-10)) * dt)


def run(plant, scene, duration_s, *, reference=None, terminal=None):
    if not np.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration must be finite and non-negative")
    start = plant.time_s
    if reference is not None:
        plant.set_reference(reference)
    reason = "horizon"
    last_observed = -np.inf
    checker = get_checker(plant.robot, scene,
                          padding=DEFAULT_CONTRACT.numerical_slack_m
                          + rollout_padding(plant.robot, plant.config.timestep_s))
    low, high = plant.robot.lo_rad, plant.robot.hi_rad
    dt, lipschitz = plant.config.timestep_s, plant.robot.clearance_lipschitz_m
    covered = float(np.linalg.norm(get_model(plant.robot).plant.GetVelocityUpperLimits()))*dt
    previous = None
    def monitor(context):
        nonlocal reason, last_observed, previous
        now = float(context.get_time())
        if now <= last_observed + 1e-12:
            return EventStatus.Succeeded()
        last_observed = now
        q, dq = plant.q, plant.dq
        reason = "joint_limits" if not np.all((q >= low) & (q <= high)) else "horizon"
        if reason == "horizon":
            try:
                if not checker.CheckConfigCollisionFree(q):
                    reason = "geometry_uncertain"
                elif previous is not None:
                    travel = max(float(np.linalg.norm(q-previous[0])),
                                 dt*max(float(np.linalg.norm(previous[1])), float(np.linalg.norm(dq))))
                    need = DEFAULT_CONTRACT.numerical_slack_m + lipschitz*travel/2.
                    if travel > covered and not float(np.min(clearance(plant.robot, scene,
                                                                        np.stack([q, previous[0]])))) >= need:
                        reason = "geometry_uncertain"
            except RuntimeError:
                reason = "geometry_unknown"
        previous = (np.array(q, dtype=float), np.array(dq, dtype=float))
        if reason == "horizon" and terminal is not None:
            reason = terminal(plant, now - start) or reason
        return EventStatus.Succeeded() if reason == "horizon" else EventStatus.ReachedTermination(plant.native.diagram, reason)
    plant.simulator.set_monitor(monitor)
    monitor(plant.root_context)
    if reason == "horizon" and duration_s > 0:
        plant.simulator.AdvanceTo(tick_duration(start + duration_s, plant.config.timestep_s))
    plant.simulator.clear_monitor()
    log = plant.log()
    if not len(log.sample_times()) or abs(log.sample_times()[-1] - plant.time_s) > 1e-10:
        plant.native.diagram.ForcedPublish(plant.root_context)
    return result(plant, scene, Trace.from_log(plant, start), reason)

class Dwell:
    def __init__(self, qualifies, required_s):
        self.qualifies, self.required_s = qualifies, required_s
        self.started_s = None

    def __call__(self, plant, elapsed_s):
        if not self.qualifies(plant):
            self.started_s = None
            return None
        if self.started_s is None:
            self.started_s = elapsed_s
        if elapsed_s - self.started_s >= self.required_s - 1e-12:
            return "dwell"


@dataclass(frozen=True)
class ExecutionReport:
    run: RunResult
    held: bool
    outcome: str
    final_tcp_error_m: float


def execute_trajectory(plant, scene, trajectory, *, goal_xy_m=None,
                       hold_seconds=DEFAULT_CONTRACT.dwell_s, required_dwell_s=None,
                       goal_tol_m=DEFAULT_CONTRACT.goal_tol_m,
                       hold_speed_rad_s=DEFAULT_CONTRACT.hold_speed_rad_s):
    required = hold_seconds if required_dwell_s is None else required_dwell_s
    robot = plant.robot
    goal = (tcp(robot, trajectory.value(trajectory.end_time()).ravel()) if goal_xy_m is None
            else np.asarray(goal_xy_m, dtype=float))
    plant.reset(trajectory.value(0.).ravel(), np.zeros(robot.n_joints))
    motion_end = tick_duration(trajectory.end_time(), plant.config.timestep_s)
    dwell = Dwell(lambda live: live.time_s >= motion_end-1e-12
                  and np.linalg.norm(tcp(robot, live.q)-goal) <= goal_tol_m
                  and np.max(np.abs(live.dq)) <= hold_speed_rad_s, required)
    combined = run(plant, scene, motion_end+hold_seconds, reference=trajectory, terminal=dwell)
    final_error = float(np.linalg.norm(tcp(robot, plant.q)-goal))
    reached = final_error <= goal_tol_m
    held = combined.termination == "dwell"
    outcome = "held" if held else combined.termination if combined.termination != "horizon" else "reached_not_held" if reached else "missed"
    return ExecutionReport(combined, held, outcome, final_error)
