from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import gymnasium as gym
from math import isfinite

import numpy as np

from planar_physics.certificate import CertifierConfig, edge_verdicts
from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.kinematics import tcp
from planar_physics.budget import Budget, Ledger
from planar_dynamics.mission import KTO_PROFILE, prepare

from .scenes import route

ALGORITHM, VERSION = 'rl_step_policy', 1

OBSERVATION_FIELDS = (
    'to_goal_x', 'to_goal_y',
    'to_aim_x', 'to_aim_y',
    'wall_ahead',
    'steps_left', 'stuck')


def observation_fields(config):
    from planar_physics.fixtures import ROBOT_EQUAL_030
    return tuple(f'q{k+1}' for k in range(ROBOT_EQUAL_030.n_joints)) + OBSERVATION_FIELDS


@dataclass(frozen=True)
class StepConfig:
    step_rad: float = .30
    timing_profile: str = KTO_PROFILE
    tcp_observed: bool = False
    line_search: int = 9
    horizon: int = 128
    shaping_discount: float = 1.

    progress_weight: float = 4.
    step_cost: float = .01
    blocked_cost: float = .05
    joint_clip_cost: float = .05
    success: float = 5.
    gain_scale: float = 5.
    reward_clip: float = 1.

    goal_tol_m: float = DEFAULT_CONTRACT.goal_tol_m
    hysteresis_m: float = .10
    aim_keep: float = .25
    scene_observed: bool = False

    def __post_init__(self):
        if self.timing_profile != KTO_PROFILE or self.tcp_observed or self.scene_observed:
            raise ValueError('Only the thesis KTO profile and joint-step observation are supported')


RETIRED_OPTIONS = {'adaptive_step': False, 'task_space_actions': False, 'large_first': False,
                   'backtrack_after': 0, 'escape_on_refusal': False, 'pose_potential_weight': 0.,
                   'pose_potential_kind': 'nearby', 'geometry_feature': 'wall_distance',
                   'gap_observed': False, 'omit_aim': False, 'aim_observed': True}


def sense(robot, scene, q, goal_xy, steps_left, config, committed=None, stuck=0, *, tip=None):
    tip = tcp(robot, q) if tip is None else tip
    goal = np.asarray(goal_xy, dtype=float)
    aim, length, committed = route(scene, robot, tip, goal, committed,
                                   hysteresis_m=config.hysteresis_m, keep=config.aim_keep)
    reach = robot.reach_m
    ahead = scene.wall.plane_x_m-tip[0] if scene.wall else 0.
    parts = [np.asarray(q, dtype=float)/np.pi, (goal-tip)/reach, (aim-tip)/reach,
              np.array([ahead/reach, steps_left/config.horizon,
                        min(stuck, config.horizon)/config.horizon], dtype=float)]
    return np.concatenate(parts).astype(np.float32), -length/reach, committed


def terminal_reward(robot, scene, path, *, config, reference_s, overhead_s,
                    margin_m=DEFAULT_CONTRACT.margin_m, wall_time_s, keep=None):
    path = np.asarray(path, dtype=float)
    if len(path) == 1:
        path = np.repeat(path, 2, axis=0)
    started = perf_counter()
    done = prepare(robot, scene, path, ledger=Ledger(Budget(wall_clock_s=wall_time_s)),
                   margin_m=margin_m, dt=DEFAULT_CONTRACT.dt_s, wall_time_s=wall_time_s)
    prepare_s = perf_counter()-started
    if keep is not None:
        keep['prepared'] = done
    code = done.optimization if done.outcome is None else done.reason
    if done.reference is None:
        return 0., {'code': code, 'duration_s': None, 'margin': None}
    duration = done.reference.end_time()
    margin = (reference_s-duration-overhead_s-prepare_s)/reference_s
    gain = float(np.clip(config.gain_scale*margin, -config.reward_clip, config.reward_clip))
    return config.success+gain, {'code': 'reached', 'duration_s': float(duration), 'margin': float(margin)}


@lru_cache(maxsize=None)
def _factors(n):
    return (1.,) if n == 1 else tuple(np.geomspace(1.5, .1, n))


def step_factors(config):
    return _factors(max(1, int(config.line_search)))


def action_step(action, config):
    return np.clip(np.asarray(action, dtype=float).ravel(), -1., 1.) * config.step_rad


def advance(robot, q, step, factors, free):
    here = q.tolist()
    for factor in factors:
        candidate = np.clip(q+step*factor, robot.lo_rad, robot.hi_rad)
        if not _allclose(candidate.tolist(), here) and free(q, candidate):
            return candidate
    return None


def _allclose(a, b, rtol=1e-05, atol=1e-08):
    for x, y in zip(a, b):
        if not ((abs(x-y) <= atol + rtol*abs(y) and isfinite(y)) or x == y):
            return False
    return True


def joint_clip_fraction(robot, q, displacement):
    magnitude = float(np.linalg.norm(displacement))
    if magnitude == 0.:
        return 0.
    retained = np.clip(q+displacement, robot.lo_rad, robot.hi_rad)-q
    return float(np.clip(np.linalg.norm(displacement-retained)/magnitude, 0., 1.))


class StepEnv(gym.Env):
    def __init__(self, tasks, robot, config, *,
                 margin_m=DEFAULT_CONTRACT.margin_m, seed=0, wall_time_s):
        super().__init__()
        if config.timing_profile != KTO_PROFILE or any(
                task.get('reference_profile') != KTO_PROFILE for task in tasks):
            raise ValueError('training uses the KTO profile and matching references')
        self.tasks, self.robot, self.config = list(tasks), robot, config
        self.margin_m, self.wall_time_s = margin_m, wall_time_s
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(
            -10., 10., (len(observation_fields(config)),), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (robot.n_joints,), np.float32)
        self.task, self.last = self.tasks[0], {}
        self._begin(self.tasks[0])

    def _begin(self, task):
        self.task = task
        self.q = np.asarray(task['start'], dtype=float)
        self.goal_xy = np.asarray(task['goal_xy'], dtype=float)
        self.path = [self.q.copy()]
        self.steps = self.blocked = self.stuck = 0
        self.compute_s = 0.
        self.committed = None
        self.kept = {}
        self.free = edge_verdicts(self.robot, task['scene'], cfg=CertifierConfig(margin_m=self.margin_m))
        started = perf_counter()
        observation, self.phi, self.committed = sense(
            self.robot, self.task['scene'], self.q, self.goal_xy,
            self.config.horizon, self.config, self.committed, self.stuck)
        self.compute_s += perf_counter()-started
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self._begin(self.tasks[int(self.rng.integers(len(self.tasks)))]), {}

    def reached(self):
        return float(np.linalg.norm(tcp(self.robot, self.q)-self.goal_xy)) <= self.config.goal_tol_m

    def step(self, action):
        config, scene = self.config, self.task['scene']
        previous_q = self.q
        started = perf_counter()
        step = action_step(action, config)
        candidate = advance(self.robot, self.q, step, step_factors(config), self.free)
        self.compute_s += perf_counter()-started
        moved = candidate is not None
        if moved:
            self.q = candidate
            self.path.append(self.q.copy())
            self.stuck = 0
        else:
            self.blocked += 1
            self.stuck += 1
        self.steps += 1

        reached = self.reached()
        terminated = reached or self.steps >= config.horizon
        started = perf_counter()
        observation, phi, self.committed = sense(self.robot, scene, self.q, self.goal_xy,
                                                 config.horizon-self.steps, config,
                                                 self.committed, self.stuck)
        if not terminated:
            self.compute_s += perf_counter()-started
        reward = config.progress_weight*(config.shaping_discount*phi-self.phi)-config.step_cost
        if not moved:
            reward -= config.blocked_cost
        self.phi = phi
        if config.joint_clip_cost:
            clip_fraction = joint_clip_fraction(
                self.robot, previous_q, step*step_factors(config)[0])
            clip_penalty = config.joint_clip_cost*clip_fraction
            reward -= clip_penalty
        info = {'code': 'horizon' if terminated else 'walking'}
        if reached:
            bonus, info = terminal_reward(
                self.robot, scene, np.asarray(self.path), config=config,
                reference_s=self.task['reference_s'], overhead_s=self.compute_s,
                margin_m=self.margin_m, wall_time_s=self.wall_time_s, keep=self.kept)
            reward += bonus
        self.last = {**info, 'steps': self.steps, 'blocked': self.blocked, 'reached': reached}
        return observation, float(reward), terminated, False, self.last
