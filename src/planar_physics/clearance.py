import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydrake.common import Parallelism

from .budget import Ledger
from .contract import DEFAULT_CONTRACT
from .drake_model import get_checker, get_model
from .kinematics import _q
from .robot import Robot
from .scene import Scene

def _point_segment(p, a, b):
    ab, ap = b - a, p - a
    t = np.clip((ap*ab).sum(-1) / np.maximum((ab*ab).sum(-1), 1e-12), 0., 1.)
    e = ap - t[..., None]*ab
    return np.sqrt((e*e).sum(-1))

def _segment_segment(a, b, c, d):
    a, b, c, d = np.broadcast_arrays(a, b, c, d)
    p, q, r = np.stack([a, a, c, c]), np.stack([b, b, d, d]), np.stack([c, d, a, b])
    side = (q[..., 0]-p[..., 0])*(r[..., 1]-p[..., 1]) - (q[..., 1]-p[..., 1])*(r[..., 0]-p[..., 0])
    crossing = (side[0]*side[1] < 0) & (side[2]*side[3] < 0)
    return np.where(crossing, 0., _point_segment(r, p, q).min(axis=0))

def exact_clearance(robot: Robot, scene: Scene | None, q: ArrayLike) -> NDArray[np.float64]:
    states = np.asarray(q, dtype=float).reshape(-1, robot.n_joints)
    theta = np.cumsum(states, axis=-1)
    steps = np.asarray(robot.lengths_m)[:, None]*np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    base = np.broadcast_to(np.asarray(robot.base_xy_m, dtype=float), (len(states), 1, 2))
    points = np.concatenate([base, base + np.cumsum(steps, axis=1)], axis=1)
    a, b = points[:, :-1], points[:, 1:]
    radius, n = robot.link_radius_m, robot.n_joints
    pairs = [(i, j) for i in range(n) for j in range(i+2, n)] if robot.self_collision else []
    first, second, offset = [a[:, [i for i, _ in pairs]]], [b[:, [i for i, _ in pairs]]], [np.full(len(pairs), 2*radius)]
    third, fourth = [a[:, [j for _, j in pairs]]], [b[:, [j for _, j in pairs]]]
    pieces = np.empty((0, 4)) if scene is None else scene.wall_pieces_m
    if len(pieces):
        k = len(pieces)
        first.append(np.repeat(a, k, axis=1)); second.append(np.repeat(b, k, axis=1))
        third.append(np.broadcast_to(np.tile(pieces[:, :2], (n, 1)), (len(states), n*k, 2)))
        fourth.append(np.broadcast_to(np.tile(pieces[:, 2:], (n, 1)), (len(states), n*k, 2)))
        offset.append(np.full(n*k, radius + scene.wall.thickness_m/2))
    gaps = [np.full(len(states), np.inf)]
    if sum(x.shape[1] for x in first):
        gaps.append((_segment_segment(*(np.concatenate(x, axis=1) for x in (first, second, third, fourth)))
                     - np.concatenate(offset)).min(axis=1))
    return np.minimum.reduce(gaps).reshape(np.shape(q)[:-1])

def clearance(robot: Robot, scene: Scene | None, q: ArrayLike,
              ledger: Ledger | None = None) -> NDArray[np.float64]:
    model = get_model(robot, scene)
    values = _q(robot.n_joints, q)
    states = values.reshape(-1, robot.n_joints)
    if ledger is not None:
        ledger.charge("collision_queries", max(len(states), 1))
    result = np.full(len(states), np.nan)
    for k, state in enumerate(states):
        if not np.isfinite(state).all() or not np.all((state >= robot.lo_rad) & (state <= robot.hi_rad)):
            continue
        try:
            distances = model.checker.CalcRobotClearance(state, model.influence_distance_m).distances()
            if len(distances) == model.checker.MaxNumDistances() and np.isfinite(distances).all():
                result[k] = np.min(distances, initial=np.inf)
        except RuntimeError:
            pass
    return result.reshape(values.shape[:-1])

def is_free(robot: Robot, scene: Scene, q: ArrayLike, *, margin_m: float = 0.0,
            numerical_slack_m: float = DEFAULT_CONTRACT.numerical_slack_m, ledger: Ledger | None = None) -> NDArray[np.bool_]:
    values = _q(robot.n_joints, q)
    states = values.reshape(-1, robot.n_joints)
    result = np.zeros(len(states), dtype=bool)
    valid = np.isfinite(states).all(-1) & ((states >= robot.lo_rad) & (states <= robot.hi_rad)).all(-1)
    if ledger is not None:
        ledger.charge("collision_queries", max(len(states), 1))
    try:
        if valid.any():
            checker = get_checker(robot, scene, padding=margin_m + numerical_slack_m)
            result[valid] = checker.CheckConfigsCollisionFree(states[valid], parallelize=Parallelism(1))
            result[result] &= exact_clearance(robot, scene, states[result]) >= margin_m
    except RuntimeError:
        pass
    return result.reshape(values.shape[:-1])
