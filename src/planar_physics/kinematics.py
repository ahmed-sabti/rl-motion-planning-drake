import numpy as np
from numpy.typing import ArrayLike, NDArray

from .robot import Robot
from .drake_model import get_model

def _q(n: int, q: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(q, dtype=np.float64)
    if a.ndim == 0 or a.shape[-1] != n:
        raise ValueError(f"expected {n} joint angles, got shape {a.shape}")
    return a

def _evaluate(robot, q, shape, query):
    model = get_model(robot)
    values = _q(robot.n_joints, q)
    out = np.full(values.shape[:-1] + shape, np.nan)
    for state, result in zip(values.reshape(-1, robot.n_joints), out.reshape((-1,) + shape)):
        if np.isfinite(state).all():
            model.plant.SetPositions(model.plant_context, state)
            result[:] = query(model)
    return out

def joint_positions(robot: Robot, q: ArrayLike) -> NDArray[np.float64]:
    def query(model):
        return np.vstack([robot.base_xy_m, *[
            model.plant.CalcPointsPositions(model.plant_context, body.body_frame(),
                [robot.lengths_m[i], 0, 0], model.plant.world_frame()).ravel()[:2]
            for i, body in enumerate(model.link_bodies)]])
    return _evaluate(robot, q, (robot.n_joints + 1, 2), query)

def tcp(robot: Robot, q: ArrayLike) -> NDArray[np.float64]:
    return _evaluate(robot, q, (2,), lambda model:
        model.plant.CalcPointsPositions(model.plant_context, model.tip_frame,
                                       [0, 0, 0], model.plant.world_frame()).ravel()[:2])

def within_limits(robot: Robot, q: ArrayLike) -> NDArray[np.bool_]:
    a = _q(robot.n_joints, q)
    return np.all((a >= robot.lo_rad) & (a <= robot.hi_rad), axis=-1)
