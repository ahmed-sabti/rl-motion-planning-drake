import math
from dataclasses import dataclass

import numpy as np

from .contract import DEFAULT_CONTRACT
from .robot import Robot
from .scene import Scene

_EPSILON = 1e-6

@dataclass(frozen=True, slots=True)
class InfeasibilityCertificate:
    kind: str
    detail: str

def blocked_goal_certificate(robot: Robot, scene: Scene, goal_xy,
                             success_tol_m: float = DEFAULT_CONTRACT.goal_tol_m
                             ) -> InfeasibilityCertificate | None:
    goal = np.asarray(goal_xy, dtype=np.float64)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        return None
    gx, gy = float(goal[0]), float(goal[1])
    reach = robot.reach_m
    bx, by = robot.base_xy_m

    radial = math.hypot(gx - bx, gy - by) - success_tol_m
    if radial > reach + _EPSILON:
        return InfeasibilityCertificate(
            "radial_reach",
            f"The nearest point of the {success_tol_m * 100:.0f} cm goal tolerance is "
            f"{radial:.3f} m from the base; the arm reaches {reach:.2f} m.")

    pieces = np.asarray(scene.wall_pieces_m, dtype=np.float64)
    if pieces.size == 0 or not np.isfinite(pieces).all():
        return None
    x = float(pieces[0, 0])
    if x <= bx + _EPSILON or gx - success_tol_m <= x + _EPSILON:
        return None
    if not (np.all(pieces[:, 0] == x) and np.all(pieces[:, 2] == x)):
        return None

    shell = robot.link_radius_m + scene.wall.thickness_m / 2.0
    occupied = sorted((float(lo) - shell, float(hi) + shell) for lo, hi in pieces[:, [1, 3]])
    free: list[tuple[float, float]] = [(-math.inf, occupied[0][0])]
    top = occupied[0][1]
    for lo, hi in occupied[1:]:
        if lo > top:
            free.append((top, lo))
        top = max(top, hi)
    free.append((top, math.inf))

    direct_y = by + (gy - by) * (x - bx) / (gx - bx)
    required = math.inf
    for lo, hi in free:
        y = max(lo, min(hi, direct_y))
        required = min(required,
                       math.hypot(x - bx, y - by) + math.hypot(gx - x, gy - y) - success_tol_m)
    if required <= reach + _EPSILON:
        return None
    return InfeasibilityCertificate(
        "wall_crossing_reach",
        f"Every available crossing, including both outer ends, needs at least "
        f"{required:.3f} m of arm even with the {success_tol_m * 100:.0f} cm goal "
        f"tolerance; only {reach:.2f} m is available.")
