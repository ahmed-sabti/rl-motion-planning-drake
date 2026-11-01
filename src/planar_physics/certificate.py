from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import numpy as np
from pydrake.common import Parallelism
from .budget import Ledger
from .contract import DEFAULT_CONTRACT
from .clearance import exact_clearance
from .drake_model import get_checker

class EdgeStatus(IntEnum):
    FREE = 1
    BLOCKED = 0
    UNKNOWN = -1

@dataclass(frozen=True, slots=True)
class CertifierConfig:
    margin_m: float = 0.0
    numerical_slack_m: float = DEFAULT_CONTRACT.numerical_slack_m
    edge_step_rad: float = .004

    def __post_init__(self):
        if not (isfinite(self.margin_m) and self.margin_m >= 0 and isfinite(self.numerical_slack_m)
                and self.numerical_slack_m >= 0 and isfinite(self.edge_step_rad) and self.edge_step_rad > 0):
            raise ValueError("certifier margin and slack must be finite and >= 0, edge step finite and > 0")

    def padding(self, robot, moving=True):
        still = self.margin_m + self.numerical_slack_m
        return still + robot.clearance_lipschitz_m * self.edge_step_rad / 2 if moving else still

@dataclass(frozen=True, slots=True)
class Certificate:
    status: EdgeStatus

    @property
    def certified(self):
        return self.status is EdgeStatus.FREE


_SERIAL = Parallelism(1)


def edge_verdicts(robot, scene, *, cfg=CertifierConfig(), ledger: Ledger | None = None, strict=False):
    n = robot.n_joints
    lo, hi = tuple(float(x) for x in robot.lo_rad), tuple(float(x) for x in robot.hi_rad)
    step = cfg.edge_step_rad
    checkers = {}

    def free(a, b):
        try:
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if a.shape != (n,) or b.shape != (n,):
                return False
            for x, y, low, high in zip(a.tolist(), b.tolist(), lo, hi):
                if (not (isfinite(x) and isfinite(y))
                        or x < low or y < low or x > high or y > high):
                    return False
            delta = (b - a)[None, :]
            distance = float(np.sqrt(np.add.reduce(delta*delta, axis=1))[0])
            padding = cfg.padding(robot, moving=distance != 0.)
            if ledger is not None:
                ledger.charge("edge_certifications", 1)
                if ledger.deadline_exceeded():
                    return False
                ledger.charge("collision_queries", int(max(1., float(np.ceil(distance/step)))) + 1)
            key = (padding, None if scene is None else scene.identity)
            checker = checkers.get(key)
            if checker is None:
                checker = checkers[key] = get_checker(robot, scene, padding=padding, step=step)
            verdict = all(checker.CheckEdgesCollisionFree([(a, b)], parallelize=_SERIAL))
            if verdict:
                u = np.linspace(0., 1., int(max(1., float(np.ceil(distance/step)))) + 1)[:, None]
                verdict = bool(exact_clearance(robot, scene, a + u*(b - a)).min()
                               >= padding - cfg.numerical_slack_m)
            return verdict and not (ledger is not None and ledger.deadline_exceeded())
        except (RuntimeError, ValueError, OverflowError):
            if strict:
                raise
            return False
    return free


def certify_path(robot, scene, path, *, cfg=CertifierConfig(), ledger: Ledger | None = None,
                 classify_blocked=True):
    unknown = Certificate(EdgeStatus.UNKNOWN)
    points = np.asarray(path, dtype=float)
    if (points.ndim != 2 or points.shape[1] != robot.n_joints or len(points) < 2
            or not np.isfinite(points).all() or np.any(points < robot.lo_rad) or np.any(points > robot.hi_rad)):
        return unknown
    free = edge_verdicts(robot, scene, cfg=cfg, ledger=ledger)
    if all(free(a, b) for a, b in zip(points[:-1], points[1:])):
        return Certificate(EdgeStatus.FREE)
    if not classify_blocked or (ledger is not None and ledger.deadline_exceeded()):
        return unknown
    try:
        checker = get_checker(robot, scene, padding=cfg.margin_m, step=cfg.edge_step_rad)
        if not all(checker.CheckEdgesCollisionFree(list(zip(points[:-1], points[1:])),
                                                   parallelize=_SERIAL)):
            return Certificate(EdgeStatus.BLOCKED)
    except (RuntimeError, ValueError, OverflowError):
        pass
    return unknown
