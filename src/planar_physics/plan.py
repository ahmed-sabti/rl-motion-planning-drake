from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .budget import Budget, OperationCounts
from .certificate import CertifierConfig
from .robot import Robot
from .rng import SeedSpec
from .scene import Scene

class SolveStatus(StrEnum):
    SOLVED = "solved"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"

@dataclass(frozen=True, slots=True)
class IKConfig:
    samples: int = 16

    tol_m: float = 0.015
    branch_separation_rad: float = 0.25
    max_branches: int = 8
    margin_m: float = 0.0


@dataclass(frozen=True, eq=False)
class PlanRequest:
    robot: Robot
    scene: Scene
    start_q_rad: NDArray[np.float64]
    goal: NDArray[np.float64] | tuple[float, float]
    seed: SeedSpec
    budget: Budget = field(default_factory=Budget)
    certifier: CertifierConfig = field(default_factory=CertifierConfig)
    ik: IKConfig = field(default_factory=IKConfig)


@dataclass(frozen=True, eq=False)
class PlanResult:
    status: SolveStatus
    algorithm: str
    reason: str = ""

    waypoints_rad: NDArray[np.float64] | None = None
    graph: object | None = None
    ops: OperationCounts = field(default_factory=dict)

    wall_clock_s: float = 0.0
    wall_clock_overrun_s: float = 0.0
    search_stats: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.status is SolveStatus.SOLVED
