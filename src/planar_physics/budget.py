import time
from dataclasses import dataclass, field

OperationCounts = dict[str, int]


class BudgetExhausted(Exception):
    def __init__(self, field_name: str) -> None:
        super().__init__(f"budget exhausted: {field_name}")
        self.field = field_name


@dataclass(frozen=True, slots=True)
class Budget:
    edge_certifications: int = 20_000
    wall_clock_s: float | None = 30.0

    def __post_init__(self):
        if self.edge_certifications is not None and self.edge_certifications < 0:
            raise ValueError("edge_certifications must be >= 0")
        if self.wall_clock_s is not None and not self.wall_clock_s >= 0:
            raise ValueError("wall_clock_s must be >= 0 or None")


@dataclass(slots=True)
class Ledger:
    budget: Budget = field(default_factory=Budget)
    counts: OperationCounts = field(default_factory=dict)
    started_s: float = field(default_factory=time.monotonic)

    def charge(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n
        cap = getattr(self.budget, name, None)
        if cap is not None and self.counts[name] > cap:
            raise BudgetExhausted(name)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_s

    def deadline_exceeded(self) -> bool:
        cap = self.budget.wall_clock_s
        return cap is not None and self.elapsed_s() > cap

    def snapshot(self) -> OperationCounts:
        return dict(self.counts)
