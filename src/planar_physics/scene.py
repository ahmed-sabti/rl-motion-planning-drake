from dataclasses import asdict, dataclass, field
import numpy as np

@dataclass
class WallSpec:
    plane_x_m: float = 0.
    half_height_m: float = 0.
    thickness_m: float = 0.
    gap_centres_m: list[float] = field(default_factory=list)
    gap_widths_m: list[float] = field(default_factory=list)

@dataclass
class Scene:
    wall: WallSpec | None = None

    @property
    def identity(self):
        w = self.wall
        return None if w is None else (w.plane_x_m, w.half_height_m, w.thickness_m,
                                       tuple(w.gap_centres_m), tuple(w.gap_widths_m))

    def digest(self):
        from .contract import digest
        return digest(asdict(self))

    @property
    def wall_pieces_m(self):
        if self.wall is None:
            return np.empty((0, 4))
        w = self.wall
        c, widths = np.asarray(w.gap_centres_m, dtype=float), np.asarray(w.gap_widths_m, dtype=float)
        lows, highs = np.r_[-w.half_height_m, c + widths/2], np.r_[c - widths/2, w.half_height_m]
        return np.array([[w.plane_x_m, lo, w.plane_x_m, hi] for lo, hi in zip(lows, highs)])
