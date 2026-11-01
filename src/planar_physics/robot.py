from dataclasses import asdict, dataclass
from functools import cached_property
import numpy as np
from .contract import ASSETS, SCENARIO, digest

PARAMETERS = frozenset((
    'n_joints', 'lengths_m', 'reach_m', 'lo_rad', 'hi_rad', 'joint_limits_rad',
    'max_joint_speed_rad_s', 'max_joint_torque_nm', 'joint_viscous_damping_nms_rad',
    'link_masses_kg', 'link_radius_m', 'clearance_lipschitz_m'))

def _readonly(value):
    a = np.asarray(value, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)

@dataclass(frozen=True, eq=False)
class Robot:
    urdf: str = SCENARIO.robot
    base_xy_m: tuple[float, ...] = tuple(SCENARIO.base_xy_m)
    gravity_m_s2: tuple[float, ...] = tuple(SCENARIO.gravity_m_s2)
    max_joint_torque_rate_nm_s: tuple[float, ...] = tuple(SCENARIO.contract.torque_rate_nm_s)
    self_collision: bool = True

    def __post_init__(self):
        object.__setattr__(self, "urdf", str(self.urdf))
        for name in ("base_xy_m", "gravity_m_s2", "max_joint_torque_rate_nm_s"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "_urdf_text", self.urdf_path.read_text())
        fields = asdict(self)
        if not fields.pop("self_collision"):
            fields["self_collision"] = False
        object.__setattr__(self, "_identity", digest([self._urdf_text, fields]))

    @property
    def urdf_path(self):
        return ASSETS / self.urdf

    def digest(self):
        return self._identity

    def __hash__(self):
        return hash(self._identity)

    def __eq__(self, other):
        return isinstance(other, Robot) and self._identity == other._identity

    @cached_property
    def parameters(self):
        from .drake_model import get_model
        return get_model(self).parameters

    def __getattr__(self, name):
        if name not in PARAMETERS:
            raise AttributeError(name)
        return getattr(self.parameters, name)
