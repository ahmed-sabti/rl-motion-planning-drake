from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math
import os
from pydrake.common.yaml import yaml_load_typed
from .scene import Scene

ASSETS = Path(__file__).with_name('assets')

def digest(value, *, namespace='planar-v5'):
    return hashlib.sha256((namespace + json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)).encode()).hexdigest()

@dataclass
class Contract:
    margin_m: float = 0.
    numerical_slack_m: float = 0.
    goal_tol_m: float = 0.
    dwell_s: float = 0.
    settle_s: float = 0.
    hold_speed_rad_s: float = 0.
    dt_s: float = 0.
    dense_dt_s: float = 0.
    deadline_s: float = 0.
    torque_rate_nm_s: list[float] = field(default_factory=list)
    kp: float = 0.
    kd: float = 0.

@dataclass
class Scenario:
    robot: str = ''
    base_xy_m: list[float] = field(default_factory=list)
    gravity_m_s2: list[float] = field(default_factory=list)
    contract: Contract = field(default_factory=Contract)
    goals: dict[str, list[float]] = field(default_factory=dict)
    scenes: dict[str, Scene | None] = field(default_factory=dict)
    protocol: str = ''

def load_scenario(path=ASSETS / 'scenario.yaml'):
    scenario = yaml_load_typed(schema=Scenario, filename=str(path))
    values = asdict(scenario.contract).values()
    if any(not math.isfinite(x) or x <= 0 for value in values for x in (value if isinstance(value, list) else [value])):
        raise ValueError('Scenario contract values must be finite and positive')
    return scenario

VARIANTS = {'3r': 'scenario.yaml', '2r': 'scenario-2r.yaml'}
VARIANT = os.environ.get('PLANAR2D_ROBOT', '3r')
if VARIANT not in VARIANTS:
    raise ValueError(f'PLANAR2D_ROBOT must be one of {sorted(VARIANTS)}')
SCENARIO = load_scenario(ASSETS / VARIANTS[VARIANT])
DEFAULT_CONTRACT = SCENARIO.contract

