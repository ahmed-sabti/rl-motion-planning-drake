from pathlib import Path
import json

import numpy as np

from planar_physics.fixtures import ROBOT_EQUAL_030
from planar_physics.records import json_ready
from planar_physics.contract import VARIANT
from planar_physics.scene import Scene, WallSpec
from planar_learn.scenes import SCENE_PROFILE, SceneSpec
from planar_dynamics.mission import KTO_PROFILE

ROBOT = ROBOT_EQUAL_030
FORMAT = 'step-bank-v3'


def rebuild(payload):
    wall = payload.get('wall')
    if payload.get('obstacles'):
        raise ValueError('scenes have no circle obstacles')
    return Scene(wall=WallSpec(**wall) if wall else None)


def write_bank(output, records, *, split, margin_m, search_budget_s, planners, **fields):
    body = {'format': FORMAT, 'split': split, **fields, 'margin_m': margin_m,
            'config': {'search_budget_s': search_budget_s, 'planners': list(planners)},
            'issued': len(records), 'solved': sum(r['status'] == 'ok' for r in records),
            'records': records}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_ready(body), indent=2, allow_nan=False)+'\n')
    return body


def load_tasks(path, robot):
    body = json.loads(Path(path).read_text())
    if body.get('format') != FORMAT:
        raise ValueError('step bank format mismatch')
    if body.get('timing_profile') != KTO_PROFILE:
        raise ValueError('Only task banks with the thesis KTO references are supported')
    if body.get('robot_variant', '3r') != VARIANT:
        raise ValueError(f"bank {Path(path).name} is for the {body.get('robot_variant', '3r')} arm, not {VARIANT}")
    if VARIANT == '2r' and body.get('scene_profile') != SCENE_PROFILE:
        raise ValueError('2R training needs a bank from the current scene profile')
    usable = [r for r in body.get('records', []) if r.get('status') == 'ok']
    if not usable:
        raise ValueError(f'bank {Path(path).name} has no usable task')
    for r in usable:
        ref = r.get('reference_s')
        if (len(r.get('start', [])) != robot.n_joints or len(r.get('goal_xy', [])) != 2
                or np.asarray(r.get('goals', []), dtype=float).reshape(-1, robot.n_joints).size == 0
                or not isinstance(ref, (int, float)) or not np.isfinite(ref) or ref <= 0):
            raise ValueError(f"bank {Path(path).name}: task {r.get('seed')} does not fit the {VARIANT} arm")
    spec = SceneSpec(**body['scene_distribution'])
    tasks = [{'scene': rebuild(record['scene']),
              'start': np.asarray(record['start'], dtype=float),
              'goal_xy': np.asarray(record['goal_xy'], dtype=float),
              'goals': np.asarray(record['goals'], dtype=float),
              'reference_s': float(record['reference_s']),
              'reference_profile': body['timing_profile'],
              'seed': record['seed'],
              'scene_name': record.get('world') or ('wall' if record['scene'].get('wall') else 'free')}
             for record in usable]
    return tasks, body, spec
