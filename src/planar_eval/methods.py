from dataclasses import dataclass

from planar_physics.planners import ompl_planners
from planar_physics.planners.ompl_planners import KINDS

PLANNER_BACKEND = 'ompl'
SAC_NAMES = ('rl_sac',)
METHOD_NAMES = (*KINDS, *SAC_NAMES)


@dataclass
class Method:
    name: str
    planner: object
    identity: dict


class Capture:
    def __init__(self, planner, *, collect_graph=False):
        self.inner, self.result = planner, None
        if collect_graph and isinstance(planner, ompl_planners.OmplPlanner):
            from dashboard.search_view import planner_graph
            planner.graph_capture = planner_graph

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def solve(self, request, goals=None):
        self.result = self.inner.solve(request, goals=goals)
        return self.result


def pin_threads(threads=1):
    import torch
    torch.set_num_threads(threads)


def build(name, *, checkpoint=None):
    learned = name in SAC_NAMES
    if name not in METHOD_NAMES or learned != (checkpoint is not None):
        raise ValueError('Choose an active method; only a learned method requires a checkpoint')
    identity = {'method': name}
    if learned:
        from planar_eval.sac_support import load_candidate
        planner = load_candidate(checkpoint)
        payload = planner.payload
        training = payload.get('training') or {}
        identity.update(checkpoint=str(checkpoint), parameters=int(payload.get('parameters') or 0),
                        attempts=planner.RETRIES, training_seed=training.get('seed'),
                        training_stage=training.get('stage'))
        settings = payload.get('config')
    else:
        planner = ompl_planners.OmplPlanner(name)
        settings = dict(planner.cfg)
        identity['attempts'] = 1
    identity.update(algorithm=planner.name, algorithm_version=planner.version,
                    settings=settings, planner_backend=PLANNER_BACKEND)
    return Method(name, planner, identity)
