from pathlib import Path
import argparse, hashlib, sys
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE/'src'))
import numpy as np
import confirm as C
from planar_physics.contract import VARIANT
from planar_dynamics.plant import PlanarPlant
from planar_eval.sac_support import load_candidate

CHAINS = {'rrtc': ('rrt_connect', 'rrt_connect', 'rrt_connect'),
          'sac': ('candidate', 'candidate', 'rrt_connect')}
STEP = 10_000_000


def attempt(name, policies, plant, t, k):
    scene = C.rebuild(t['scene'])
    req = C.mission_request(np.asarray(t['start']), tuple(t['goal_xy']), t['seed'] + k*STEP)
    goals, reason, _ = C.resolve_outside_clocks(scene, req)
    arm = C.run_arm(name, policies, scene, plant, req, goals=(goals, reason))
    keep = ('held', 'outcome', 'reason', 'planning_s', 'prepare_s', 'motion_hold_s', 'total_s', 'points', 'executed')
    return dict(method=name, seed=t['seed'] + k*STEP, **{f: arm.get(f) for f in keep})


def chains(t, i, policies, plant):
    row = dict(index=i, seed=t['seed'], kind=t['kind'], chains={})
    for chain, names in CHAINS.items():
        tries = []
        for k, name in enumerate(names):
            tries.append(attempt(name, policies, plant, t, k))
            if tries[-1]['held']:
                break
        row['chains'][chain] = tries
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('select', 'kaskade'))
    p.add_argument('--candidate', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int); p.add_argument('--count', type=int)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--requests', type=Path)
    a = p.parse_args()
    select = a.mode == 'select'
    if not select and a.seed is None:
        p.error('kaskade needs --seed')
    from planar_eval.splits import SELECTION
    seed = SELECTION[0] if a.seed is None else a.seed
    count = (SELECTION[1] if select else 100) if a.count is None else a.count
    try:
        i0, n = map(int, a.shard.split('/'))
    except ValueError:
        p.error('--shard must look like i/n')
    if not 0 <= i0 < n or count < 1:
        p.error('need 0 <= i < n for --shard i/n and --count >= 1')
    if not C.snopt_ready():
        raise SystemExit('SNOPT is not available; the KTO curve step cannot run')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    if select:
        C.ARMS = ('candidate', 'rrt_connect')
    candidate = load_candidate(a.candidate)
    if VARIANT == '2r' and candidate.payload.get('training', {}).get('scene_profile') != C.SCENE_PROFILE:
        raise ValueError('This 2R checkpoint was trained on another scene profile')
    C.write_json(out/'freeze.json', dict(
        mode=a.mode,
        candidate=a.candidate, lineage=candidate.payload.get('training', {}).get('lineage'),
        timing_profile=candidate.config.timing_profile, seed=seed, count=count, shard=a.shard,
        arms=list(C.ARMS) if select else None, chains=None if select else CHAINS, step=STEP,
        scene_profile=C.SCENE_PROFILE,
        requests_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest() if a.requests else None,
        candidate_sha256=hashlib.sha256(Path(a.candidate).with_suffix('.zip').read_bytes()).hexdigest(),
        serving='sampled-retry', deadline_s=C.DEADLINE_S))
    plant = PlanarPlant.build(C.R)
    tasks = C.load_requests(a.requests, seed, count) if a.requests else C.generate(argparse.Namespace(seed=seed, count=count))
    if not select:
        C.write_json(out/'requests.json', tasks)
    env, rows = dict(start=C.run_environment('start')), []
    for i, t in enumerate(tasks):
        if i % n != i0 or not t['issued']:
            continue
        rows.append(C.evaluate_request(t, i, {'candidate': candidate}, plant) if select
                    else chains(t, i, {'candidate': candidate}, plant))
        C.write_json(out/'results.json', rows)
    env['end'] = C.run_environment('end')
    C.write_json(out/'run-environment.json', env)
    (out/'done').write_text('ok\n')


if __name__ == '__main__':
    main()
