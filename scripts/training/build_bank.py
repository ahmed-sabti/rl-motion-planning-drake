from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path
import argparse, hashlib, json, os, sys, time
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path[:0] = [str(CODE/'src'), str(HERE)]
import numpy as np
from planar_eval import methods
from planar_eval.splits import TRAINING as DEFAULTS
from planar_physics.contract import DEFAULT_CONTRACT, VARIANT
from planar_physics.fixtures import ROBOT_EQUAL_030 as R
from planar_physics.budget import Budget, Ledger
from planar_physics.certificate import CertifierConfig
from planar_physics.kinematics import tcp
from planar_physics.plan import IKConfig, PlanRequest, SolveStatus
from planar_physics.planners._common import resolve_goals
from planar_physics.rng import SeedSpec
from planar_learn.scenes import SCENE_PROFILE, SceneSpec, sample_goal, sample_scene
from planar_dynamics.mission import KTO_PROFILE, prepare
from planar_eval.static_task import held_start
from planar_eval.step_bank import write_bank

def spec_for(part):
    return SceneSpec() if part == 'mixed' else replace(SceneSpec(), wall_probability=1.)


def references(i, t):
    refs = {}
    for name in (('rrt_connect', 'rrt') if i % 2 == 0 else ('rrt', 'rrt_connect')):
        req = PlanRequest(R, t['scene'], t['start'], t['goal_xy'], SeedSpec(t['seed']),
                          budget=Budget(wall_clock_s=1.), certifier=CertifierConfig(margin_m=.01),
                          ik=IKConfig(margin_m=.01, tol_m=DEFAULT_CONTRACT.goal_tol_m))
        started = time.perf_counter()
        plan = methods.build(name).planner.solve(req, goals=(np.asarray(t['goals'], dtype=float), ''))
        ref = dict(search_s=time.perf_counter()-started, total_s=None, duration_s=None, prepare_s=0.,
                   code=str(plan.status))
        if plan.status is SolveStatus.SOLVED:
            started = time.perf_counter()
            try:
                done = prepare(R, t['scene'], plan.waypoints_rad, ledger=Ledger(Budget(wall_clock_s=3.)),
                               margin_m=.01, dt=DEFAULT_CONTRACT.dt_s, wall_time_s=3.)
                curve, code = done.reference, done.optimization if done.outcome is None else done.reason
            except (ValueError, RuntimeError) as error:
                curve, code = None, f'error:{type(error).__name__}: {str(error)[:120]}'
            ref.update(code=code, prepare_s=time.perf_counter()-started)
            if curve is not None:
                ref.update(duration_s=float(curve.end_time()),
                           total_s=ref['search_s']+ref['prepare_s']+float(curve.end_time()))
        refs[name] = ref
    return refs


def task(item):
    methods.pin_threads(1)
    i, seed, part = item
    rng = np.random.default_rng(seed)
    spec = spec_for(part)
    scene = sample_scene(rng, spec, R)
    row = dict(seed=seed, part=part, scene=asdict(scene), scene_profile=SCENE_PROFILE)
    started = time.perf_counter()
    try:
        goal = sample_goal(rng, spec, R, scene)
        start = np.asarray(held_start(seed, scene=scene)['q'], dtype=float)
    except ValueError as error:
        return dict(row, status=f'start:{error}'[:120], seconds=time.perf_counter()-started)
    row.update(start=start.tolist(), goal_xy=np.asarray(goal).tolist())
    if float(np.linalg.norm(tcp(R, start)-goal)) <= DEFAULT_CONTRACT.goal_tol_m:
        return dict(row, status='start_in_goal', seconds=time.perf_counter()-started)
    req = PlanRequest(R, scene, start, goal, SeedSpec(seed), budget=Budget(wall_clock_s=None),
                      certifier=CertifierConfig(margin_m=.01),
                      ik=IKConfig(margin_m=.01, tol_m=DEFAULT_CONTRACT.goal_tol_m))
    goals, reason = resolve_goals(req, Ledger(req.budget))
    if not len(goals):
        return dict(row, status=reason or 'ik:no_candidate', seconds=time.perf_counter()-started)
    row['goals'] = np.asarray(goals).tolist()
    refs = references(i, dict(scene=scene, start=start, goal_xy=goal, goals=goals, seed=seed))
    good = {k: v for k, v in refs.items() if v['total_s'] is not None}
    row.update(classical=refs, status='strict:both_refused', seconds=time.perf_counter()-started)
    if good:
        winner = min(good, key=lambda k: good[k]['total_s'])
        row.update(status='ok', reference_s=good[winner]['total_s'], reference_planner=winner,
                   reference_code=good[winner]['code'])
    return row


def main():
    p = argparse.ArgumentParser(description='Task bank (scene, start, goal, IK goals) with classical references '
                                            'under the evaluation curve step; PLANAR2D_ROBOT selects the arm.')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--split', choices=('train', 'dev'), default='train')
    for key, value in DEFAULTS.items():
        p.add_argument('--'+key.replace('_', '-'), type=int, default=value)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Bank exists; preserve it')
    if a.split == 'dev':
        spans = lambda seed, n: set(range(seed, seed+n))
        train = spans(DEFAULTS['mixed_seed'], DEFAULTS['mixed']) | spans(DEFAULTS['wall_seed'], DEFAULTS['wall'])
        if (spans(a.mixed_seed, a.mixed) | spans(a.wall_seed, a.wall)) & train:
            raise SystemExit('A development bank needs seeds disjoint from the training bank')
    items = ([(i, a.mixed_seed+i, 'mixed') for i in range(a.mixed)]
             + [(a.mixed+i, a.wall_seed+i, 'wall') for i in range(a.wall)])
    records, started = [], time.perf_counter()
    with Pool(a.workers) as pool:
        for n, row in enumerate(pool.imap(task, items, chunksize=2), 1):
            records.append(row)
            if n % 64 == 0 or n == len(items):
                print(json.dumps(dict(stage='bank', done=n, of=len(items), usable=sum(r['status'] == 'ok' for r in records),
                                      seconds=round(time.perf_counter()-started))), flush=True)
    composition = dict(mixed=a.mixed, mixed_seed=a.mixed_seed, wall=a.wall, wall_seed=a.wall_seed if a.wall else None,
                       workers=a.workers, loadavg=os.getloadavg(), seconds=time.perf_counter()-started,
                       status={s: sum(r['status'] == s for r in records) for s in sorted({r['status'] for r in records})})
    write_bank(a.output, records, split=a.split, composition=composition, margin_m=.01, search_budget_s=1.,
               planners=('rrt', 'rrt_connect'), source='distribution', scene_profile=SCENE_PROFILE, robot_variant=VARIANT,
               robot_digest=R.digest(), scene_distribution=asdict(SceneSpec()), timing_profile=KTO_PROFILE,
               prepare_budget_s=3.,
               record_sha256=hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest(),
               reference_rule='fastest accepted of RRT and RRT-Connect under simplifyMax+KTO+TOPPRA+limits, 1 s search')
    print(json.dumps(dict(stage='complete', tasks=len(records), usable=sum(r['status'] == 'ok' for r in records),
                          composition=composition)), flush=True)


if __name__ == '__main__':
    main()
