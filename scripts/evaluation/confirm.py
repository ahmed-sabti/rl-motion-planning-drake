from dataclasses import asdict, fields as fields_of, replace
from datetime import datetime, timezone
from pathlib import Path
import argparse, faulthandler, hashlib, json, os, platform, sys, time, traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from planar_eval import methods
from planar_physics.fixtures import ROBOT_EQUAL_030 as R
from planar_physics.budget import Budget, Ledger
from planar_physics.contract import DEFAULT_CONTRACT, VARIANT
from planar_physics.kinematics import tcp
from planar_physics.planners._common import resolve_goals
from planar_physics.rng import SeedSpec
from planar_learn.scenes import SCENE_PROFILE, SceneSpec, sample_scene, sample_goal
from planar_learn.step_env import observation_fields
from planar_learn.step_policy import served_chord
from planar_dynamics.plant import PlanarPlant
from planar_dynamics.mission import MissionRequest, plan_request, run_mission
from planar_dynamics.rollout import tick_duration
from planar_eval.static_task import held_start
from planar_eval.step_bank import rebuild
from planar_eval.sac_support import load_candidate
from planar_eval.reference_metrics import measure_reference, REVERSAL_STEP_S, REVERSAL_DEADBAND_RAD_S
from planar_dynamics.trajectory import trajectory_record

DEADLINE_S = 20.
PIPELINE = 'kto-3s'
CLASSICAL = ('rrt', 'rrt_star', 'rrt_connect')
ARMS = ('candidate',)+CLASSICAL
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'SAC_TORCH_THREADS')


def candidate_hashes(candidate):
    stem = Path(candidate)
    return {name: hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
            for name, suffix in (('candidate_sha256', '.zip'), ('candidate_manifest_sha256', '.json'))}


def jsonable(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        return str(item)
    return json.loads(json.dumps(value, default=convert))


def write_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json(path, value):
    write_text(path, json.dumps(jsonable(value), indent=2))


class Rows:
    def __init__(self, path):
        self.path, self.parts = Path(path), []

    def append(self, row):
        self.parts.append('\n'.join('  '+line for line in json.dumps(jsonable(row), indent=2).splitlines()))
        write_text(self.path, '[\n' + ',\n'.join(self.parts) + '\n]')


def run_environment(stage='start'):
    import torch
    return dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), machine=platform.machine(),
                platform=platform.platform(), processor=platform.processor(), python=sys.version,
                cpu_count=os.cpu_count(), loadavg=list(os.getloadavg()),
                thread_environment={key: os.environ.get(key) for key in THREAD_VARIABLES},
                torch=torch.__version__, torch_threads=torch.get_num_threads(),
                numpy=np.__version__, argv=list(sys.argv))


def mission_request(q, goal, seed):
    return MissionRequest(q, goal, SeedSpec(seed), deadline_s=DEADLINE_S, prune_waypoints=True,
                          budget=Budget(wall_clock_s=3.))


def request_fields():
    sample = mission_request(np.zeros(3), (.5, 0.), 0)
    return jsonable({f.name: (asdict(getattr(sample, f.name)) if f.name == 'budget' else getattr(sample, f.name))
                     for f in fields_of(sample) if f.name not in ('start_q_rad', 'goal_xy_m', 'seed')})


def arm_order(i):
    names = list(ARMS)
    k = (i//2) % len(names)
    return names[k:]+names[:k]


def resolve_outside_clocks(scene, req):
    request = plan_request(R, scene, req)
    started = time.perf_counter()
    goals, reason = resolve_goals(request, Ledger(request.budget))
    return goals, reason, time.perf_counter()-started


def snopt_ready():
    try:
        from pydrake.solvers import SnoptSolver
        solver = SnoptSolver()
        return bool(solver.available() and solver.enabled())
    except Exception:
        return False


def torque_use(report):
    if report.execution is None:
        return dict(peak_torque_ratio=None, mean_torque_ratio=None)
    ratio = np.abs(np.asarray(report.execution.run.trace.tau_applied, dtype=float))/np.asarray(R.max_joint_torque_nm)
    if not ratio.size:
        return dict(peak_torque_ratio=None, mean_torque_ratio=None)
    return dict(peak_torque_ratio=float(ratio.max()), mean_torque_ratio=float(ratio.mean()))


class RecordingPlanner:
    def __init__(self, planner):
        self.inner, self.last, self.cpu = planner, None, None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def solve(self, request, goals=None):
        process, thread = time.process_time(), time.thread_time()
        try:
            self.last = self.inner.solve(request, goals=goals)
        finally:
            self.cpu = dict(planning_process_cpu_s=time.process_time()-process,
                            planning_thread_cpu_s=time.thread_time()-thread)
        return self.last

    def record(self):
        plan = self.last
        row = dict(self.cpu or dict(planning_process_cpu_s=None, planning_thread_cpu_s=None))
        if plan is None:
            return dict(row, search_stats=None, raw_waypoints=None, raw_waypoints_rad=None, plan_reason=None)
        stats, waypoints = dict(plan.search_stats or {}), plan.waypoints_rad
        return dict(row, search_stats=jsonable(stats), rungs_used=stats.get('rungs_used'),
                    rungs_available=stats.get('rungs_available'),
                    raw_waypoints_rad=None if waypoints is None else jsonable(np.asarray(waypoints, dtype=float)),
                    raw_waypoints=None if waypoints is None else int(len(waypoints)),
                    plan_status=str(plan.status), plan_reason=plan.reason,
                    plan_wall_clock_s=plan.wall_clock_s, plan_ops=jsonable(dict(plan.ops)))


def error_row(name, planner, started, error):
    label = f'error:{type(error).__name__}'
    arm = dict(held=False, outcome=label, reason=f'{label}: {str(error)[:300]}', accepted=False,
               total_s=time.perf_counter()-started, planning_s=None, prepare_s=None, motion_hold_s=None,
               points=None, optimization=None, executed=False, traceback=traceback.format_exc()[-2000:])
    if planner is not None:
        try:
            arm.update(planner.record())
        except Exception:
            pass
    print(json.dumps(dict(stage='arm-error', arm=name, error=arm['reason'])), file=sys.stderr, flush=True)
    return arm


def run_arm(name, policies, scene, plant, req, goals=None):
    started, planner = time.perf_counter(), None
    try:
        planner = RecordingPlanner(policies[name] if name in policies else methods.build(name).planner)
        report = run_mission(R, scene, plant, planner, req, goals=goals)
    except Exception as error:
        return error_row(name, planner, started, error)
    arm = dict(held=report.succeeded, outcome=str(report.outcome), reason=report.reason,
               accepted=report.accepted_trajectory, planning_s=report.planning_s,
               prepare_s=report.preparation_s, motion_hold_s=report.motion_and_hold_s,
               total_s=report.request_to_completion_s, optimization=report.optimization,
               points=None if report.prepared_path_rad is None else len(report.prepared_path_rad),
               prepared_kind=report.prepared_kind, limits_scale=report.limits_scale,
               executed=report.execution is not None,
               final_tcp_error_m=None if report.execution is None else float(report.execution.final_tcp_error_m),
               prepared_path_rad=None if report.prepared_path_rad is None
               else jsonable(np.asarray(report.prepared_path_rad, dtype=float)),
               reference_duration_s=None if report.reference_trajectory is None
               else float(report.reference_trajectory.end_time()),
               trajectory_search=jsonable(report.trajectory_search),
               preparation_ops=jsonable(report.preparation_ops),
               execution_wall_s=report.execution_wall_s, **torque_use(report), **planner.record())
    if report.execution is not None:
        arm['executed_metrics'] = jsonable(asdict(report.execution.run.metrics))
    if report.accepted_trajectory:
        arm['reference_metrics'] = measure_reference(R, report.reference_trajectory, req.goal_xy_m)
        arm['reference_record'] = jsonable(trajectory_record(report.reference_trajectory))
        dt = plant.config.timestep_s
        nominal = (report.planning_s + report.preparation_s
                   + tick_duration(report.reference_trajectory.end_time(), dt)
                   + tick_duration(req.hold_seconds, dt))
        arm['reference_nominal_total_s'] = float(nominal)
        arm['reference_within_deadline'] = bool(nominal <= req.deadline_s)
    kto = report.trajectory_search.get('kto') or {}
    arm.update(toppra_rung=kto.get('toppra_rung'), kto_solves=kto.get('solves'), kto_stop=kto.get('stop'))
    if name in policies:
        stats = arm.get('search_stats') or {}
        known = stats.get('steps') is not None and arm.get('raw_waypoints') is not None
        arm.update(chord_used=(stats['steps'] == 0 and arm['raw_waypoints'] == 2) if known else None,
                   chord_used_source='steps_and_points' if known else None)
    return arm


def evaluate_request(t, i, policies, plant):
    stamp = lambda: dict(utc=datetime.now(timezone.utc).isoformat(), loadavg=list(os.getloadavg()))
    row = dict(index=i, seed=t['seed'], kind=t['kind'], known=bool(t.get('known')), arms={}, **stamp())
    order = row['arm_order'] = arm_order(i)
    scene = rebuild(t['scene'])
    req = mission_request(np.asarray(t['start']), tuple(t['goal_xy']), t['seed'])
    if 'goals' in t:
        goals, reason, row['ik_s'] = np.asarray(t['goals'], dtype=float).reshape(-1, R.n_joints), t['ik_reason'], 0.
        row['goal_source'] = 'stored-shared-snopt-goals'
    else:
        goals, reason, row['ik_s'] = resolve_outside_clocks(scene, req)
    for name in order:
        print(json.dumps(dict(stage='arm-start', index=i, arm=name)), file=sys.stderr, flush=True)
        row['arms'][name] = run_arm(name, policies, scene, plant, req, goals=(goals, reason))
    try:
        chord = served_chord(R, scene, req.start_q_rad, goals, margin_m=req.clearance_margin_m, strict=True) is not None
    except (RuntimeError, ValueError, OverflowError) as error:
        chord, row['direct_chord_error'] = None, f'{type(error).__name__}: {error}'
    row.update(ik_goals=int(len(goals)), ik_reason=reason, direct_chord=chord, end=stamp())
    return row


def generate(a):
    tasks = []
    for i in range(a.count):
        seed = a.seed+i
        rng = np.random.default_rng(seed)
        spec = SceneSpec() if i % 2 == 0 else replace(SceneSpec(), wall_probability=1.)
        scene = sample_scene(rng, spec, R)
        try:
            goal = sample_goal(rng, spec, R, scene)
            held = held_start(seed, scene=scene)
            if float(np.linalg.norm(tcp(R, held['q'])-np.asarray(goal))) <= DEFAULT_CONTRACT.goal_tol_m:
                raise ValueError('start_in_goal')
            task = dict(seed=seed, scene=asdict(scene), start=held['q'], goal_xy=np.asarray(goal).tolist(),
                        kind='mixed' if i % 2 == 0 else 'wall', issued=True, scene_profile=SCENE_PROFILE)
        except ValueError as error:
            tasks.append(dict(seed=seed, issued=False, reason=str(error), scene_profile=SCENE_PROFILE))
            continue
        if VARIANT == '2r':
            req = mission_request(np.asarray(task['start']), tuple(task['goal_xy']), seed)
            goals, reason, _ = resolve_outside_clocks(scene, req)
            task.update(goals=np.asarray(goals).tolist(), ik_reason=reason)
        tasks.append(task)
    return tasks


def load_requests(path, seed, count):
    tasks = json.loads(Path(path).read_text())
    if len(tasks) != count or any(task['seed'] != seed+i for i, task in enumerate(tasks)):
        raise ValueError('Saved request split does not match seed and count')
    if VARIANT == '2r' and any(task.get('scene_profile') != SCENE_PROFILE or
        (task['issued'] and ('goals' not in task or 'ik_reason' not in task)) for task in tasks):
        raise ValueError('2R requests need the current scene profile and stored shared goal sets, including empty sets')
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--count', type=int, default=512)
    p.add_argument('--requests', type=Path)
    a = p.parse_args()
    if not snopt_ready():
        raise SystemExit('SNOPT is not available; the KTO curve step cannot run')
    out = Path(a.output).resolve()
    if out.exists():
        raise SystemExit('Output exists; a stage runs once')
    environment = dict(start=run_environment('start'))
    candidate = load_candidate(a.candidate)
    if VARIANT == '2r' and candidate.payload.get('training', {}).get('scene_profile') != SCENE_PROFILE:
        raise ValueError('This 2R checkpoint was trained on another scene profile')
    freeze = dict(serving='sampled-retry', pipeline=PIPELINE, paths={'candidate': a.candidate},
                  **candidate_hashes(a.candidate),
                  scene_profile=SCENE_PROFILE, scene_distribution=asdict(SceneSpec()),
                  requests_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest() if a.requests else None,
                  classical=list(CLASSICAL), arms=list(ARMS), seed=a.seed, count=a.count,
                  reference_measurement=dict(dt_s=.001, measured_after_mission=True,
                      reversals=dict(sample_dt_max_s=REVERSAL_STEP_S,
                                     deadband_rad_s=REVERSAL_DEADBAND_RAD_S,
                                     interval='entire reference', aggregation='sum over joints')),
                  deadline_s=DEADLINE_S,
                  fields=observation_fields(candidate.config), mission_request=request_fields(),
                  candidate_identity=dict(trained_timing_profile=candidate.config.timing_profile,
                                          retries=candidate.RETRIES),
                  planner_backend=methods.PLANNER_BACKEND,
                  classical_identity={name: methods.build(name).identity for name in CLASSICAL})
    plant = PlanarPlant.build(R)
    tasks = load_requests(a.requests, a.seed, a.count) if a.requests else generate(a)
    out.mkdir(exist_ok=False)
    faults = open(out/'faulthandler.log', 'w')
    faulthandler.enable(file=faults, all_threads=True)
    write_json(out/'requests.json', tasks)
    write_json(out/'freeze.json', freeze)
    write_json(out/'run-environment.json', environment)
    results = Rows(out/'results.json')
    for i, t in enumerate(tasks):
        results.append(evaluate_request(t, i, {'candidate': candidate}, plant) if t['issued'] else dict(index=i, **t))
        print(json.dumps(dict(stage='confirmation', n=i+1, of=len(tasks))), flush=True)
    environment['end'] = run_environment('end')
    write_json(out/'run-environment.json', environment)
    write_text(out/'done', 'ok\n')
    print(json.dumps(dict(stage='confirmation', n=len(tasks), complete=True)), flush=True)


if __name__ == '__main__':
    main()
