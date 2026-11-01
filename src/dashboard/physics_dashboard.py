import argparse
import fcntl
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, replace
from functools import cache, lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
import re
import shutil
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from pydrake.common import FindResourceOrThrow

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from planar_physics.contract import DEFAULT_CONTRACT, VARIANT
from planar_physics.records import json_ready
from planar_dynamics.mission import MissionRequest, plan_request, run_mission
from planar_dynamics.plant import PlanarPlant, PlantConfig
from planar_dynamics.rollout import run, execute_trajectory
from planar_physics.budget import Budget, Ledger
from planar_physics.certificate import CertifierConfig
from planar_physics.planners._common import resolve_goals
from planar_physics.feasibility import blocked_goal_certificate
from planar_physics.rng import SeedSpec
from planar_physics.clearance import clearance, exact_clearance, is_free
from planar_physics.fixtures import ROBOT_EQUAL_030, SCENES
from planar_eval.methods import Capture, build
from planar_eval.reference_metrics import joint_reversals, REVERSAL_STEP_S
from planar_physics.kinematics import joint_positions, tcp, within_limits
from dashboard.search_view import project_search_graph

from planar_eval.static_task import held_start
from dashboard.fixed_goals import GOALS, goal_names
from planar_eval.splits import SELECTION, FINAL
from planar_eval.step_bank import rebuild
from planar_learn.scenes import SCENE_PROFILE, SceneSpec, sample_goal, sample_scene

ROBOT = ROBOT_EQUAL_030
LANES = ("rrt", "rrt_star", "rrt_connect", "policy")
MODES = {"rrt": "RRT", "rrt_star": "RRT*", "rrt_connect": "RRT-Connect", "policy": "Policy"}
EVAL_ORDER = ("policy", "rrt", "rrt_star", "rrt_connect")
PLANNING_S, DEADLINE_S = 3., 20.
DRAKE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drake")
START_LOCK = threading.Lock()
RUN_MARK = ROOT / "runs" / f"timed-run-{VARIANT}.json"

@contextmanager
def machine_lock():
    RUN_MARK.parent.mkdir(parents=True, exist_ok=True)
    with START_LOCK, open(RUN_MARK.parent / ".dashboard.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def timed_run_elsewhere():
    for mark in (ROOT / "runs").glob("timed-run-*.json"):
        if mark == RUN_MARK:
            continue
        try:
            entry = json.loads(mark.read_text())
            if not alive(entry):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if time.time() < entry.get("until", 0):
            return True
    return False

def on_drake(fn, *args):
    return DRAKE.submit(fn, *args).result()

TRAIN_DIR = ROOT / "runs" / ("dashboard-training" if VARIANT == "3r" else f"dashboard-training-{VARIANT}")
CHECKPOINT_DIRS = [ROOT / "policy", TRAIN_DIR] + [
    Path(p).expanduser() for p in os.environ.get("PLANAR2D_CHECKPOINT_DIRS", "").split(":") if p]

_SCAN = [0., None]

def checkpoints():
    if time.monotonic() - _SCAN[0] < 10 and _SCAN[1] is not None:
        return _SCAN[1]
    found = {}
    for base in CHECKPOINT_DIRS:
        if not base.is_dir():
            continue
        for archive in sorted(base.rglob("*.zip")):
            if "episodes" in archive.parts:
                continue
            meta_path = archive.with_suffix(".json")
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                continue
            training = meta.get("training") if isinstance(meta, dict) else None
            settings = meta.get("config", {}) if isinstance(meta, dict) else {}
            if (not isinstance(training, dict) or not isinstance(settings, dict)
                    or training.get("robot_variant") != VARIANT
                    or training.get("learning_algorithm") != "SAC"
                    or training.get("hidden_width") not in (64, 256, 1024)
                    or meta.get("format") != "step-sb3-v1"
                    or settings.get("timing_profile") != "prune-kto-v1"
                    or settings.get("tcp_observed") is not False
                    or settings.get("scene_observed") is not False):
                continue
            stem = archive.with_suffix("")
            key = "~/" + str(stem.relative_to(Path.home())) if stem.is_relative_to(Path.home()) else str(stem)
            width, steps = (v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None
                            for v in (training.get("hidden_width"), training.get("timesteps")))
            where = archive.parent.name if base == TRAIN_DIR else base.name
            found[key] = {"key": key, "name": archive.stem, "width": width, "steps": steps,
                          "lineage": training.get("lineage"), "folder": where, "path": str(stem),
                          "trained_scene_profile": training.get("scene_profile"),
                          "label": f"width {width} · {steps/1e6:g}M steps" if width and steps else archive.stem}
            if VARIANT == "2r" and training.get("scene_profile") != SCENE_PROFILE:
                found[key]["label"] += " · previous scene, transfer preview"
    _SCAN[:] = [time.monotonic(), dict(sorted(found.items(), key=lambda kv: (
        kv[1]["folder"] == "policy", kv[1]["width"] or 0, kv[1]["steps"] or 0, kv[1]["name"])))]
    return _SCAN[1]

def default_policy(found):
    wanted = os.environ.get("PLANAR2D_DEFAULT_POLICY")
    if wanted:
        for key, entry in found.items():
            if entry["name"] == wanted:
                return key
    wanted = "kto256-s963-at2000000" if VARIANT == "3r" else "aperture2r256-s965-at3000000"
    for key, entry in found.items():
        if entry["name"] == wanted:
            return key
    if VARIANT == "2r":
        current = {k: v for k, v in found.items() if v.get("trained_scene_profile") == SCENE_PROFILE}
        if current:
            return max(current, key=lambda k: (current[k]["steps"] or 0, current[k]["width"] or 0))
    return max(found, key=lambda k: (found[k]["folder"] == "policy", found[k]["steps"] or 0, found[k]["width"] or 0),
               default=None)

@lru_cache(maxsize=4)
def learned_planner(key):
    import torch
    torch.set_num_threads(1)
    return build("rl_sac", checkpoint=Path(key).expanduser()).planner

SCENE_NAMES = {"nominal": "Wall on", "free": "Wall off"}
DEFAULT_Q = [2.6, .35, -.4] if ROBOT.n_joints == 3 else held_start(0, scene=SCENES["nominal"])["q"]

def two_link_angles(goal):
    l1, l2 = ROBOT.lengths_m
    p = np.asarray(goal, dtype=float) - np.asarray(ROBOT.base_xy_m, dtype=float)
    c = (p @ p - l1*l1 - l2*l2) / (2*l1*l2)
    if abs(c) > 1:
        return np.empty((0, 2))
    q2 = np.array([1., -1.])*np.arccos(c)
    q1 = (np.arctan2(p[1], p[0]) - np.arctan2(l2*np.sin(q2), l1 + l2*np.cos(q2)) + np.pi) % (2*np.pi) - np.pi
    return np.column_stack((q1, q2))

@cache
def goal_poses():
    from planar_physics.ik import solve_ik
    from planar_physics.plan import IKConfig
    poses = {}
    for name, goal in GOALS.items():
        if ROBOT.n_joints == 2:
            padding = CertifierConfig(margin_m=DEFAULT_CONTRACT.margin_m).padding(ROBOT)
            candidates = [q for q in two_link_angles(goal) if within_limits(ROBOT, q)
                          and is_free(ROBOT, SCENES['nominal'], q, margin_m=padding, numerical_slack_m=0.)]
            if candidates:
                poses[name] = min(candidates, key=lambda q: abs(q[0])).tolist()
                continue
        result = solve_ik(ROBOT, SCENES['nominal'], tuple(goal), rng=SeedSpec(0).stream('ik.samples'),
                          seed_q=np.asarray(DEFAULT_Q), cfg=IKConfig(tol_m=DEFAULT_CONTRACT.goal_tol_m,
                          margin_m=DEFAULT_CONTRACT.margin_m), certifier=CertifierConfig(margin_m=DEFAULT_CONTRACT.margin_m))
        if not len(result.configs):
            raise ValueError(f"The fixed {name} goal has no pose accepted by the common IK check.")
        poses[name] = result.configs[0].tolist()
    return poses

STARTED_AT = time.time()
SESSION = f"{int(STARTED_AT)}-{os.getpid()}"

def world(name):
    if name not in SCENE_NAMES:
        raise ValueError("Unknown surroundings.")
    return SCENES[name]

def drawn_world(value):
    if not isinstance(value, dict) or set(value) - {"wall", "obstacles"} or value.get("obstacles"):
        raise ValueError("A drawn scene is a wall description.")
    wall = value.get("wall")
    if wall is None:
        return rebuild({"wall": None})
    if not isinstance(wall, dict) or set(wall) != {"plane_x_m", "half_height_m", "thickness_m",
                                                     "gap_centres_m", "gap_widths_m"}:
        raise ValueError("Invalid wall description.")
    real = lambda x, low, high, label: number(x if isinstance(x, (int, float)) and not isinstance(x, bool)
                                              else float("nan"), low, high, label)
    half = real(wall["half_height_m"], .1, 2., "Wall half height")
    centres, widths = wall["gap_centres_m"], wall["gap_widths_m"]
    if not isinstance(centres, list) or not isinstance(widths, list) or not 1 <= len(centres) == len(widths) <= 4:
        raise ValueError("A wall needs one to four openings, each with a centre and a width.")
    centres = [real(c, -half, half, "Opening centre") for c in centres]
    widths = [real(w, .01, 2*half, "Opening width") for w in widths]
    centres, widths = map(list, zip(*sorted(zip(centres, widths))))
    spans = [(c - w/2, c + w/2) for c, w in zip(centres, widths)]
    if spans[0][0] < -half or spans[-1][1] > half or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("The openings must lie inside the wall and must not overlap.")
    return rebuild({"wall": {"plane_x_m": real(wall["plane_x_m"], .05, 1.2, "Wall plane"), "half_height_m": half,
                             "thickness_m": real(wall["thickness_m"], .005, .2, "Wall thickness"),
                             "gap_centres_m": centres, "gap_widths_m": widths}})

def scene_of(data):
    if data.get("source"):
        return rebuild(data["world"])
    return drawn_world(data["world"]) if data.get("world") is not None else world(data["scene"])

EVAL_SETS = {"selection": ("Selection set" if VARIANT == "3r" else "New selection preview", *SELECTION),
             "final": ("Final set" if VARIANT == "3r" else "New final preview", *FINAL)}
TRAIN_BANK = ROOT / "banks" / ("train-2r-aperture.json" if VARIANT == "2r" else "train-3r.json")

def draw_task(seed, mixed):
    rng = np.random.default_rng(seed)
    spec = SceneSpec() if mixed else replace(SceneSpec(), wall_probability=1.)
    scene = sample_scene(rng, spec, ROBOT)
    goal = np.asarray(sample_goal(rng, spec, ROBOT, scene), dtype=float)
    held = held_start(seed, scene=scene)
    if float(np.linalg.norm(tcp(ROBOT, held["q"]) - goal)) <= DEFAULT_CONTRACT.goal_tol_m:
        raise ValueError("start_in_goal")
    return scene, goal, held

@cache
def eval_task(name, index):
    label, base, count = EVAL_SETS[name]
    if not 0 <= index < count:
        raise ValueError(f"{label} has requests 0 to {count-1}.")
    seed = base + index
    try:
        scene, goal, held = draw_task(seed, index % 2 == 0)
    except ValueError as error:
        raise ValueError(f"{label} request {index} (seed {seed}) is not issued in the evaluation: {error}.")
    k = (index // 2) % len(EVAL_ORDER)
    return {"seed": seed, "scene": asdict(scene), "start": list(held["q"]), "goal_xy": goal.tolist(),
            "label": f"{label} #{index} · {'mixed' if index % 2 == 0 else 'wall'} · seed {seed}",
            "order": list(EVAL_ORDER[k:] + EVAL_ORDER[:k])}

@cache
def bank_records():
    try:
        body = json.loads(TRAIN_BANK.read_text())
    except OSError:
        raise ValueError(f"The training bank {TRAIN_BANK.name} is missing; build it with scripts/training/build_bank.py.")
    if VARIANT == "2r" and body.get("scene_profile") != SCENE_PROFILE:
        raise ValueError("This training bank belongs to another 2R scene profile.")
    return [r for r in body["records"] if r["status"] == "ok"]

def bank_task(index):
    records = bank_records()
    if not 0 <= index < len(records):
        raise ValueError(f"The training bank has tasks 0 to {len(records)-1}.")
    r = records[index]
    return {"seed": r["seed"], "scene": r["scene"], "start": list(r["start"]), "goal_xy": list(r["goal_xy"]),
            "label": f"Training task #{index} · seed {r['seed']}", "classical": r.get("classical"),
            "reference_s": r.get("reference_s"), "reference_planner": r.get("reference_planner")}

def bank_index(seed):
    return next((i for i, r in enumerate(bank_records()) if r["seed"] == seed), None)

def source_task(payload):
    source = payload.get("source")
    index = integer(payload.get("index", 0), 0, 10**6, "Request number")
    if source == "eval":
        name = payload.get("set")
        if name not in EVAL_SETS:
            raise ValueError("Unknown evaluation set.")
        return {"kind": "eval", "set": name, "index": index, **eval_task(name, index)}
    if source == "bank":
        return {"kind": "bank", "index": index, **bank_task(index)}
    raise ValueError("Unknown request source.")

def vector(value, n, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"{label} must contain {n} finite numbers.")
    return a

def number(value, low, high, label):
    x = float(value)
    if not np.isfinite(x) or not low <= x <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return x

def integer(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}.")
    return value

def query(payload):
    source = None
    if payload.get("source") in ("eval", "bank"):
        task = source_task(payload)
        source = {k: task[k] for k in ("kind", "set", "index", "label", "classical", "reference_s",
                                       "reference_planner", "order") if k in task}
        payload = {**payload, "scene": "drawn", "q": task["start"], "goal": task["goal_xy"],
                   "goal_q": None, "seed": task["seed"]}
        drawn = task["scene"]
        rebuild(drawn)
        name = "drawn"
    else:
        name = payload.get("scene", "nominal")
        drawn = payload.get("world") if name == "drawn" else None
        if name == "drawn":
            drawn = asdict(drawn_world(drawn))
        else:
            world(name)
    q = vector(payload.get("q", DEFAULT_Q), ROBOT.n_joints, "Joint angles")
    if not within_limits(ROBOT, q):
        raise ValueError("A starting joint angle is outside its limit.")
    goal_q = payload.get("goal_q")
    if goal_q is not None:
        goal_q = vector(goal_q, ROBOT.n_joints, "Goal joint angles")
        if not within_limits(ROBOT, goal_q):
            raise ValueError("A goal joint angle is outside its limit.")
    if "goal" in payload or goal_q is None:
        goal = vector(payload.get("goal", GOALS["upper"]), 2, "Goal")
    else:
        goal = np.asarray(tcp(ROBOT, goal_q), dtype=float)
    if np.abs(goal).max() > 1.2:
        raise ValueError("Place the goal inside the displayed workspace.")
    seed = integer(payload.get("seed", 0), 0, 2**31-1, "Seed")
    return {"physics_dt_s": DEFAULT_CONTRACT.dt_s, "q": q.tolist(), "goal": goal.tolist(), "source": source,
            "goal_q": None if goal_q is None else goal_q.tolist(), "scene": name, "world": drawn, "seed": seed,
            "deadline_s": DEADLINE_S, "margin_m": DEFAULT_CONTRACT.margin_m, "planning_s": PLANNING_S}

def pose(payload, *, scene=None):
    data = query(payload)
    if scene is None:
        scene = scene_of(data)
    robot = ROBOT
    gap = float(np.minimum(clearance(robot, scene, data["q"]), np.min(exact_clearance(robot, scene, data["q"]))))
    cert = blocked_goal_certificate(robot, scene, data["goal"], DEFAULT_CONTRACT.goal_tol_m)
    return {**data, "world": asdict(scene),
            "points": joint_positions(robot, data["q"]).tolist(),
            "goal_points": None if data["goal_q"] is None else joint_positions(robot, data["goal_q"]).tolist(),
            "clearance_m": gap, "valid": gap > DEFAULT_CONTRACT.numerical_slack_m,
            "planning_valid": gap >= CertifierConfig(margin_m=data["margin_m"]).padding(robot),
            "feasibility": "certified_infeasible" if cert else "unknown",
            "evidence": cert.detail if cert else "No impossibility certificate. Search failure remains unknown."}

def random_request(payload):
    data = query({**payload, "scene": "nominal", "world": None})
    for k in range(64):
        seed = (data["seed"] + k) % 2**31
        try:
            scene, goal, start = draw_task(seed, seed % 2 == 0)
        except ValueError:
            continue
        return pose({**data, "q": start["q"], "goal": goal.tolist(), "goal_q": None, "seed": seed,
                     "scene": "drawn", "world": asdict(scene)})
    raise ValueError("No request could be drawn for this seed.")

def two_link_sides(scene, goal):
    sides = set()
    for q in two_link_angles(goal):
        if within_limits(ROBOT, q) and is_free(ROBOT, scene, q, margin_m=DEFAULT_CONTRACT.margin_m):
            sides.add(np.sign(q[0]))
    return sides

def random_pose(payload):
    if payload.get("random_goal") is True:
        return random_request(payload)
    data = query(payload)
    scene = scene_of(data)
    sides = two_link_sides(scene, data["goal"]) if ROBOT.n_joints == 2 and scene.wall is not None else set()
    for k in range(64):
        start = held_start(data["seed"] + k, scene=scene, margin_m=max(.01, data["margin_m"]))
        if not sides or np.sign(start["q"][0]) in sides:
            break
    return {**pose({**data, "q": start["q"]}, scene=scene), "start_validation": start}

def trace_metrics(recorded):
    trace, measured, series = recorded.trace, recorded.metrics, recorded.series
    t, q, tau = trace.t, trace.q, trace.sample_torque
    points, distance, joint_arc, jrms, elapsed, mean_speed = (
        series[name] for name in ('points', 'distance', 'joint_arc', 'jrms', 'elapsed', 'mean_speed'))
    display_stride = max(1, round(.02 / (t[1]-t[0]))) if len(t) > 1 else 1
    indices = sorted(set(range(0, len(t), display_stride)) | {len(t)-1})
    frames = [{"t": float(t[i]), "q": q[i].tolist(), "points": points[i].tolist(),
               "mean_speed": float(mean_speed[i]), "distance": float(distance[i]),
               "joint_arc": joint_arc[i].tolist(), "jerk_rms": float(jrms[i])} for i in indices]
    return frames, {"distance_m": measured.distance_m, "mean_speed_m_s": float(mean_speed[-1]),
                    "jerk_rms_rad_s3": measured.jerk_rms_rad_s3,
                    "peak_torque_nm": np.abs(tau).max(axis=0).tolist(),
                    "joint_traverse_rad": joint_arc[-1].tolist(), "physical_s": float(elapsed[-1])}

def path_display(points, robot=ROBOT):
    samples = np.vstack([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.04))+1))
                         for a, b in zip(points[:-1], points[1:])])
    return tcp(robot, samples).tolist()

def reference_reversals(reference, step_s=REVERSAL_STEP_S, until=None):
    start, end = reference.start_time(), reference.end_time()
    if until is not None:
        end = max(start, min(end, start + until))
    t = np.linspace(start, end, max(2, int(np.ceil((end - start)/step_s)) + 1))
    return joint_reversals(reference.MakeDerivative().vector_values(t).T)

def learned_or_search(mode, checkpoint=None):
    if mode == "policy":
        if not checkpoint:
            raise ValueError("No policy checkpoint is available for this arm.")
        return learned_planner(checkpoint)
    return build(mode).planner

def graph_display(graph, robot=ROBOT):
    return project_search_graph(graph, lambda q: tcp(robot, q),
                                lambda path: path_display(path, robot))

@lru_cache(maxsize=4)
def lane_plant(timestep_s, robot=ROBOT):
    return PlanarPlant.build(robot, PlantConfig(timestep_s=timestep_s))

def mission_of(data):
    return MissionRequest(np.asarray(data["q"]), tuple(data["goal"]), SeedSpec(data["seed"]),
                          deadline_s=data["deadline_s"], clearance_margin_m=data["margin_m"],
                          budget=Budget(wall_clock_s=data["planning_s"]))

def shared_goals(data):
    request = plan_request(ROBOT, scene_of(data), mission_of(data))
    return resolve_goals(request, Ledger(request.budget))

def run_lane(data, mode, submitted_at=None, goals=None, checkpoint=None):
    queue_s = max(0., time.monotonic()-submitted_at) if submitted_at is not None else 0.
    scene, robot = scene_of(data), ROBOT
    plant = lane_plant(data.get("physics_dt_s", DEFAULT_CONTRACT.dt_s), robot)
    result = {"mode": mode, "name": MODES[mode], "raw_path": [], "prepared_nodes": [], "reference_path": [],
              "search": None, "joint_reversals": None, "queue_s": queue_s, "raw_waypoint_count": 0,
              "feasibility": pose(data, scene=scene)["feasibility"]}
    planner = Capture(learned_or_search(mode, checkpoint), collect_graph=True)
    req = mission_of(data)
    if goals is None:
        request = plan_request(robot, scene, req)
        goals = resolve_goals(request, Ledger(request.budget))
    report = run_mission(robot, scene, plant, planner, req, goals=goals)
    plan = planner.result
    if plan is not None:
        result["search"] = graph_display(plan.graph, robot)
        if plan.solved and plan.waypoints_rad is not None:
            result.update(raw_path=path_display(plan.waypoints_rad, robot),
                          raw_nodes=tcp(robot, plan.waypoints_rad).tolist(), raw_waypoint_count=len(plan.waypoints_rad))
    if report.prepared_path_rad is not None:
        result["prepared_nodes"] = tcp(robot, report.prepared_path_rad).tolist()
    result.update(outcome=report.outcome, reason=report.reason, success=report.succeeded,
                  planning_s=report.planning_s, preparation_s=report.preparation_s,
                  mission_s=report.request_to_completion_s)
    if report.execution is None:
        plant.reset(np.asarray(data["q"]))
        recorded = run(plant, scene, 0.)
    else:
        recorded = report.execution.run
    if report.reference_trajectory is not None:
        reference = report.reference_trajectory
        duration = reference.end_time() - reference.start_time()
        ref_t = np.linspace(reference.start_time(), reference.end_time(), max(2, int(np.ceil(duration/.01))+1))
        result["reference_path"] = tcp(robot, reference.vector_values(ref_t).T).tolist()
        if report.execution is not None:
            result["joint_reversals"] = reference_reversals(reference, until=recorded.duration_s)
    frames, metrics = trace_metrics(recorded)
    result.update(frames=frames, metrics=metrics)
    return json_ready(result)


class Jobs:
    def __init__(self):
        self.pool = self._pool()
        self.chain = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jobs")
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False

    def _pool(self):
        return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    def run(self, fn, *args):
        with self.lock:
            try:
                if getattr(self.pool, "_broken", False):
                    raise BrokenProcessPool(self.pool._broken)
                return self.pool.submit(fn, *args)
            except BrokenProcessPool:
                self.pool.shutdown(wait=False, cancel_futures=True)
                self.pool = self._pool()
                return self.pool.submit(fn, *args)

    def busy(self):
        with self.lock:
            return any(any(f.running() for f in j["futures"].values()) if j["cancelled"] else not self._done(j)
                       for j in self.jobs.values())

    @staticmethod
    def _done(job):
        return (job["cancelled"] and all(f.done() for f in job["futures"].values())) or job.get("failed") is not None or (
            len(job["futures"]) == len(job["run_order"]) and all(f.done() for f in job["futures"].values()))

    def submit(self, data, modes, goals, checkpoint):
        modes = [mode for mode in LANES if mode in modes]
        if not modes:
            raise ValueError("Turn on at least one arm.")
        if "policy" in modes and not checkpoint:
            raise ValueError("No policy checkpoint is available; hide the policy arm or add a checkpoint.")
        order = (data.get("source") or {}).get("order") or EVAL_ORDER
        with self.lock:
            if self.busy():
                raise ValueError("A run, or an arm of a stopped run, is still calculating. It ends within the 20 s deadline.")
            for key in [k for k, j in self.jobs.items() if self._done(j) or j["cancelled"]][:-8]:
                del self.jobs[key]
            key = uuid.uuid4().hex
            RUN_MARK.write_text(json.dumps({"pid": os.getpid(), "pid_started": started(os.getpid()),
                                            "until": time.time() + 40*len(modes)}))
            self.jobs[key] = {"query": data, "futures": {}, "order": list(LANES), "cancelled": False,
                              "run_order": [m for m in order if m in modes], "goals": goals,
                              "checkpoint": checkpoint, "failed": None}
        self._next(key)
        return {"id": key}

    def _next(self, key):
        with self.lock:
            job = self.jobs.get(key)
            if job is None or job["cancelled"] or job["failed"] is not None:
                if not self.busy():
                    RUN_MARK.unlink(missing_ok=True)
                return
            if any(not f.done() for f in job["futures"].values()):
                return
            for mode, future in list(job["futures"].items()):
                if (not future.cancelled() and isinstance(future.exception(), BrokenProcessPool)
                        and mode not in job.setdefault("retried", set())):
                    job["retried"].add(mode)
                    del job["futures"][mode]
            pending = [m for m in job["run_order"] if m not in job["futures"]]
            if not pending:
                if not self.busy():
                    RUN_MARK.unlink(missing_ok=True)
                return
            mode = pending[0]
            try:
                future = self.run(run_lane, job["query"], mode, time.monotonic(), job["goals"], job["checkpoint"])
            except Exception as error:
                job["failed"] = f"The worker could not start {MODES[mode]}: {error}"
                if not self.busy():
                    RUN_MARK.unlink(missing_ok=True)
                return
            job["futures"][mode] = future
        future.add_done_callback(lambda _: None if self.closed else self.chain.submit(self._next, key))

    def get(self, key):
        with self.lock:
            job = self.jobs.get(key)
            if job is None:
                raise ValueError("This replay has expired on the server. Start a new run.")
            lanes = []
            for mode in job["order"]:
                lane = {"mode": mode, "name": MODES.get(mode, mode)}
                future = job["futures"].get(mode)
                if mode not in job["run_order"]:
                    lanes.append({**lane, "state": "off"})
                elif future is None:
                    lanes.append({**lane, "state": "error", "reason": "Run stopped."} if job["cancelled"]
                                 else {**lane, "state": "error", "reason": job["failed"]} if job["failed"]
                                 else {**lane, "state": "queued"})
                elif not future.done():
                    lanes.append({**lane, "state": "calculating"})
                else:
                    try:
                        lanes.append({"state": "done", **future.result()})
                    except Exception as error:
                        lanes.append({**lane, "state": "error", "reason": str(error)})
            return {"id": key, "query": job["query"], "done": self._done(job), "cancelled": job["cancelled"], "lanes": lanes}

    def cancel(self, key):
        with self.lock:
            job = self.jobs.get(key)
            if job is not None:
                job["cancelled"] = True
                for future in job["futures"].values():
                    future.cancel()
                if not self.busy():
                    RUN_MARK.unlink(missing_ok=True)
        return {"id": key, "cancelled": job is not None}

def simulate_episode(body, dt=DEFAULT_CONTRACT.dt_s):
    from pydrake.trajectories import PiecewisePolynomial
    curve = body["curve"]
    t, q, dq = (np.asarray(curve[k], dtype=float) for k in ("t", "q", "dq"))
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    reference = PiecewisePolynomial.CubicHermite(t[keep], q[keep].T, dq[keep].T)
    scene = rebuild(body["scene"])
    goal = np.asarray(body["goal_xy"], dtype=float)
    report = execute_trajectory(lane_plant(dt), scene, reference, goal_xy_m=goal)
    frames, metrics = trace_metrics(report.run)
    torque = ROBOT.max_joint_torque_nm
    return json_ready({"frames": frames, "metrics": metrics, "outcome": report.outcome, "held": report.held,
                       "final_error_m": report.final_tcp_error_m,
                       "peak_torque_share": max(v / c for v, c in zip(metrics["peak_torque_nm"], torque))})

def episode_run(body, sim=None, speed_rad_s=1.2, longest_s=6., dt=.02):
    tip = lambda qs: tcp(ROBOT, np.asarray(qs, dtype=float).reshape(-1, ROBOT.n_joints)).tolist()
    walked = tip(body["path"])
    lane = {"state": "done", "raw_path": walked, "raw_nodes": walked}
    if body.get("pruned"):
        lane["prepared_nodes"] = tip(body["pruned"])
    run = {"id": f"training-{body.get('training')}-env{body.get('env', 0)}-{body['episode']}",
           "query": {"q": body["path"][0], "goal": body["goal_xy"]}, "lanes": [lane]}
    if body.get("curve"):
        qs = np.asarray(body["curve"]["q"])
        lane["reference_path"] = tip(qs[::5])
        if sim is not None:
            lane["frames"] = sim["frames"]
            return {**run, "duration_s": float(sim["metrics"]["physical_s"])}
        t = np.asarray(body["curve"]["t"])
        pick = sorted(set(range(0, len(t), max(1, round(dt / max(t[1] - t[0], 1e-9))))) | {len(t) - 1})
        lane["frames"] = [{"t": float(t[i]), "q": qs[i].tolist(), "points": joint_positions(ROBOT, qs[i]).tolist()}
                          for i in pick]
        return {**run, "duration_s": float(t[-1])}
    path = np.asarray(body.get("pruned") or body["path"], dtype=float)
    if len(path) == 1:
        path = np.repeat(path, 2, axis=0)
    arc = np.concatenate([[0.], np.cumsum(np.abs(np.diff(path, axis=0)).max(axis=1))])
    times = arc / max(speed_rad_s, arc[-1] / longest_s)
    t = np.arange(0., times[-1] + dt, dt) if times[-1] > 0 else np.array([0.])
    qs = np.column_stack([np.interp(t, times, path[:, j]) for j in range(path.shape[1])])
    lane["frames"] = [{"t": float(ti), "q": qi.tolist(), "points": joint_positions(ROBOT, qi).tolist()} for ti, qi in zip(t, qs)]
    return {**run, "duration_s": float(t[-1])}

def started(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        pass
    try:
        return subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None

def alive(entry):
    pid = entry["pid"]
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    if started(pid) != entry.get("pid_started"):
        return False
    try:
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        try:
            if sys.platform.startswith("linux"):
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            else:
                state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                       capture_output=True, text=True).stdout.strip()
            return bool(state) and not state.startswith(("Z", "X"))
        except (OSError, IndexError):
            return False

class Training:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = TRAIN_DIR / "current.json"
        self.recent = {}
        self.simulating = {}
        self.simulate = None

    def _load(self):
        try:
            return json.loads(self.current.read_text())
        except (OSError, ValueError):
            return None

    def _save(self, entry):
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        self.current.write_text(json.dumps(entry, indent=1))

    def status(self):
        with self.lock:
            entry = self._load()
            info = {"bank": TRAIN_BANK.name if TRAIN_BANK.is_file() else None, "state": "idle",
                    "training_elsewhere": self.active(), "run_elsewhere": timed_run_elsewhere()}
            if not entry:
                return info
            out, name = Path(entry["out"]), entry["name"]
            lines = []
            try:
                lines = (out / "train.log").read_text(errors="replace").splitlines()
            except OSError:
                pass
            progress, complete = None, False
            for line in lines:
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get("stage") == "sac":
                        progress = record
                    elif record.get("stage") == "complete":
                        complete = True
            running = alive(entry)
            state = ("stopping" if entry.get("state") == "stopped" else "running") if running else \
                entry.get("state") if entry.get("state") == "stopped" else \
                "finished" if complete and (out / f"{name}.zip").is_file() else "failed"
            if state != entry.get("state") and not running:
                entry["state"] = state
                self._save(entry)
            total = entry["settings"]["steps"]
            checkpoints = sorted((p.name for p in out.glob(f"{name}-at*.zip")),
                                 key=lambda n: int(n.rsplit("-at", 1)[1][:-4]))
            reached = int(checkpoints[-1].rsplit("-at", 1)[1][:-4]) if checkpoints else 0
            steps = total if state == "finished" else max((progress or {}).get("steps", 0), reached)
            rate = progress["steps"] / progress["seconds"] if progress and progress.get("seconds") else None
            if (out / f"{name}.zip").is_file():
                checkpoints.append(f"{name}.zip")
            info.update(state=state, name=name, settings=entry["settings"], started=entry["started"],
                        steps=steps, total=total, rate=rate,
                        eta_s=(total - steps) / rate if rate and state == "running" else None,
                        episodes=(progress or {}).get("episodes"), checkpoints=checkpoints,
                        folder=str(out.relative_to(ROOT)),
                        last_line=lines[-1][:300] if lines and state == "failed" else None)
            return info

    def start(self, payload, jobs_busy=False):
        if jobs_busy or timed_run_elsewhere():
            raise ValueError("A run is calculating. Wait for it or stop it, so the training does not distort its timings.")
        if self.active():
            raise ValueError("A training is already running on this machine. Stop it or wait until it ends.")
        cores = os.cpu_count() or 1
        steps = integer(payload.get("steps"), 10_000, 10_000_000, "Steps")
        width = integer(payload.get("width"), 64, 1024, "Width")
        if width not in (64, 256, 1024):
            raise ValueError("Width must be 64, 256 or 1024.")
        envs, threads = (10, 8) if width == 1024 else (5, 2)
        threads, preview = min(threads, cores), 1
        if steps % envs:
            raise ValueError(f"Steps must be a multiple of {envs}.")
        if not TRAIN_BANK.is_file():
            raise ValueError(f"The training bank {TRAIN_BANK.name} is missing.")
        with self.lock:
            entry = self._load()
            if entry and alive(entry):
                raise ValueError("A training is already running. Stop it first.")
            every = max(round(steps / 5 / envs) * envs, envs)
            base = f"dash-{VARIANT}-w{width}-{time.strftime('%Y%m%d-%H%M%S')}"
            name, k = base, 1
            while (TRAIN_DIR / name).exists():
                k += 1
                name = f"{base}-{k}"
            out = TRAIN_DIR / name
            try:
                out.mkdir(parents=True)
            except OSError as error:
                raise ValueError(f"Cannot create the training folder: {error}")
            command = ["nice", "-n", "5", sys.executable, str(ROOT / "scripts" / "training" / "train_kto.py"),
                       "--bank", str(TRAIN_BANK), "--name", name, "--steps", str(steps),
                       "--checkpoint-every", str(every), "--envs", str(envs), "--torch-threads", str(threads),
                       "--width", str(width), "--seed", "963", "--out", str(out), "--min-tasks", "1",
                       "--preview-every", str(preview), "--preview-dir", str(out / "episodes")]
            env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT / "src"), PLANAR2D_ROBOT=VARIANT)
            with open(out / "train.log", "w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            self._save({"pid": process.pid, "pid_started": started(process.pid), "name": name, "out": str(out), "state": "running",
                        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "settings": {"steps": steps, "width": width, "envs": envs, "threads": threads,
                                     "checkpoint_every": every, "preview": preview}})
        return self.status()

    def episode(self, kind="reached"):
        with self.lock:
            entry = self._load()
        if not entry:
            raise ValueError("No training has been started from the dashboard.")
        folder = Path(entry["out"]) / "episodes"
        files = [f for f in folder.glob("env*.json") if f.stem.endswith("-reached") == (kind == "reached")]
        files += [folder / "latest.json"] if (folder / "latest.json").is_file() else []
        newest = None
        for f in files:
            try:
                body = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if kind == "reached" and not body.get("reached"):
                continue
            if newest is None or body.get("written", 0) > newest.get("written", 0):
                newest = body
        if newest is None:
            raise ValueError("No goal-reaching episode recorded yet." if kind == "reached"
                             else "No training episode recorded yet.")
        body = newest
        body["approx_steps"] = body["transitions"] * entry["settings"]["envs"]
        body["training"] = entry["name"]
        key = episode_run(body)["id"]
        with self.lock:
            if key in self.recent:
                return self.recent[key][0]
            waiting = self.simulating.get(key)
            if waiting is None:
                self.simulating[key] = threading.Event()
        if waiting is not None:
            waiting.wait(90)
            with self.lock:
                if key in self.recent:
                    return self.recent[key][0]
            raise ValueError("The episode could not be prepared; the next one follows.")
        try:
            return self._prepare_episode(body)
        finally:
            with self.lock:
                self.simulating.pop(key).set()

    def _prepare_episode(self, body):
        index = bank_index(body["seed"]) if body.get("seed") is not None else None
        if index is not None:
            body.update(bank_index=index, classical=bank_records()[index].get("classical"))
        sim = None
        if body.get("curve") and "dq" in body["curve"] and self.simulate is not None:
            try:
                sim = self.simulate(body)
                body["simulation"] = {k: sim[k] for k in ("outcome", "held", "final_error_m", "peak_torque_share")}
            except Exception as error:
                body["simulation"] = {"outcome": "error", "reason": str(error)}
        run = episode_run(body, sim)
        body.update(id=run["id"], replay_s=run["duration_s"], pruned_count=len(body.get("pruned") or []))
        with self.lock:
            self.recent[run["id"]] = (body, run)
            while len(self.recent) > 8:
                self.recent.pop(next(iter(self.recent)))
        return body

    def runs(self):
        with self.lock:
            entry = self._load()
            live = entry["name"] if entry and alive(entry) else None
        out = []
        for folder in sorted(TRAIN_DIR.glob("dash-*"), reverse=True):
            if not folder.is_dir():
                continue
            size = 0
            for f in folder.rglob("*"):
                try:
                    size += f.stat().st_size if f.is_file() else 0
                except OSError:
                    pass
            out.append({"name": folder.name, "running": folder.name == live,
                        "checkpoints": sum(1 for f in folder.glob("*.zip")), "mb": round(size / 1e6, 1)})
        return {"runs": out}

    def delete(self, name, jobs_busy=False):
        if jobs_busy or timed_run_elsewhere():
            raise ValueError("Wait until the calculating run has ended before deleting checkpoints.")
        if not isinstance(name, str) or not re.fullmatch(r"dash-[A-Za-z0-9-]+", name):
            raise ValueError("Unknown training.")
        folder = TRAIN_DIR / name
        if not folder.is_dir():
            raise ValueError("Unknown training.")
        with self.lock:
            entry = self._load()
            if entry and entry["name"] == name:
                if alive(entry):
                    raise ValueError("Stop this training before deleting it.")
                self.current.unlink(missing_ok=True)
                self.recent.clear()
            shutil.rmtree(folder)
            _SCAN[:] = [0., None]
            learned_planner.cache_clear()
        return self.runs()

    def replay(self, key):
        with self.lock:
            found = self.recent.get(key)
        if found is None:
            raise ValueError("This training episode is no longer cached; the next one follows.")
        return found

    def active(self):
        for current in (ROOT / "runs").glob("dashboard-training*/current.json"):
            try:
                entry = json.loads(current.read_text())
            except (OSError, ValueError):
                continue
            if alive(entry):
                return True
        return False

    def stop(self):
        with self.lock:
            entry = self._load()
            if not entry or not alive(entry):
                raise ValueError("No training is running.")
            try:
                os.killpg(entry["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            entry["state"] = "stopped"
            self._save(entry)
        return self.status()

def config():
    return {"source": VARIANT, "variant": VARIANT, "scene_profile": SCENE_PROFILE,
            "arm_label": f"{ROBOT.n_joints}-link arm",
            "robot": {**asdict(ROBOT_EQUAL_030), "joint_limits_rad": ROBOT.joint_limits_rad,
                      "max_joint_torque_nm": list(ROBOT.max_joint_torque_nm)},
            "scenes": SCENE_NAMES, "goals": GOALS, "goal_poses": goal_poses(),
            "scene_goals": {name: goal_names(name) for name in SCENE_NAMES},
            "modes": MODES, "lanes": list(LANES), "physics_dt_s": DEFAULT_CONTRACT.dt_s, "pose": pose({}),
            "peer": os.environ.get("DASHBOARD_PEER") or None,
            "peer_port": int(os.environ["DASHBOARD_PEER_PORT"]) if os.environ.get("DASHBOARD_PEER_PORT") else None,
            "peer_label": os.environ.get("DASHBOARD_PEER_LABEL") or None,
            "eval_sets": {k: {"label": v[0], "count": v[2]} for k, v in EVAL_SETS.items()},
            "bank": {"name": TRAIN_BANK.name, "count": len(bank_records())} if TRAIN_BANK.is_file() else None}

VIEW_ONLY = {"search", "raw_path", "raw_nodes", "prepared_nodes", "reference_path"}

class Handler(BaseHTTPRequestHandler):
    jobs: Jobs
    timeout = 30

    def send(self, code, data, mime="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(json_ready(data), allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            self._get()
        except Exception as error:
            self.send(500, {"error": f"{type(error).__name__}: {error}"})

    def _get(self):
        path = self.path.split("?")[0]
        if path in ("/favicon.ico", "/physics_dashboard_logo.svg"):
            self.send(200, (HERE / "physics_dashboard_logo.svg").read_bytes(), "image/svg+xml")
        elif path == "/meshcat.js":
            self.send(200, Path(FindResourceOrThrow("drake/geometry/meshcat.js")).read_bytes(), "text/javascript")
        elif re.fullmatch(r"/cas-v1/[a-f0-9]{64}", path):
            asset = self.views.asset(path)
            if asset is None:
                self.send(404, {"error": "Viewer resource not found"})
            else:
                self.send(200, *asset)
        elif path in ("/", "/physics_dashboard_meshcat.html"):
            name = "physics_dashboard.html" if path == "/" else "physics_dashboard_meshcat.html"
            self.send(200, (HERE / name).read_bytes(), "text/html; charset=utf-8")
        elif path in ("/physics_dashboard.js", "/physics_dashboard.css"):
            self.send(200, (HERE / path[1:]).read_bytes(),
                      "text/javascript" if path.endswith(".js") else "text/css")
        elif path == "/api/config":
            self.send(200, on_drake(config))
        elif path == "/api/policies":
            found = checkpoints()
            self.send(200, {"policies": list(found.values()), "default": default_policy(found)})
        elif path == "/api/training":
            self.send(200, self.training.status())
        elif path == "/api/training/runs":
            self.send(200, self.training.runs())
        elif path == "/api/training/episode":
            kind = "any" if "kind=any" in self.path else "reached"
            try:
                body = self.training.episode(kind)
            except ValueError as error:
                self.send(404, {"error": str(error)}); return
            self.send(200, {k: body.get(k) for k in ("training", "episode", "approx_steps", "code", "reached", "steps",
                                                     "blocked", "duration_s", "reference_s", "margin", "written",
                                                     "id", "replay_s", "bank_index", "classical", "seed", "env",
                                                     "pruned_count", "simulation")})
        elif path.startswith("/api/jobs/"):
            try:
                job = self.jobs.get(path.rsplit("/", 1)[-1])
                keep = ("t", "mean_speed", "distance", "joint_arc", "jerk_rms")
                job["lanes"] = [{**{k: v for k, v in lane.items() if k not in VIEW_ONLY},
                                 **({"frames": [{k: f[k] for k in keep} for f in lane["frames"]]} if "frames" in lane else {})}
                                for lane in job["lanes"]]
                self.send(200, job)
            except ValueError as error:
                self.send(404, {"error": str(error)})
        else:
            self.send(404, {"error": "Not found"})

    def do_POST(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.send(415, {"error": "Send the request as application/json."}); return
        origin = self.headers.get("Origin")
        if origin and origin.split("://", 1)[-1] != self.headers.get("Host", ""):
            self.send(403, {"error": "Cross-origin requests are not accepted."}); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected an object.")
            if self.path == "/api/view" and payload.get("training_episode"):
                body, run = self.training.replay(str(payload["training_episode"]))
                options = {**payload, "q": run["query"]["q"], "goal": run["query"]["goal"], "reach": False,
                           "executed": True, "off": [], "goal_q": None}
                result = {**self.views.update(str(payload.get("view_id", "")), rebuild(body["scene"]), options, run),
                          "session": SESSION}
            elif self.path == "/api/view":
                data = on_drake(query, payload)
                try:
                    run = self.jobs.get(str(payload["id"])) if payload.get("id") else None
                except ValueError:
                    run = None
                if run is not None:
                    data = run["query"]
                result = {**self.views.update(str(payload.get("view_id", "")),
                    scene_of(data), {**payload, "q": data["q"], "goal": data["goal"], "goal_q": data.get("goal_q"),
                                     "busy": self.jobs.busy() or self.training.active() or timed_run_elsewhere()}, run),
                    "session": SESSION}
            elif self.path == "/api/training/start":
                with machine_lock():
                    self.views.pause_maps()
                    result = self.training.start(payload, self.jobs.busy())
            elif self.path == "/api/training/stop":
                result = self.training.stop()
            elif self.path == "/api/training/delete":
                with machine_lock():
                    result = self.training.delete(payload.get("name"), self.jobs.busy())
            elif self.path == "/api/pose":
                result = on_drake(pose, payload)
            elif self.path == "/api/random":
                result = on_drake(random_pose, payload)
            elif self.path == "/api/cancel":
                result = self.jobs.cancel(str(payload.get("id", "")))
            elif self.path == "/api/run":
                with machine_lock():
                    if self.training.active():
                        raise ValueError("A training is running on this machine and would distort the timings. Stop it or wait until it ends.")
                    if timed_run_elsewhere():
                        raise ValueError("A run is calculating in the other dashboard. Stop it or wait until it ends.")
                    self.views.pause_maps()
                    data = on_drake(query, payload)
                    initial = on_drake(pose, data)
                    if not initial["valid"]:
                        raise ValueError("The starting pose touches geometry. Adjust the joints.")
                    if not initial["planning_valid"]:
                        raise ValueError("The starting pose does not meet the planning clearance margin.")
                    modes = payload.get("modes")
                    if modes is not None and (not isinstance(modes, list) or not all(isinstance(m, str) for m in modes)):
                        raise ValueError("The arm selection must be a list of lane names.")
                    modes = list(LANES) if modes is None else modes
                    if any(mode not in LANES for mode in modes) or len(set(modes)) != len(modes):
                        raise ValueError("Choose each supported arm at most once.")
                    if not modes:
                        raise ValueError("Turn on at least one arm.")
                    checkpoint = None
                    if "policy" in modes:
                        found = checkpoints()
                        checkpoint = payload.get("policy") or default_policy(found)
                        if checkpoint not in found:
                            raise ValueError("Choose a policy checkpoint from the list.")
                        data = {**data, "policy": found[checkpoint]}
                    goals = on_drake(shared_goals, data)
                    result = self.jobs.submit(data, modes, goals, checkpoint)
            else:
                self.send(404, {"error": "Not found"}); return
            self.send(200, result)
        except (ValueError, TypeError, OverflowError, IndexError) as error:
            self.send(400, {"error": str(error)})
        except Exception as error:
            self.send(500, {"error": f"{type(error).__name__}: {error}"})

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4880)
    parser.add_argument("--meshcat-port", type=int, default=7100)
    args = parser.parse_args()
    from dashboard.meshcat_view import MeshcatViews
    Handler.views = MeshcatViews(ROBOT, host=args.host, first_port=args.meshcat_port)
    Handler.jobs = Jobs()
    Handler.training = Training()
    Handler.training.simulate = lambda body: Handler.jobs.run(simulate_episode, body).result(timeout=60)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Physics workspace http://{args.host}:{args.port} · {ROBOT.n_joints}-link arm", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        server.serve_forever()
    finally:
        Handler.jobs.closed = True
        Handler.jobs.pool.shutdown(wait=False, cancel_futures=True)
        Handler.views.close()

if __name__ == "__main__":
    main()
