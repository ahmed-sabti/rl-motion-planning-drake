import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pydrake.common import Parallelism

from planar_physics.clearance import exact_clearance
from planar_physics.drake_model import get_checker
from planar_physics.contract import DEFAULT_CONTRACT, SCENARIO, VARIANT
from planar_physics.fixtures import ROBOT_EQUAL_030, SCENES
from planar_physics.pose_availability import postures

ROBOT = ROBOT_EQUAL_030
FORMAT = 'pose-count-map-v2'
CELL_BLOCK = 4000
MAX_BLOCK_CANDIDATES = 4_000_000


def counts(robot, scene, points, phis, *, margin_m, workers=1, statistics=None):
    owner, q = postures(robot, points, phis)
    solutions = len(q)
    inside = np.all((q >= robot.lo_rad) & (q <= robot.hi_rad), axis=1) if len(q) else np.zeros(0, bool)
    owner, q = owner[inside], q[inside]
    free = np.zeros(len(q), dtype=bool)
    if len(q):
        checker = get_checker(robot, scene, padding=margin_m + DEFAULT_CONTRACT.numerical_slack_m)
        try:
            free[:] = checker.CheckConfigsCollisionFree(q, parallelize=Parallelism(workers))
        except RuntimeError:
            for i, config in enumerate(q):
                try:
                    free[i] = checker.CheckConfigCollisionFree(config)
                except RuntimeError:
                    free[i] = False
        free[free] &= exact_clearance(robot, scene, q[free]) >= margin_m
    value = np.bincount(owner[free], minlength=len(points)).astype(int)
    if statistics is not None:
        for name, number in {'grid_points': len(points), 'ik_solutions': solutions,
                             'joint_limit_rejected': solutions - len(q),
                             'checked_postures': len(q), 'free_postures': int(value.sum())}.items():
            statistics[name] = statistics.get(name, 0) + number
    return value


def grid(cell_m, reach_m):
    n = int(np.floor(reach_m/cell_m + 1e-10))
    edge = (np.arange(-n, n) + .5)*cell_m
    xs, ys = np.meshgrid(edge, edge, indexing='xy')
    return edge, np.stack([xs.ravel(), ys.ravel()], axis=1)


def _phis(n_phi):
    return -np.pi + (np.arange(n_phi) + .5)*2.*np.pi/n_phi


def _count_block(args):
    robot, scene, start, points, phis, margin_m = args
    totals = {}
    value = counts(robot, scene, points, phis, margin_m=margin_m, statistics=totals)
    return start, value, totals


class MapCancelled(Exception):
    pass


def grid_counts(robot, scene, points, phis, *, margin_m, workers=1, statistics=None,
                checkpoint_dir=None, progress=None, cancelled=None, block_cells=None):
    size = min(CELL_BLOCK, max(1, MAX_BLOCK_CANDIDATES//(2*len(phis)))) if robot.n_joints == 3 else CELL_BLOCK
    if block_cells is not None:
        if not isinstance(block_cells, int) or block_cells < 1:
            raise ValueError('The checkpoint block size must be a positive integer.')
        size = min(size, block_cells)
    directory = None if checkpoint_dir is None else Path(checkpoint_dir)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        identity = {'robot': robot.digest(), 'scene': None if scene is None else scene.digest(),
                    'points': hashlib.sha256(np.asarray(points).tobytes()).hexdigest(),
                    'phis': hashlib.sha256(np.asarray(phis).tobytes()).hexdigest(),
                    'margin_m': margin_m, 'numerical_slack_m': DEFAULT_CONTRACT.numerical_slack_m,
                    'block_size': size, 'format': FORMAT}
        manifest = directory/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Saved map work belongs to different inputs. Use another checkpoint directory.')
        manifest.write_text(json.dumps(identity, sort_keys=True))
    value, pending, done = np.zeros(len(points), dtype=int), [], 0
    totals = {}
    names = ('grid_points', 'ik_solutions', 'joint_limit_rejected', 'checked_postures', 'free_postures')
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise MapCancelled('Heat map paused. Completed blocks are retained for resuming.')
    def accept(start, result, stats):
        nonlocal done
        value[start:start+len(result)] = result
        done += len(result)
        for name in names:
            totals[name] = totals.get(name, 0) + int(stats[name])
    for start in range(0, len(points), size):
        check_cancelled()
        block = points[start:start+size]
        saved = None if directory is None else directory/f'{start:09d}.npz'
        if saved is not None and saved.exists():
            with np.load(saved, allow_pickle=False) as data:
                result = data['counts']
                stats = dict(zip(names, data['statistics']))
                candidates = 2 if robot.n_joints == 2 else 2*len(phis)
                if (result.shape != (len(block),) or not np.issubdtype(result.dtype, np.integer)
                        or np.any(result < 0) or np.any(result > candidates)
                        or stats['grid_points'] != len(block)
                        or stats['free_postures'] != int(result.sum())
                        or not 0 <= stats['free_postures'] <= stats['checked_postures'] <= stats['ik_solutions']
                        or stats['joint_limit_rejected'] != stats['ik_solutions'] - stats['checked_postures']):
                    raise ValueError(f'Invalid saved map block: {saved}')
                accept(start, result, stats)
        else:
            pending.append((robot, scene, start, block, phis, margin_m))
    if progress is not None:
        progress(done, len(points), totals)
    def consume(results):
        for start, result, stats in results:
            if directory is not None:
                saved = directory/f'{start:09d}.npz'
                temporary = saved.with_suffix(f'.{os.getpid()}.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, counts=result, statistics=[stats[name] for name in names])
                temporary.replace(saved)
            accept(start, result, stats)
            if progress is not None:
                progress(done, len(points), totals)
            check_cancelled()
    if workers > 1 and pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for start in range(0, len(pending), 2*workers):
                check_cancelled()
                consume(pool.map(_count_block, pending[start:start+2*workers]))
    else:
        consume(map(_count_block, pending))
    if statistics is not None:
        statistics.update(totals)
    return value


def map_info(value, *, candidates, cell_m, source, statistics=None):
    value = np.asarray(value)
    info = {'candidates': candidates, 'cell_m': cell_m, 'source': source,
            'grid_points': int(value.size), 'free_postures': int(value.sum())}
    checked = statistics.get('checked_postures') if isinstance(statistics, dict) else None
    if (isinstance(checked, int) and not isinstance(checked, bool)
            and info['free_postures'] <= checked <= info['grid_points']*candidates
            and statistics.get('grid_points') == info['grid_points']
            and statistics.get('free_postures') == info['free_postures']):
        info['checked_postures'] = checked
    return info


def build_scene(robot, scene, cell_m, n_phi, *, scene_name='live', margin_m=DEFAULT_CONTRACT.margin_m, workers=1,
                checkpoint_dir=None, progress=None, cancelled=None, block_cells=None):
    edge, points = grid(cell_m, robot.reach_m)
    statistics = {}
    value = grid_counts(robot, scene, points, _phis(n_phi), margin_m=margin_m, workers=workers,
                        statistics=statistics, checkpoint_dir=checkpoint_dir, progress=progress,
                        cancelled=cancelled, block_cells=block_cells)
    return {'format': FORMAT, 'scene': scene_name, 'cell_m': cell_m, 'robot_variant': VARIANT,
            'robot_digest': robot.digest(), 'scene_digest': None if scene is None else scene.digest(),
            'scene_geometry': None if scene is None else asdict(scene),
            'fixed_goals': SCENARIO.goals,
            'numerical_slack_m': DEFAULT_CONTRACT.numerical_slack_m,
            'n_phi': None if robot.n_joints == 2 else n_phi,
            'sampling': 'analytic-elbow-branches' if robot.n_joints == 2 else 'tool-orientations',
            'candidates': 2 if robot.n_joints == 2 else 2*n_phi, 'margin_m': margin_m,
            'edge_m': [float(v) for v in edge], 'shape': [len(edge), len(edge)],
            'counts': value.reshape(len(edge), len(edge)).tolist(), 'statistics': statistics}


def build(scene_name, cell_m, n_phi, *, margin_m=DEFAULT_CONTRACT.margin_m, workers=1,
          checkpoint_dir=None, progress=None, block_cells=None):
    scene = SCENES[scene_name] if scene_name != 'none' else None
    return build_scene(ROBOT, scene, cell_m, n_phi, scene_name=scene_name, margin_m=margin_m, workers=workers,
                       checkpoint_dir=checkpoint_dir, progress=progress, block_cells=block_cells)


def cache_path(robot, scene, directory):
    scene_id = 'none' if scene is None else scene.digest()[:12]
    return Path(directory)/f'pose-map-{VARIANT}-{robot.digest()[:12]}-{scene_id}.json'


@contextmanager
def map_lock(output):
    path = Path(output).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def progress_writer(robot, scene, cell_m, n_phi, output):
    last = 0.
    def write(done, total, statistics):
        nonlocal last
        now = time.time()
        if done != total and now - last < 10:
            return
        last = now
        body = {'status': 'computing' if done < total else 'saving', 'updated_at': now,
                'robot_digest': robot.digest(), 'scene_digest': None if scene is None else scene.digest(),
                'cell_m': cell_m, 'n_phi': None if robot.n_joints == 2 else n_phi,
                'completed_cells': done, 'grid_points': total, 'statistics': statistics}
        save_map(body, Path(output).with_suffix('.progress.json'))
        print(json.dumps({'completed_cells': done, 'grid_points': total,
                          'percent': round(100*done/total, 2)}), flush=True)
    return write


def save_map(body, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(body, separators=(',', ':')))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='How many postures the arm can hold at each point of the workspace (fig:haltungen).')
    parser.add_argument('--scene', default='nominal', choices=[*SCENES, 'none'])
    parser.add_argument('--cell-m', type=float, default=.001)
    parser.add_argument('--n-phi', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path, help='Save completed blocks and resume the same inputs.')
    parser.add_argument('--block-cells', type=int, help='Maximum cells per checkpoint block. Live dashboard work uses 16.')
    args = parser.parse_args(argv)
    if not np.isfinite(args.cell_m) or args.cell_m <= 0 or args.n_phi <= 0 or args.workers <= 0:
        parser.error('Cell size, orientation count and worker count must be positive.')
    if args.block_cells is not None and args.block_cells < 1:
        parser.error('Block size must be positive.')
    scene = SCENES[args.scene] if args.scene != 'none' else None
    if args.output:
        with map_lock(args.output):
            progress = progress_writer(ROBOT, scene, args.cell_m, args.n_phi, args.output)
            try:
                body = build(args.scene, args.cell_m, args.n_phi, workers=args.workers,
                             checkpoint_dir=args.checkpoint_dir, progress=progress, block_cells=args.block_cells)
                save_map(body, args.output)
                args.output.with_suffix('.progress.json').unlink(missing_ok=True)
            except Exception as error:
                path = args.output.with_suffix('.progress.json')
                status = json.loads(path.read_text()) if path.exists() else {}
                save_map({**status, 'status': 'error', 'reason': str(error), 'updated_at': time.time()}, path)
                raise
    else:
        body = build(args.scene, args.cell_m, args.n_phi, workers=args.workers,
                     checkpoint_dir=args.checkpoint_dir, block_cells=args.block_cells)
    value = np.asarray(body['counts'])
    print(json.dumps({'scene': args.scene, 'cells': int(value.size), 'reachable': int((value > 0).sum()),
                      'max': int(value.max())}), flush=True)
    return body


if __name__ == '__main__':
    main()
