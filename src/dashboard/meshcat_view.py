from collections import OrderedDict
from functools import cache
from pathlib import Path
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event
import base64
import io
import json
import math
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from PIL import Image
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.tri import Triangulation

from pydrake.common import MemoryFile
from pydrake.geometry import (IllustrationProperties, InMemoryMesh, Mesh, Meshcat, MeshcatParams,
    MeshcatVisualizer, MeshcatVisualizerParams, Rgba, Role, RoleAssign, Sphere)
from pydrake.math import RigidTransform
from pydrake.perception import PointCloud

from planar_physics.drake_model import build_model
from planar_physics.contract import DEFAULT_CONTRACT, VARIANT
from planar_physics.kinematics import joint_positions
from dashboard.search_view import draw_search_tree
from dashboard.fixed_goals import GOALS, goal_names

COLORS = ((.157, .490, .424), (.792, .506, .267), (.447, .451, .710), (.769, .361, .522))
MUTED = (.47, .52, .50)
INK = (.10, .13, .12)

def _dense(polygon, step=.004):
    polygon = np.asarray(polygon, dtype=float)
    parts = []
    for a, b in zip(polygon[:-1], polygon[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b-a)/step)))
        parts.append(a + np.arange(count)[:, None]/count*(b-a))
    parts.append(polygon[-1:])
    return np.vstack(parts)


@cache
def marker_height(urdf):
    found = re.search(r'name="tip_marker".*?<origin xyz="[-\d.]+ [-\d.]+ ([-\d.]+)"',
                      urdf, re.S)
    return float(found.group(1)) if found else 0.


POSE_MAP_CELL_M, POSE_MAP_N_PHI, POSE_MAP_WORKERS = .001, 5000, max(1, min(6, (os.cpu_count() or 1) - 2))
POSE_MAP_DIR = Path(__file__).resolve().parents[2]/'data'/'pose-maps'
POSE_MAP_MAX_CELLS, POSE_MAP_MAX_BYTES = 4_000_000, 60_000_000
POSE_MAP_FRACTION_MAX = .6
POSE_MAP_ONE = (1., 0., .85)
POSE_MAP_STOPS = np.array([[.80, .20, .25], [.93, .60, .18],
                           [.92, .83, .28], [.15, .68, .43]])


def _recorded_pose_map(scene, robot):
    from dashboard.pose_map import cache_path, map_info
    beste = None
    expected = cache_path(robot, scene, POSE_MAP_DIR)
    paths = [expected] if expected.is_file() else sorted(POSE_MAP_DIR.glob('pose-map-*.json'))
    for pfad in paths:
        if pfad.stat().st_size > POSE_MAP_MAX_BYTES:
            continue
        try:
            with pfad.open() as datei:
                body = json.load(datei)
        except (OSError, ValueError):
            continue
        if body.get('format') != 'pose-count-map-v2' or body.get('robot_variant', '3r') != VARIANT:
            continue
        if robot.n_joints == 2 and (body.get('candidates') != 2
                or body.get('sampling') != 'analytic-elbow-branches' or body.get('n_phi') is not None):
            continue
        if robot.n_joints == 3 and (body.get('sampling') != 'tool-orientations'
                or not isinstance(body.get('n_phi'), int) or body['n_phi'] < POSE_MAP_N_PHI
                or body.get('candidates') != 2*body['n_phi']):
            continue
        if body.get('cell_m', float('inf')) > POSE_MAP_CELL_M:
            continue
        if body.get('robot_digest') != robot.digest() or body.get('scene_digest') != (None if scene is None else scene.digest()):
            continue
        if (body.get('margin_m') != DEFAULT_CONTRACT.margin_m
                or body.get('numerical_slack_m') != DEFAULT_CONTRACT.numerical_slack_m
                or body.get('candidates', 0) <= 0):
            continue
        edge = np.asarray(body['edge_m'], dtype=float)
        value = np.asarray(body['counts'])
        if (value.size > POSE_MAP_MAX_CELLS or value.shape != (len(edge), len(edge))
                or len(edge) < 2 or not np.isfinite(value).all()
                or np.any(value < 0) or np.any(value > body['candidates'])
                or not np.allclose(np.diff(edge), body['cell_m'])):
            continue
        if beste is None or (body['cell_m'], -body['candidates']) < (beste['cell_m'], -beste['candidates']):
            beste = body
    if beste is None:
        return None, None, None, None
    edge = np.asarray(beste['edge_m'], dtype=float)
    dtype = np.uint16 if beste['candidates'] <= np.iinfo(np.uint16).max else np.uint32
    value = np.asarray(beste['counts'], dtype=dtype).ravel()
    return edge, None, value, map_info(
        value, candidates=beste['candidates'], cell_m=beste['cell_m'], source='recorded',
        statistics=beste.get('statistics'))


def _pose_texture_mesh(edge, values, candidates):
    cells = np.asarray(values).reshape(len(edge), len(edge))
    colours = np.rint(255*_pose_colour(cells[::-1].ravel(), candidates)).astype(np.uint8)
    png = io.BytesIO()
    Image.fromarray(colours.reshape(len(edge), len(edge), 3)).save(png, format='PNG')
    h = float(edge[1] - edge[0])/2
    low, high = float(edge[0] - h), float(edge[-1] + h)
    positions = np.array([[low, -.01, -low], [high, -.01, -low],
                          [high, -.01, -high], [low, -.01, -high]], dtype='<f4')
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype='<f4')
    indices = np.array([0, 1, 2, 0, 2, 3], dtype='<u2')
    buffer = positions.tobytes() + uv.tobytes() + indices.tobytes()
    encoded = lambda data: base64.b64encode(data).decode('ascii')
    body = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
            'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'TEXCOORD_0': 1},
                                                            'indices': 2, 'material': 0}]}],
            'buffers': [{'byteLength': len(buffer), 'uri': 'data:application/octet-stream;base64,' + encoded(buffer)}],
            'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 48},
                            {'buffer': 0, 'byteOffset': 48, 'byteLength': 32},
                            {'buffer': 0, 'byteOffset': 80, 'byteLength': 12}],
            'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3',
                           'min': [low, -.01, -high], 'max': [high, -.01, -low]},
                          {'bufferView': 1, 'componentType': 5126, 'count': 4, 'type': 'VEC2'},
                          {'bufferView': 2, 'componentType': 5123, 'count': 6, 'type': 'SCALAR'}],
            'samplers': [{'magFilter': 9728, 'minFilter': 9728, 'wrapS': 33071, 'wrapT': 33071}],
            'images': [{'uri': 'data:image/png;base64,' + encoded(png.getvalue())}],
            'textures': [{'sampler': 0, 'source': 0}],
            'materials': [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                                       'metallicFactor': 0, 'roughnessFactor': 1},
                           'extensions': {'KHR_materials_unlit': {}}}],
            'extensionsUsed': ['KHR_materials_unlit']}
    return Mesh(InMemoryMesh(mesh_file=MemoryFile(json.dumps(body), '.gltf', 'pose-map.gltf')))


def _pose_colour(counts, candidates=1000):
    counts = np.asarray(counts, dtype=float)
    fraction = np.clip(counts/float(candidates)/POSE_MAP_FRACTION_MAX, 0., 1.)
    stops = np.linspace(0., 1., len(POSE_MAP_STOPS))
    rgb = np.column_stack([np.interp(fraction, stops, channel) for channel in POSE_MAP_STOPS.T])
    rgb[counts <= 0] = (1., 1., 1.)
    rgb[counts == 1] = POSE_MAP_ONE
    return rgb


def _scene_key(scene):
    wall = getattr(scene, 'wall', None)
    teile = [] if wall is None else [wall.plane_x_m, wall.half_height_m, wall.thickness_m,
                                     *wall.gap_centres_m, *wall.gap_widths_m]
    return tuple(round(float(v), 6) for v in teile)


def _xyz(points, z=0.):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.vstack((points.T, np.full(len(points), z)))

class _ModelView:
    def __init__(self, meshcat, spec, scene, prefix, color=None, alpha=1.):
        self.model = build_model(spec, scene)
        model = self.model
        for body in model.link_bodies:
            for geometry in model.plant.GetVisualGeometriesForBody(body):
                if color is None:
                    model.scene_graph.RemoveRole(model.scene_context,
                        model.plant.get_source_id(), geometry, Role.kIllustration)
                else:
                    props = IllustrationProperties()
                    inspector = model.scene_graph.model_inspector()
                    name = inspector.GetName(geometry)
                    if 'marker' in name:
                        xy = inspector.GetPoseInFrame(geometry).translation()[:2]
                        model.scene_graph.ChangeShape(model.scene_context,
                            model.plant.get_source_id(), geometry,
                            Sphere(spec.link_radius_m + .002), RigidTransform([*xy, 0]))
                    shade = (1., 1., 1.) if 'marker' in name else color
                    props.AddProperty("phong", "diffuse", Rgba(*shade, alpha))
                    model.scene_graph.AssignRole(model.scene_context,
                        model.plant.get_source_id(), geometry, props, RoleAssign.kReplace)
        self.visualizer = MeshcatVisualizer(meshcat, MeshcatVisualizerParams(prefix=prefix))
        self.context = self.visualizer.CreateDefaultContext()

    def publish(self, q, time=0.):
        self.model.plant.SetPositions(self.model.plant_context, q)
        self.context.SetTime(time)
        self.visualizer.query_object_input_port().FixValue(self.context,
            self.model.scene_graph.get_query_output_port().Eval(self.model.scene_context))
        self.visualizer.ForcedPublish(self.context)

_MAPS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="posemap")
_MAP_CACHE, _MAP_PENDING, _MAP_FAILED = OrderedDict(), {}, {}


def _compute_map(scene, spec, stop):
    from dashboard.pose_map import (MapCancelled, build_scene, cache_path, map_info,
                                    map_lock, progress_writer, save_map)
    output = cache_path(spec, scene, POSE_MAP_DIR)
    with map_lock(output):
        if stop.is_set():
            raise MapCancelled('Heat map paused before computation.')
        edge, points, value, info = _recorded_pose_map(scene, spec)
        if value is None:
            try:
                body = build_scene(spec, scene, POSE_MAP_CELL_M, POSE_MAP_N_PHI, workers=POSE_MAP_WORKERS,
                                   checkpoint_dir=output.with_suffix('.work'), cancelled=stop.is_set, block_cells=16,
                                   progress=progress_writer(spec, scene, POSE_MAP_CELL_M, POSE_MAP_N_PHI, output))
            except MapCancelled:
                path = output.with_suffix('.progress.json')
                progress = _map_progress(scene, spec)
                save_map({**progress, 'status': 'paused', 'updated_at': time.time()}, path)
                raise
            edge = np.asarray(body['edge_m'])
            value = np.asarray(body['counts'], dtype=np.uint16).ravel()
            info = map_info(value, candidates=body['candidates'], cell_m=body['cell_m'], source='live',
                            statistics=body['statistics'])
            save_map(body, output)
            output.with_suffix('.progress.json').unlink(missing_ok=True)
    return edge, points, value, info


def _cancel_maps(key=None):
    for other, (job, stop) in tuple(_MAP_PENDING.items()):
        if key is None or other == key:
            stop.set()
            job.cancel()


def _pause_maps():
    _cancel_maps()
    deadline = time.monotonic()+15
    for key, (job, _) in tuple(_MAP_PENDING.items()):
        try:
            job.result(timeout=max(0., deadline-time.monotonic()))
        except TimeoutError:
            raise ValueError('The heat map is still stopping. Retry the run after a few seconds.')
        except Exception:
            pass
        del _MAP_PENDING[key]


def _map_progress(scene, spec):
    from dashboard.pose_map import cache_path
    path = cache_path(spec, scene, POSE_MAP_DIR).with_suffix('.progress.json')
    try:
        body = json.loads(path.read_text())
        if (body.get('robot_digest') == spec.digest()
                and body.get('scene_digest') == (None if scene is None else scene.digest())
                and body.get('cell_m') == POSE_MAP_CELL_M
                and body.get('n_phi') == (None if spec.n_joints == 2 else POSE_MAP_N_PHI)
                and (body.get('externally_managed') is True or body.get('status') == 'error'
                     or time.time() - body.get('updated_at', 0) < 300)):
            return body
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _map_for(scene, spec, busy=False):
    from dashboard.pose_map import MapCancelled
    key = _scene_key(scene)
    if busy:
        _cancel_maps()
    if key in _MAP_CACHE:
        _MAP_CACHE.move_to_end(key)
        return _MAP_CACHE[key]
    for other in [k for k in _MAP_PENDING if k != key]:
        _cancel_maps(other)
        if _MAP_PENDING[other][0].done():
            del _MAP_PENDING[other]
    pending = _MAP_PENDING.get(key)
    job = pending[0] if pending is not None else None
    progress = _map_progress(scene, spec)
    if progress.get('status') == 'error':
        raise RuntimeError(progress.get('reason') or 'The map computation failed. Resume its saved blocks.')
    if job is None and progress.get('status') in ('queued', 'computing', 'saving'):
        return None
    if job is None and busy:
        return None
    if job is None and time.monotonic() - _MAP_FAILED.get(key, -1e9) < 60:
        raise RuntimeError("The heat map for this scene failed; it is retried after a minute.")
    if job is None:
        stop = Event()
        _MAP_PENDING[key] = (_MAPS.submit(_compute_map, scene, spec, stop), stop)
        return None
    if not job.done():
        return None
    del _MAP_PENDING[key]
    try:
        result = job.result()
    except (MapCancelled, CancelledError):
        return None
    except Exception:
        _MAP_FAILED[key] = time.monotonic()
        raise
    _MAP_CACHE[key] = result
    while len(_MAP_CACHE) > 4:
        _MAP_CACHE.popitem(last=False)
    return result


class _View:
    def __init__(self, spec, host, port=None):
        self.spec = spec
        self._pose_map_key = None
        self._map_requested_key = None
        self.pose_map_info = None
        self.meshcat = Meshcat(MeshcatParams(host=host, port=port, show_stats_plot=False))
        reach = spec.reach_m * 1.18
        self.meshcat.Set2dRenderMode(xmin=-reach, xmax=reach, ymin=-reach, ymax=reach)
        self.meshcat.SetCameraPose([*spec.base_xy_m, 1], [*spec.base_xy_m, 0])
        self.models = {}
        self.scene = None
        self.static_key = None
        self.marker_z = marker_height(spec._urdf_text)
        self.meshcat.SetObject("goal", Sphere(.012), Rgba(.79, .51, .27))
        self.meshcat.SetObject("start", Sphere(.012), Rgba(.35, .38, .37))
        angle = np.linspace(0, 2 * np.pi, 361)
        radius = spec.reach_m
        circle = np.asarray(spec.base_xy_m) + radius * np.column_stack((np.cos(angle), np.sin(angle)))
        self._line("reach", circle, MUTED, .65, dashed=True)
        for name, goal in GOALS.items():
            circle = np.asarray(goal) + DEFAULT_CONTRACT.goal_tol_m * np.column_stack((np.cos(angle), np.sin(angle)))
            self._line(f"goals/{name}", circle, MUTED, .85, z=self.marker_z)
        for axis, name in enumerate(("x", "y")):
            endpoints = np.zeros((2, 2))
            endpoints[:, axis] = [-reach, reach]
            self._line(f"axes/{name}", endpoints, MUTED, .55)
            for value in np.arange(-.9, .91, .3):
                point = np.zeros(2)
                point[axis] = round(value, 1)
                tick = np.zeros(2)
                tick[1-axis] = .009
                self._line(f"axes/{name}/ticks/{value:.1f}", [point-tick, point+tick], MUTED)
                if abs(value) > .01:
                    point[1-axis] -= .06
                    self._draw_label(f"axes/{name}/labels/{value:.1f}", f"{value:.1f}".replace(".", ","), point, INK)
            point = np.zeros(2)
            point[axis], point[1-axis] = reach-(.20 if axis else .07), .05
            self._draw_label(f"axes/{name}/unit", f"{name} (m)", point, INK)

    def _draw_label(self, name, text, point, color):
        glyphs = TextPath((0, 0), text, size=.035)
        box = glyphs.get_extents()
        offset = np.asarray(point) - [box.x0+box.width/2, box.y0+box.height/2]
        outline = [np.asarray(polygon) + offset for polygon in glyphs.to_polygons() if len(polygon) > 2]
        if not outline:
            self.meshcat.Delete(name)
            return
        points = np.vstack([_dense(polygon) for polygon in outline])
        try:
            mesh = Triangulation(points[:, 0], points[:, 1])
        except (ValueError, RuntimeError):
            self._segments(name, outline, color)
            return
        contours = [MplPath(polygon) for polygon in outline]
        centres = points[mesh.triangles].mean(axis=1)
        crossings = np.sum([contour.contains_points(centres) for contour in contours], axis=0)
        keep = mesh.triangles[crossings % 2 == 1]
        if not len(keep):
            self._segments(name, outline, color)
            return
        vertices = np.vstack((points.T, np.zeros(len(points))))
        self.meshcat.SetTriangleMesh(name, vertices, keep.T.astype(int), Rgba(*color))

    def _line(self, name, points, color, alpha=1., dashed=False, z=0.):
        points = _xyz(points, z)
        if points.shape[1] < 2:
            self.meshcat.Delete(name)
        elif dashed:
            self.meshcat.SetLineSegments(name, points[:, :-1:2], points[:, 1::2],
                                         2., Rgba(*color, alpha))
        else:
            self.meshcat.SetLine(name, points, 2., Rgba(*color, alpha))

    def _segments(self, name, polylines, color, alpha=1.):
        segments = [(_xyz(p)[:, :-1], _xyz(p)[:, 1:]) for p in polylines if len(p) > 1]
        if segments:
            self.meshcat.SetLineSegments(name, np.hstack([s[0] for s in segments]),
                np.hstack([s[1] for s in segments]), 1.5, Rgba(*color, alpha))
        else:
            self.meshcat.Delete(name)

    def _pose_map(self, scene, show, busy=False):
        key = _scene_key(scene)
        if self._map_requested_key is not None and (not show or self._map_requested_key != key):
            _cancel_maps(self._map_requested_key)
        self._map_requested_key = key if show else None
        try:
            if show and self._pose_map_key != key:
                ready = _map_for(scene, self.spec, busy)
                if ready is None:
                    progress = _map_progress(scene, self.spec)
                    active = progress.get('status') if progress.get('status') in ('queued', 'computing', 'saving') else None
                    self.pose_map_info = {**progress, "status": active or (
                        "waiting" if busy and _scene_key(scene) not in _MAP_PENDING else "computing")}
                    self.meshcat.SetProperty("posemap", "visible", False)
                    self.meshcat.SetProperty("posemap_one", "visible", False)
                    return
                edge, points, value, info = ready
                self.meshcat.Delete('posemap')
                self.meshcat.Delete('posemap_one')
                self.meshcat.SetObject('posemap/cells', _pose_texture_mesh(edge, value, info['candidates']))
                self._pose_map_key = key
                self.pose_map_info = {**info, 'status': 'ready'}
            self.meshcat.SetProperty("posemap", "visible", bool(show))
            self.meshcat.SetProperty("posemap_one", "visible", bool(show))
        except Exception as error:
            self.pose_map_info = {'status': 'error', 'reason': str(error)}
            self.meshcat.SetProperty('posemap', 'visible', False)
            self.meshcat.SetProperty('posemap_one', 'visible', False)

    def update(self, scene, options, run):
        meshcat = self.meshcat
        self._pose_map(scene, options.get("reach"), bool(options.get("busy")))
        cursor = float(options.get("cursor_s", 0.))
        if not math.isfinite(cursor) or cursor < 0:
            raise ValueError("Replay cursor must be finite and nonnegative.")
        q = np.asarray(options["q"], float)
        goal = np.asarray(options["goal"], float)
        if q.shape != (self.spec.n_joints,) or not np.isfinite(q).all():
            raise ValueError("Invalid view joint positions.")
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("Invalid view goal.")
        meshcat.SetProperty("goal", "visible", True)
        scene_name = options.get("scene", "nominal")
        names = goal_names(scene_name)
        preset = scene_name in ("nominal", "free") and any(
            float(np.linalg.norm(goal - np.asarray(GOALS[name], float))) < 1e-9 for name in names)
        meshcat.SetProperty("goals", "visible", preset)
        meshcat.SetProperty("goals/edge", "visible", scene_name == "nominal")
        scene_digest = scene.digest()
        if scene_digest != self.scene:
            meshcat.Delete("world")
            self.models["world"] = _ModelView(meshcat, self.spec, scene, "world")
            self.models["world"].publish(q)
            self.scene = scene_digest
        meshcat.SetTransform("goal", RigidTransform([*goal, self.marker_z]))
        start_tip = np.asarray(joint_positions(self.spec, q))[-1]
        meshcat.SetTransform("start", RigidTransform([*start_tip[:2], self.marker_z]))
        meshcat.SetProperty("start", "visible", True)
        angle = np.linspace(0, 2 * np.pi, 97)
        self._line("goal/tolerance", DEFAULT_CONTRACT.goal_tol_m *
                   np.column_stack((np.cos(angle), np.sin(angle))), (.79, .51, .27))
        meshcat.SetProperty("/Background", "visible", True)
        color = [1., 1., 1.]
        meshcat.SetProperty("/Background/<object>", "top_color", color)
        meshcat.SetProperty("/Background/<object>", "bottom_color", color)
        lanes = run["lanes"] if run else [{"state": "setup"}]*max(1, min(4, int(options.get("lanes", 1))))
        off = {int(i) for i in options.get("off") or [] if isinstance(i, int)}
        lanes = [{"state": "off"} if i in off else lane for i, lane in enumerate(lanes)]

        show_search = bool(options.get("search"))
        static_key = (run["id"] if run else None, tuple(l.get("state") for l in lanes), show_search)
        if static_key != self.static_key:
            meshcat.Delete("overlays")
            self.static_key = None
            for i, lane in enumerate(lanes):
                root = f"overlays/lane{i}"
                graph = lane.get("search") or {} if show_search else {}
                draw_search_tree(meshcat, f"{root}/tree", graph, (COLORS[i], MUTED))
                self._line(f"{root}/paths/reference", lane.get("reference_path", []),
                           COLORS[i], .7, dashed=True)
                self._line(f"{root}/raw/line", lane.get("raw_path", []), COLORS[i], .95, dashed=True)
                raw_nodes = lane.get("raw_nodes") or []
                if raw_nodes:
                    raw_cloud = PointCloud(len(raw_nodes))
                    raw_cloud.mutable_xyzs()[:] = _xyz(raw_nodes)
                    meshcat.SetObject(f"{root}/raw/nodes", raw_cloud, 6., Rgba(*COLORS[i]))
                nodes = lane.get("prepared_nodes", [])
                if nodes:
                    self._line(f"{root}/paths/prepared", nodes, COLORS[i], .9)
                    cloud = PointCloud(len(nodes))
                    cloud.mutable_xyzs()[:] = _xyz(nodes)
                    meshcat.SetObject(f"{root}/paths/nodes", cloud, 8., Rgba(*COLORS[i]))
            self.static_key = static_key
        for i in range(4):
            visible = i < len(lanes) and lanes[i].get("state") != "off"
            meshcat.SetProperty(f"lane{i}", "visible", visible)
            meshcat.SetProperty(f"overlays/lane{i}", "visible", visible)
            if not visible:
                continue
            name, lane = f"lane{i}", lanes[i]
            if name not in self.models:
                self.models[name] = _ModelView(meshcat, self.spec, None, name, COLORS[i])
            frames = lane.get("frames", [])
            physical = cursor
            index = max(0, int(np.searchsorted([f["t"] for f in frames], physical + 1e-9,
                                              side="right")) - 1)
            frame = frames[index] if frames else None
            self.models[name].publish(frame["q"] if frame else q, physical)
            root = f"overlays/{name}"
            search = show_search and bool(lane.get("search"))
            meshcat.SetProperty(f"{root}/tree", "visible", search)
            meshcat.SetProperty(f"{root}/raw", "visible",
                                bool(options.get("found")) and bool(lane.get("raw_path")))
            meshcat.SetProperty(f"{root}/paths", "visible", True)
            meshcat.SetProperty(f"{root}/paths/reference", "visible",
                                bool(options.get("reference", True)))
            for node in ("prepared", "nodes"):
                meshcat.SetProperty(f"{root}/paths/{node}", "visible",
                                    bool(options.get("route", True)))
            self._line(f"{root}/paths/executed", [f["points"][-1] for f in frames[:index + 1]], COLORS[i])
            meshcat.SetProperty(f"{root}/paths/executed", "visible",
                                bool(options.get("executed", True)))
        ghost = bool(run and options.get("ghost"))
        if ghost:
            if "ghost" not in self.models:
                self.models["ghost"] = _ModelView(meshcat, self.spec, None, "ghost", MUTED, .18)
            self.models["ghost"].publish(run["query"]["q"])
        meshcat.SetProperty("ghost", "visible", ghost)
        target = options.get("goal_q")
        preview = target is not None and run is None
        if preview:
            if "goalpose" not in self.models:
                self.models["goalpose"] = _ModelView(meshcat, self.spec, None, "goalpose",
                                                     (.79, .51, .27), .22)
            self.models["goalpose"].publish(np.asarray(target, dtype=float))
        meshcat.SetProperty("goalpose", "visible", bool(preview))
        meshcat.Flush()
        return {"meshcat_url": meshcat.web_url(), "pose_map": self.pose_map_info}

class MeshcatViews:
    def __init__(self, robot_spec, *, host="127.0.0.1", max_views=4, first_port=7000, port_count=21):
        self.spec, self.host, self.max_views = robot_spec, host, max_views
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meshcat")
        self.views = OrderedDict()
        self.ports = [first_port + k for k in range(port_count)]
        self.next_index = 0

    def update(self, view_id, scene_spec, options, run=None):
        if not isinstance(view_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", view_id):
            raise ValueError("Invalid browser view identifier.")
        return self.pool.submit(self._update, view_id, scene_spec, options, run).result()

    def _update(self, view_id, scene, options, run):
        if view_id not in self.views:
            if len(self.views) >= self.max_views:
                self.views.popitem(last=False)
            self.views[view_id] = self._new_view()
        self.views.move_to_end(view_id)
        return self.views[view_id].update(scene, options, run)

    def _new_view(self):
        held = {view.port for view in self.views.values()}
        for _ in range(len(self.ports)):
            port = self.ports[self.next_index]
            self.next_index = (self.next_index + 1) % len(self.ports)
            if port in held:
                continue
            try:
                view = _View(self.spec, self.host, port)
            except RuntimeError:
                continue
            view.port = port
            return view
        raise RuntimeError('No free Meshcat port found.')

    def asset(self, path):
        return self.pool.submit(self._asset, path).result()

    def pause_maps(self):
        self.pool.submit(_pause_maps).result()

    def _asset(self, path):
        for view in reversed(tuple(self.views.values())):
            try:
                with urlopen(f'http://127.0.0.1:{view.port}{path}', timeout=2) as response:
                    data = response.read()
                    mime = 'image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'application/octet-stream'
                    return data, mime
            except (HTTPError, URLError, TimeoutError):
                continue
        return None

    def close(self):
        self.pause_maps()
        self.pool.submit(self.views.clear).result()
        self.pool.shutdown()
