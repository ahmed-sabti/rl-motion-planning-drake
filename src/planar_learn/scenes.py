from dataclasses import dataclass

import numpy as np

from planar_physics.certificate import CertifierConfig
from planar_physics.contract import DEFAULT_CONTRACT, VARIANT
from planar_physics.fixtures import ROBOT_EQUAL_030
from planar_physics.scene import Scene, WallSpec

SCENE_PROFILE = '3r-original-v1' if VARIANT == '3r' else '2r-aperture-clearance-v4'


def _ranges():
    reference = dict(plane_x_m=(.34, .48), half_height_m=1., gap_centre_m=(.18, .50),
                     gap_width_m=(.24, .40), goal_clearance_m=.10, goal_inset_m=.05,
                     goal_abs_y_max_m=.70)
    if VARIANT == '3r':
        return reference
    robot = ROBOT_EQUAL_030
    scale = robot.reach_m / .90
    cfg = CertifierConfig(margin_m=DEFAULT_CONTRACT.margin_m)
    reference_padding = (cfg.margin_m + cfg.numerical_slack_m
                         + np.linalg.norm([.90, .60, .30])*cfg.edge_step_rad/2.)
    shell3 = robot.link_radius_m + .04/2. + reference_padding
    shell2 = robot.link_radius_m + .04/2. + cfg.padding(robot)
    scaled = {key: tuple(scale*x for x in value) if isinstance(value, tuple) else scale*value
              for key, value in reference.items()}
    scaled['gap_width_m'] = tuple(scale*(width-2.*shell3)+2.*shell2
                                  for width in reference['gap_width_m'])
    first_link_envelope = robot.lengths_m[0]+shell2
    scaled['plane_x_m'] = (first_link_envelope+.008, first_link_envelope+.028)
    return scaled


RANGES = _ranges()


@dataclass(frozen=True)
class SceneSpec:
    plane_x_m: tuple = RANGES['plane_x_m']
    half_height_m: float = RANGES['half_height_m']
    thickness_m: float = .04
    gap_centre_m: tuple | None = RANGES['gap_centre_m'] if VARIANT == '3r' else None
    gap_width_m: tuple = RANGES['gap_width_m']
    wall_probability: float = .8
    goal_clearance_m: float = RANGES['goal_clearance_m']
    goal_inset_m: float = RANGES['goal_inset_m']
    goal_abs_y_max_m: float = RANGES['goal_abs_y_max_m']
    central_web_m: tuple | None = (.005, .010) if VARIANT == '2r' else None


def sample_scene(rng, spec, robot):
    wall = None
    if rng.random() < spec.wall_probability:
        floor = 2.*robot.link_radius_m + spec.thickness_m
        widths = np.maximum(rng.uniform(*spec.gap_width_m, size=2), floor)
        if robot.n_joints == 2:
            inner = rng.uniform(*spec.central_web_m)/2.
            centres = (widths/2.+inner)*[1., -1.]
        else:
            centres = rng.uniform(*spec.gap_centre_m, size=2)*[1., -1.]
        room = spec.half_height_m-widths/2.
        centres = np.clip(centres, -room, room)
        if centres[0]-widths[0]/2. <= centres[1]+widths[1]/2.:
            centres = np.array([1., -1.])*(widths.sum()/2.+1e-3)
        wall = WallSpec(plane_x_m=float(rng.uniform(*spec.plane_x_m)),
                        half_height_m=spec.half_height_m, thickness_m=spec.thickness_m,
                        gap_centres_m=[float(centres[1]), float(centres[0])],
                        gap_widths_m=[float(widths[1]), float(widths[0])])
    return Scene(wall=wall)


def openings(scene, robot):
    wall = scene.wall
    if wall is None:
        return []
    radius = robot.link_radius_m+wall.thickness_m/2.
    rows = []
    for centre, width in zip(wall.gap_centres_m, wall.gap_widths_m):
        half = width/2.-radius
        if half > 0:
            rows.append((float(centre), float(half)))
    return sorted(rows, key=lambda row: -row[1])


def route(scene, robot, tip, goal, committed=None, hysteresis_m=.10, keep=.25):
    tip, goal = np.asarray(tip, dtype=float), np.asarray(goal, dtype=float)
    wall = scene.wall
    if wall is None or tip[0] > wall.plane_x_m or goal[0] <= wall.plane_x_m:
        return goal, float(np.linalg.norm(goal-tip)), None
    holes = openings(scene, robot)
    if not holes:
        return goal, float(np.linalg.norm(goal-tip)), None
    far = wall.plane_x_m+wall.thickness_m/2.+robot.link_radius_m
    span = goal[0]-tip[0]
    legs, aims = [], []
    for centre, half in holes:
        room = half*keep
        crossing = tip[1]+(goal[1]-tip[1])*((wall.plane_x_m-tip[0])/(span if abs(span) > 1e-6 else 1e-6))
        aim = np.array([far, float(np.clip(crossing, centre-room, centre+room))])
        aims.append(aim)
        legs.append(float(np.linalg.norm(aim-tip)+np.linalg.norm(goal-aim)))
    best = int(np.argmin(legs))
    if committed is not None and 0 <= committed < len(legs) and legs[committed] <= legs[best]+hysteresis_m:
        best = int(committed)
    return aims[best], legs[best], best


def sample_goal(rng, spec, robot, scene):
    near = 0. if scene.wall is None else scene.wall.plane_x_m+scene.wall.thickness_m/2.
    low = near+spec.goal_clearance_m
    high = robot.reach_m-spec.goal_inset_m-spec.goal_clearance_m
    x = float(rng.uniform(low, high))
    limit = float(np.sqrt(max((robot.reach_m-spec.goal_inset_m)**2-x*x, 0.)))
    span = min(limit, spec.goal_abs_y_max_m)
    return np.array([x, float(rng.uniform(-span, span))])
