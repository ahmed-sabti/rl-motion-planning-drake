import numpy as np
from planar_physics.certificate import CertifierConfig
from planar_physics.clearance import clearance, is_free
from planar_physics.rng import SeedSpec
from planar_physics.contract import SCENARIO, DEFAULT_CONTRACT as C
from planar_physics.fixtures import ROBOT_EQUAL_030, SCENES
from planar_physics.kinematics import joint_positions
from planar_dynamics.plant import inverse_dynamics

PROTOCOL, GOALS = SCENARIO.protocol, SCENARIO.goals


def holds(robot, scene, q, *, margin_m):
    zero = np.zeros(robot.n_joints)
    if np.any(np.abs(inverse_dynamics(robot, q, zero, zero)) > np.asarray(robot.max_joint_torque_nm)):
        return 'torque'
    if not float(clearance(robot, scene, q)) >= CertifierConfig(margin_m=margin_m).padding(robot):
        return 'hold'
    return None


def held_start(seed, *, margin_m=C.margin_m, max_attempts=4096, scene=None):
    robot, scene = ROBOT_EQUAL_030, scene or SCENES['nominal']
    rng = SeedSpec(seed).stream(PROTOCOL)
    wall = scene.wall or SCENES['nominal'].wall
    near = wall.plane_x_m-wall.thickness_m/2-robot.link_radius_m-margin_m
    rejected = dict(geometry=0, torque=0, hold=0)
    for _ in range(max_attempts):
        q = rng.uniform(robot.lo_rad, robot.hi_rad)
        points = joint_positions(robot, q)
        if points[-1, 0] >= 0 or np.max(points[:, 0]) > near or not is_free(robot, scene, q, margin_m=margin_m):
            rejected['geometry'] += 1
            continue
        why = holds(robot, scene, q, margin_m=margin_m)
        if why:
            rejected[why] += 1
            continue
        return {'seed': int(seed), 'q': q.tolist(), 'rejected_candidates': rejected}
    raise ValueError(f'No validated held start in {max_attempts} candidates: {rejected}')
