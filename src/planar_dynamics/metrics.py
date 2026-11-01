import numpy as np
from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.kinematics import joint_positions


def motion_series(robot, t, q, dq):
    t, q, dq = (np.asarray(a, dtype=float) for a in (t, q, dq))
    points = joint_positions(robot, q)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[:, -1], axis=0), axis=1))]
    joint_arc = np.vstack((np.zeros(robot.n_joints), np.cumsum(np.abs(np.diff(q, axis=0)), axis=0)))
    dt = np.diff(t)
    jerk = np.zeros(len(t))
    if len(t) > 2:
        acceleration = np.diff(dq, axis=0) / dt[:, None]
        jerk[2:] = np.linalg.norm(np.diff(acceleration, axis=0) / ((dt[:-1] + dt[1:])/2)[:, None], axis=1)
    moving = np.max(np.abs(dq), axis=1) > DEFAULT_CONTRACT.hold_speed_rad_s
    travelled = np.flatnonzero(moving)
    within = np.arange(len(t)) <= (travelled[-1] if len(travelled) else 0)
    jrms = np.sqrt(np.cumsum((jerk*within)**2) / np.maximum(np.cumsum(within)-2, 1))
    elapsed = t - t[0]
    mean_speed = np.divide(distance, elapsed, out=np.zeros_like(distance), where=elapsed > 0)
    return dict(points=points, distance=distance, joint_arc=joint_arc,
                jerk=jerk, jrms=jrms, elapsed=elapsed, mean_speed=mean_speed)
