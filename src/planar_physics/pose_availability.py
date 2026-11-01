import numpy as np

DEGENERACY_EPS = 1e-6


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2.*np.pi) - np.pi


def _two_link(robot, points):
    a, b = np.asarray(robot.lengths_m, dtype=float)
    pts = np.asarray(points, dtype=float) - np.asarray(robot.base_xy_m, dtype=float)
    cos2 = (np.einsum('ij,ij->i', pts, pts) - a*a - b*b)/(2.*a*b)
    index = np.flatnonzero(np.abs(cos2) <= 1. - DEGENERACY_EPS)
    if not len(index):
        return np.zeros(0, dtype=int), np.zeros((0, 2))
    angle = np.arctan2(pts[index, 1], pts[index, 0])
    owner, configs = [], []
    for sign in (1., -1.):
        q2 = sign*np.arccos(cos2[index])
        configs.append(np.stack([_wrap(angle - np.arctan2(b*np.sin(q2), a + b*np.cos(q2))), q2], axis=1))
        owner.append(index)
    return np.concatenate(owner), np.concatenate(configs)


def postures(robot, points, phis):
    if robot.n_joints == 2:
        return _two_link(robot, points)
    a, b, c = np.asarray(robot.lengths_m, dtype=float)
    base = np.asarray(robot.base_xy_m, dtype=float)
    pts = np.asarray(points, dtype=float) - base
    owner, configs = [], []
    for phi in np.asarray(phis, dtype=float):
        wrist = pts - c*np.array([np.cos(phi), np.sin(phi)])
        r2 = np.einsum('ij,ij->i', wrist, wrist)
        cos2 = (r2 - a*a - b*b)/(2.*a*b)
        ok = np.abs(cos2) <= 1. - DEGENERACY_EPS
        if not ok.any():
            continue
        cos2 = np.clip(cos2[ok], -1., 1.)
        w = wrist[ok]
        base_angle = np.arctan2(w[:, 1], w[:, 0])
        index = np.flatnonzero(ok)
        for sign in (1., -1.):
            q2 = sign*np.arccos(cos2)
            q1 = base_angle - np.arctan2(b*np.sin(q2), a + b*np.cos(q2))
            q3 = phi - q1 - q2
            owner.append(index)
            configs.append(np.stack([_wrap(q1), q2, _wrap(q3)], axis=1))
    if not configs:
        return np.zeros(0, dtype=int), np.zeros((0, robot.n_joints))
    return np.concatenate(owner), np.concatenate(configs)

