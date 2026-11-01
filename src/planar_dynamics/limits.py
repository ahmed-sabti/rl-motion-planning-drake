import numpy as np
from pydrake.trajectories import PathParameterizedTrajectory

from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.drake_model import get_model

from .plant import inverse_dynamics


def _acceleration_extrema(curve, span, dt, *, refine=40):
    jerk = curve.MakeDerivative(3)
    fine = np.linspace(0., span, max(4, int(np.ceil(span/(dt/16.)))+1))
    values = jerk.vector_values(fine)
    lo_i, hi_i = np.nonzero(np.signbit(values[:, :-1]) != np.signbit(values[:, 1:]))
    if not len(hi_i):
        return np.empty(0)
    joints = lo_i
    lo, hi = fine[hi_i].copy(), fine[hi_i+1].copy()
    for _ in range(refine):
        mid = .5*(lo+hi)
        at_lo = jerk.vector_values(lo)[joints, np.arange(len(joints))]
        at_mid = jerk.vector_values(mid)[joints, np.arange(len(joints))]
        same = np.signbit(at_lo) == np.signbit(at_mid)
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    return np.clip(.5*(lo+hi), 0., span)


def sample_curve(curve, dt=DEFAULT_CONTRACT.dt_s):
    from .optimize import max_degree
    span = float(curve.end_time())
    steps = max(2, int(np.ceil(span/dt))+1)
    t = np.linspace(0., span, steps)
    segments = getattr(curve, 'get_segment_times', None)
    if isinstance(curve, PathParameterizedTrajectory):
        segments = curve.time_scaling().get_segment_times
    if segments is not None:
        breaks = np.clip(np.asarray(segments(), dtype=float), 0., span)
        sides = np.clip(np.r_[np.nextafter(breaks, -np.inf), breaks,
                              np.nextafter(breaks, np.inf)], 0., span)
        t = np.unique(np.concatenate([t, sides]))
    degree = max_degree(curve)
    if (degree is None or degree > 3) and span > 0.:
        t = np.unique(np.concatenate([t, _acceleration_extrema(curve, span, dt)]))
    first, second = curve.MakeDerivative(), curve.MakeDerivative(2)
    return t, curve.vector_values(t).T, first.vector_values(t).T, second.vector_values(t).T


def _torque_coefficients(robot, q, dq, ddq):
    hold = inverse_dynamics(robot, q, np.zeros_like(dq), np.zeros_like(ddq))
    one = inverse_dynamics(robot, q, dq, ddq)
    half = inverse_dynamics(robot, q, dq/2., ddq/4.)
    b_coef = 4.*half - 3.*hold - one
    d_coef = one - hold - b_coef
    return hold, b_coef, d_coef


def legality_scale(robot, curve, dt=DEFAULT_CONTRACT.dt_s, tolerance=1e-9, slack=1e-3):
    t, q, dq, ddq = sample_curve(curve, dt)
    if not all(np.isfinite(a).all() for a in (t, q, dq, ddq)):
        return np.inf
    plant = get_model(robot).plant
    v_max = np.asarray(plant.GetVelocityUpperLimits(), dtype=float)
    a_max = np.asarray(plant.GetAccelerationUpperLimits(), dtype=float)
    peak_v = np.max(np.abs(dq)/v_max)/(1.+slack) if len(dq) else 0.
    peak_a = np.max(np.abs(ddq)/a_max)/(1.+slack) if len(ddq) else 0.
    scale = max(1., float(peak_v), float(np.sqrt(peak_a)) if peak_a > 0 else 1.)
    hold, b_coef, d_coef = _torque_coefficients(robot, q, dq, ddq)
    caps = np.asarray(robot.max_joint_torque_nm, dtype=float)
    if np.any(np.abs(hold) > caps + tolerance):
        return np.inf
    lo, hi = float(scale), float(scale)
    while np.any(np.abs(hold + b_coef/hi + d_coef/hi**2) > caps + tolerance):
        hi *= 2.
        if hi > 1e6:
            return np.inf
    for _ in range(60):
        mid = .5*(lo+hi)
        if np.any(np.abs(hold + b_coef/mid + d_coef/mid**2) > caps + tolerance):
            lo = mid
        else:
            hi = mid
    return float(hi)
