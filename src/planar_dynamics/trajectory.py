import numpy as np
from pydrake.polynomial import Polynomial
from pydrake.trajectories import PathParameterizedTrajectory, PiecewisePolynomial
from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.drake_model import get_model


def waypoint_trajectory(robot, points, scale=1.0):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != robot.n_joints or len(points) < 2
            or not np.isfinite(points).all() or not np.isfinite(scale) or scale < 1.0):
        raise ValueError("a trajectory needs finite joint waypoints and a scale of at least one")
    if np.any(points < robot.lo_rad) or np.any(points > robot.hi_rad):
        raise ValueError("trajectory waypoints exceed joint limits")
    points = points[np.r_[True, np.any(np.abs(np.diff(points, axis=0)) > 1e-12, axis=1)]]
    if len(points) == 1:
        return PiecewisePolynomial.ZeroOrderHold([0., scale*DEFAULT_CONTRACT.dt_s],
                                                 np.repeat(points.T, 2, axis=1))
    delta = np.diff(points, axis=0)
    plant = get_model(robot).plant
    duration = scale * np.maximum.reduce((
        np.full(len(delta), DEFAULT_CONTRACT.dt_s),
        1.875 * np.max(np.abs(delta) / plant.GetVelocityUpperLimits(), axis=1),
        np.sqrt(5.7735026919 * np.max(np.abs(delta) / plant.GetAccelerationUpperLimits(), axis=1))))
    polynomials = [np.array([[Polynomial([q, 0., 0., 10*d/h**3, -15*d/h**4, 6*d/h**5])]
                            for q, d in zip(a, step)])
                   for a, step, h in zip(points[:-1], delta, duration)]
    return PiecewisePolynomial(polynomials, np.r_[0., np.cumsum(duration)].tolist())


def trajectory_record(curve):
    if isinstance(curve, PathParameterizedTrajectory):
        if not all(isinstance(part, PiecewisePolynomial) for part in (curve.path(), curve.time_scaling())):
            raise ValueError('trajectory records require a polynomial path and polynomial time scaling')
        return {**trajectory_record(curve.path()),
                **{'scaling_'+key: value for key, value in trajectory_record(curve.time_scaling()).items()}}
    return dict(reference_t=np.asarray(curve.get_segment_times()),
                reference_coefficients=curve.polynomials[:, :, 0, :])


def rescale_trajectory(curve, factor):
    if not np.isfinite(factor) or factor < 1.:
        raise ValueError('a slowdown must be finite and at least one')
    if factor == 1.:
        return curve
    if isinstance(curve, PathParameterizedTrajectory):
        scaling = curve.time_scaling().Clone()
        scaling.ScaleTime(factor)
        return PathParameterizedTrajectory(curve.path(), scaling)
    result = curve.Clone()
    result.ScaleTime(factor)
    return result
