import numpy as np

REVERSAL_STEP_S = .01
REVERSAL_DEADBAND_RAD_S = .05


def joint_reversals(dq, deadband=REVERSAL_DEADBAND_RAD_S):
    dq = np.asarray(dq, dtype=float)
    if dq.ndim == 1:
        dq = dq[:, None]
    if (dq.ndim != 2 or not dq.shape[1] or not np.isfinite(dq).all()
            or not np.isfinite(deadband) or deadband < 0):
        raise ValueError('reversal counting needs finite joint velocities and a nonnegative deadband')
    counts = []
    for rates in dq.T:
        signs = np.sign(rates[np.abs(rates) > deadband])
        counts.append(int(np.count_nonzero(signs[1:] != signs[:-1])))
    return counts


def reference_reversal_metrics(record, step_s=REVERSAL_STEP_S):
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError('reversal sampling interval must be positive and finite')
    knots = record.get('scaling_reference_t', record['reference_t'])
    start, end = float(knots[0]), float(knots[-1])
    times = np.linspace(start, end, max(2, int(np.ceil((end-start)/step_s))+1))
    _, dq = reference_samples(record, times)
    counts = joint_reversals(dq)
    return dict(joint_reversals=counts, joint_reversal_count=sum(counts),
                reversal_sample_dt_max_s=step_s, reversal_deadband_rad_s=REVERSAL_DEADBAND_RAD_S)


def polynomial_samples(knots, coefficients, times, derivative=0):
    knots, coefficients, times = map(np.asarray, (knots, coefficients, times))
    if (knots.ndim != 1 or coefficients.ndim != 3 or len(knots) != len(coefficients)+1
            or np.any(np.diff(knots) <= 0) or not np.isfinite(coefficients).all()
            or not np.isfinite(knots).all() or not np.isfinite(times).all()
            or np.any(times < knots[0]-1e-10) or np.any(times > knots[-1]+1e-10)):
        raise ValueError('invalid polynomial record or sample times')
    c = coefficients.copy()
    for _ in range(derivative):
        c = c[:, :, 1:] * np.arange(1, c.shape[-1])
    index = np.clip(np.searchsorted(knots, times, side='right')-1, 0, len(c)-1)
    local = (times-knots[index])[:, None]
    result = np.zeros((len(times), c.shape[1]))
    for power in range(c.shape[-1]-1, -1, -1):
        result = result*local + c[index, :, power]
    return result


def reference_samples(record, times):
    if 'scaling_reference_t' in record:
        s = polynomial_samples(record['scaling_reference_t'],
                               record['scaling_reference_coefficients'], times)[:, 0]
        ds = polynomial_samples(record['scaling_reference_t'],
                                record['scaling_reference_coefficients'], times, 1)[:, 0]
        if np.any(s < record['reference_t'][0]-1e-8) or np.any(s > record['reference_t'][-1]+1e-8):
            raise ValueError('time scaling leaves the recorded path')
        s = np.clip(s, record['reference_t'][0], record['reference_t'][-1])
    else:
        s, ds = times, np.ones(len(times))
    q = polynomial_samples(record['reference_t'], record['reference_coefficients'], s)
    dq = polynomial_samples(record['reference_t'], record['reference_coefficients'], s, 1)*ds[:, None]
    return q, dq


def reference_quality(record, dt_s=.001):
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('sampling interval must be positive and finite')
    knots = record.get('scaling_reference_t', record['reference_t'])
    start, end = float(knots[0]), float(knots[-1])
    duration = end-start
    times = start+np.arange(int(np.floor(duration/dt_s))+1)*dt_s
    q, dq = reference_samples(record, times)
    endpoints, _ = reference_samples(record, np.array([start, end]))
    path = np.vstack([q, endpoints[-1]]) if times[-1] < end else q
    jerk = np.diff(dq, n=2, axis=0)/dt_s**2
    rms = float(np.sqrt(np.mean(np.sum(jerk**2, axis=1)))) if len(jerk) else None
    amplitude = float(np.linalg.norm(endpoints[-1]-endpoints[0]))
    return dict(duration_s=duration, sample_dt_s=dt_s, jerk_samples=len(jerk),
                **reference_reversal_metrics(record),
                joint_length_rad=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                joint_displacement_rad=amplitude,
                jerk_rms_rad_s3=rms,
                dimensionless_jerk=None if rms is None or amplitude <= 1e-12
                else duration**3*rms/amplitude), path, endpoints


def measure_reference(robot, curve, goal_xy_m, dt_s=.001):
    from planar_dynamics.trajectory import trajectory_record
    from planar_physics.kinematics import tcp
    quality, path, endpoints = reference_quality(trajectory_record(curve), dt_s)
    xy = tcp(robot, path)
    quality.update(tcp_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                   endpoint_tcp_error_m=float(np.linalg.norm(tcp(robot, endpoints[-1])-goal_xy_m)))
    if any(v is not None and not np.isfinite(v).all() for v in quality.values()):
        raise ValueError('non-finite reference measurement')
    return quality
