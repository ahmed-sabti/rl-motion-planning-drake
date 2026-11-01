from pathlib import Path
import argparse
import json
import math

import numpy as np

DRAWS = 5000
SEED = 0
CANDIDATE = 'candidate'

CATEGORIES = ('held', 'not_held', 'rejected', 'rejected_kto', 'plan_failed', 'deadline',
              'error', 'other')

SECONDARY = {
    'mission_time_s': lambda arm: arm.get('total_s'),
    'planning_s': lambda arm: arm.get('planning_s'),
    'motion_hold_s': lambda arm: arm.get('motion_hold_s'),
    'tool_path_m': lambda arm: (arm.get('executed_metrics') or {}).get('distance_m'),
    'torque_utilisation': lambda arm: arm.get('peak_torque_ratio'),
    'jerk_rms_rad_s3': lambda arm: (arm.get('executed_metrics') or {}).get('jerk_rms_rad_s3'),
}

REFERENCE = {
    'pipeline_s': lambda a: sum(a[k] for k in ('planning_s', 'prepare_s', 'reference_duration_s')),
    'preparation_s': lambda a: a.get('prepare_s'),
    'duration_s': lambda a: a.get('reference_duration_s'),
    'tcp_length_m': lambda a: (a.get('reference_metrics') or {}).get('tcp_length_m'),
    'jerk_rms_rad_s3': lambda a: (a.get('reference_metrics') or {}).get('jerk_rms_rad_s3'),
    'dimensionless_jerk': lambda a: (a.get('reference_metrics') or {}).get('dimensionless_jerk'),
    'joint_reversal_count': lambda a: (a.get('reference_metrics') or {}).get('joint_reversal_count'),
    'endpoint_tcp_error_mm': lambda a: 1000*(a.get('reference_metrics') or {})['endpoint_tcp_error_m']
        if a.get('reference_metrics') else None,
}


def trajectory_group(rows, arms, planners):
    accepted = lambda r, a: r['arms'][a]['accepted'] is True
    counts = {a: dict(accepted=sum(accepted(r, a) for r in rows),
                     within_deadline=sum(r['arms'][a].get('reference_within_deadline') is True for r in rows),
                     deadline_not_recorded=sum(accepted(r, a) and
                         r['arms'][a].get('reference_within_deadline') is None for r in rows),
                     wilson_95ci=wilson(sum(accepted(r, a) for r in rows), len(rows))) for a in arms}
    comparisons = {}
    for planner in planners:
        differences = np.array([int(accepted(r, CANDIDATE))-int(accepted(r, planner)) for r in rows])
        joint = [r for r in rows if accepted(r, CANDIDATE) and accepted(r, planner)]
        metrics = {}
        for name, read in REFERENCE.items():
            pairs = [(read(r['arms'][CANDIDATE]), read(r['arms'][planner])) for r in joint]
            recorded = [p for p in pairs if all(v is not None and np.isfinite(v) for v in p)]
            metrics[name] = dict(pairs=len(recorded), missing=len(joint)-len(recorded))
            if recorded and len(recorded) == len(joint):
                metrics[name].update(mean_candidate=float(np.mean([p[0] for p in recorded])),
                                     mean_planner=float(np.mean([p[1] for p in recorded])),
                                     relative_difference=ratio_of_means(recorded),
                                     relative_difference_95ci=cluster_bootstrap(recorded, relative_difference))
        comparisons[planner] = dict(jointly_accepted=len(joint), metrics=metrics,
                                    accepted_difference=float(differences.mean()) if len(rows) else None,
                                    accepted_difference_95ci=cluster_bootstrap(differences, np.mean),
                                    candidate_only=int((differences == 1).sum()),
                                    planner_only=int((differences == -1).sum()),
                                    exact_mcnemar_p=sign_flip_p(differences) if len(rows) else None)
    return dict(arms=counts, comparisons=comparisons)


def category(arm):
    outcome, reason = str(arm.get('outcome')), str(arm.get('reason') or '')
    optimization = str(arm.get('optimization') or '')
    if outcome.startswith('error:'):
        return 'error'
    if arm.get('held'):
        return 'held'
    if reason.startswith('rejected:kto') or optimization.startswith('rejected:kto'):
        return 'rejected_kto'
    if (reason.startswith('prepare:') or reason.startswith('reference:')) and outcome != 'deadline_exceeded':
        return 'rejected'
    return {'not_held': 'not_held', 'trajectory_rejected': 'rejected', 'plan_failed': 'plan_failed',
            'deadline_exceeded': 'deadline'}.get(outcome, 'other')


def held(arm):
    if not isinstance(arm.get('held'), bool):
        raise ValueError('held must be an explicit boolean')
    if arm['held'] and (str(arm.get('outcome')).startswith('error:') or arm.get('accepted') is not True):
        raise ValueError('a held row must be an accepted trajectory without error')
    return arm['held']


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [centre-half, centre+half]


def cluster_bootstrap(values, statistic, draws=DRAWS, seed=SEED):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    index = rng.integers(len(values), size=(draws, len(values)))
    replicate = np.array([v for v in (statistic(values[i]) for i in index) if v is not None], dtype=float)
    replicate = replicate[np.isfinite(replicate)]
    if not len(replicate):
        return None
    return [float(v) for v in np.quantile(replicate, [.025, .975])]


def sign_flip_p(differences):
    d = np.asarray(differences, dtype=float)
    if not np.all(np.isin(d, (-1., 0., 1.))):
        raise ValueError('per-request held differences must be -1, 0 or +1')
    n = int(np.count_nonzero(d))
    if n == 0:
        return 1.
    observed = abs(int(d.sum()))
    total = sum(math.comb(n, b) for b in range(n+1) if abs(2*b-n) >= observed)
    return min(1., total/2**n)


def relative_difference(sample):
    reference = sample[:, 1].mean()
    if reference == 0:
        return None
    return sample[:, 0].mean()/reference-1.


def ratio_of_means(pairs):
    value = relative_difference(np.asarray(pairs, dtype=float))
    return None if value is None else float(value)


def compare(rows, planner):
    differences = np.array([int(held(r['arms'][CANDIDATE]))-int(held(r['arms'][planner])) for r in rows])
    n = len(rows)
    primary = dict(requests=n,
                   held_candidate=int(sum(held(r['arms'][CANDIDATE]) for r in rows)),
                   held_planner=int(sum(held(r['arms'][planner]) for r in rows)),
                   candidate_only=int((differences == 1).sum()),
                   planner_only=int((differences == -1).sum()),
                   discordant_indices=[r['index'] for r, d in zip(rows, differences) if d],
                   held_difference=float(differences.mean()) if n else None,
                   held_difference_95ci=cluster_bootstrap(differences, np.mean),
                   sign_flip_p_two_sided=sign_flip_p(differences) if n else None)
    joint = [r for r in rows if held(r['arms'][CANDIDATE]) and held(r['arms'][planner])]
    secondary = dict(jointly_held=len(joint), conditional_on='requests both arms hold')
    for name, read in SECONDARY.items():
        pairs = [(read(r['arms'][CANDIDATE]), read(r['arms'][planner])) for r in joint]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        if not pairs or len(pairs) < len(joint):
            secondary[name] = dict(pairs=len(pairs), missing=len(joint)-len(pairs), relative_difference=None,
                                   relative_difference_on_recorded=ratio_of_means(pairs) if pairs else None,
                                   note='not recorded for every jointly held request')
            continue
        secondary[name] = dict(pairs=len(pairs), mean_candidate=float(np.mean([p[0] for p in pairs])),
                               mean_planner=float(np.mean([p[1] for p in pairs])),
                               relative_difference=ratio_of_means(pairs),
                               relative_difference_95ci=cluster_bootstrap(
                                   np.asarray(pairs, dtype=float), relative_difference))
    return dict(primary=primary, secondary=secondary)


def arm_summary(rows, arms):
    result = {}
    for name in arms:
        counts = {c: 0 for c in CATEGORIES}
        reasons = {}
        for r in rows:
            arm = r['arms'][name]
            c = category(arm)
            counts[c] += 1
            if c != 'held':
                key = str(arm.get('reason') or arm.get('outcome'))
                if c == 'error':
                    key = str(arm.get('outcome'))
                reasons[key] = reasons.get(key, 0)+1
        k = int(sum(held(r['arms'][name]) for r in rows))
        result[name] = dict(held=k, of=len(rows), share=k/len(rows) if rows else None,
                            wilson_95ci=wilson(k, len(rows)), outcomes=counts,
                            relaxed_timing=sum(str(r['arms'][name].get('optimization')) == 'optimized_relaxed:kto'
                                               and held(r['arms'][name]) for r in rows),
                            reasons=dict(sorted(reasons.items())))
    return result


def rung1_counterfactual(rows, arms):
    if CANDIDATE not in arms:
        return None
    cells = [r['arms'][CANDIDATE] for r in rows]
    return dict(held_rungs_used_1=int(sum(held(a) and a.get('rungs_used') == 1 for a in cells)),
                held=int(sum(held(a) for a in cells)), of=len(cells),
                held_without_rungs_used=int(sum(held(a) and a.get('rungs_used') is None for a in cells)),
                note='descriptive; beside, never instead of, the primary')


def group(rows, arms, planners):
    return dict(requests=len(rows), arms=arm_summary(rows, arms),
                trajectory=trajectory_group(rows, arms, planners),
                rung1_counterfactual=rung1_counterfactual(rows, arms),
                comparisons={p: compare(rows, p) for p in planners})


def load(folder):
    folder = Path(folder)
    freeze = json.loads((folder/'freeze.json').read_text())
    requests = json.loads((folder/'requests.json').read_text())
    rows = json.loads((folder/'results.json').read_text())
    return freeze, requests, rows


def summarise(folder):
    freeze, requests, rows = load(folder)
    arms = list(freeze['arms'])
    if CANDIDATE not in arms:
        raise ValueError('the freeze names no candidate arm')
    planners = [a for a in arms if a != CANDIDATE]
    if len(rows) != freeze['count'] or len(requests) != freeze['count']:
        raise ValueError('final evaluation is incomplete')
    if len({r['seed'] for r in requests}) != len(requests):
        raise ValueError('duplicate request seeds')
    issued, refusals = [], []
    for i, (request, row) in enumerate(zip(requests, rows)):
        if row.get('index') != i or request['seed'] != row['seed']:
            raise ValueError('pair identity/order mismatch')
        if not request['issued']:
            if row.get('issued') is not False or row.get('arms'):
                raise ValueError('generation refusal relabelled')
            refusals.append(dict(index=i, seed=row['seed'], reason=row.get('reason')))
            continue
        if set(row.get('arms', {})) != set(arms):
            raise ValueError(f'request {i}: arm results differ from the frozen arms')
        for name in arms:
            arm = row['arms'][name]
            held(arm)
            if 'accepted' not in arm or 'total_s' not in arm:
                raise ValueError(f'request {i} {name}: accepted/total_s missing')
            if not isinstance(arm['accepted'], bool):
                raise ValueError(f'request {i} {name}: accepted must be an explicit boolean')
            reversal_protocol = freeze.get('reference_measurement', {}).get('reversals')
            if arm['accepted'] and reversal_protocol:
                metrics = arm.get('reference_metrics') or {}
                counts = metrics.get('joint_reversals')
                if (not isinstance(counts, list) or len(counts) != len(request['start'])
                        or any(type(n) is not int or n < 0 for n in counts)
                        or metrics.get('joint_reversal_count') != sum(counts)
                        or metrics.get('reversal_sample_dt_max_s') != reversal_protocol['sample_dt_max_s']
                        or metrics.get('reversal_deadband_rad_s') != reversal_protocol['deadband_rad_s']
                        or not arm.get('reference_record')):
                    raise ValueError(f'request {i} {name}: reversal measurement missing or inconsistent')
            total = arm['total_s']
            if total is None or not np.isfinite(total) or total < 0 or (arm['held'] and total == 0):
                raise ValueError(f'request {i} {name}: invalid total_s')
        issued.append(row)
    groups = {'wall': [r for r in issued if r.get('kind') == 'wall'],
              'non_chord': [r for r in issued if r.get('direct_chord') is False]}
    unknown = [r['index'] for r in issued if r.get('direct_chord') is None]
    return dict(folder=str(folder), pipeline=freeze.get('pipeline'), seed=freeze.get('seed'),
                count=freeze.get('count'), deadline_s=freeze.get('deadline_s'),
                direct_chord_unknown=dict(requests=len(unknown), indices=unknown),
                rules=('held = held flag on an accepted trajectory; rejected:kto:* = Verworfen '
                       '(rejected, not held); error:* = not held; unit = request; bootstrap '
                       f'{DRAWS} draws seed {SEED}; exact sign-flip test'),
                issued=len(issued), generation_refusals=refusals,
                overall=group(issued, arms, planners),
                subgroups={k: group(v, arms, planners) for k, v in groups.items()})


def dumps(summary):
    return json.dumps(summary, indent=2, allow_nan=False)


def na(value):
    return 'n/a' if value is None else value


def headline(result):
    lines = [f"{result['pipeline']}  seed {result['seed']}  issued {result['issued']}/{result['count']}"
             f"  deadline {result['deadline_s']} s"]
    for name, arm in result['overall']['arms'].items():
        outcomes = ' '.join(f'{k}={v}' for k, v in arm['outcomes'].items() if v)
        lines.append(f"  {name:12s} held {arm['held']:4d}/{arm['of']}  {outcomes}")
    for planner, c in result['overall']['comparisons'].items():
        p = c['primary']
        ci = p['held_difference_95ci']
        if ci is None:
            lines.append(f"  SAC - {planner:12s} no issued requests")
            continue
        lines.append(f"  SAC - {planner:12s} {p['held_difference']:+.3f}  95% "
                     f"[{ci[0]:+.3f}, {ci[1]:+.3f}]  discordant +{p['candidate_only']}/-{p['planner_only']}"
                     f"  sign-flip p={p['sign_flip_p_two_sided']:.4f}"
                     f"  joint {c['secondary']['jointly_held']}  time "
                     f"{na(c['secondary']['mission_time_s'].get('relative_difference'))}")
    rung1 = result['overall'].get('rung1_counterfactual')
    if rung1:
        lines.append(f"  SAC held with rungs_used==1 (rung-1 counterfactual, descriptive): "
                     f"{rung1['held_rungs_used_1']}/{rung1['of']} (held {rung1['held']})")
    if result.get('direct_chord_unknown', {}).get('requests'):
        lines.append(f"  chord diagnostic failed on {result['direct_chord_unknown']['requests']} "
                     f"requests (counted separately, not non-chord)")
    return '\n'.join(lines)


def spread(values):
    values = [v for v in values if v is not None]
    return None if not values else dict(mean=float(np.mean(values)), min=float(min(values)), max=float(max(values)))


def aggregate(summaries):
    first = summaries[0]['overall']
    arms = {name: dict(held_per_run=[s['overall']['arms'][name]['held'] for s in summaries],
                       **(spread([s['overall']['arms'][name]['held'] for s in summaries]) or {}))
            for name in first['arms']}
    comparisons = {}
    for planner in first['comparisons']:
        diffs = [s['overall']['comparisons'][planner]['primary']['held_difference'] for s in summaries]
        times = [s['overall']['comparisons'][planner]['secondary']['mission_time_s'].get('relative_difference')
                 for s in summaries]
        comparisons[planner] = dict(held_difference_per_run=diffs, held_difference=spread(diffs),
                                    mission_time_relative_difference_per_run=times,
                                    mission_time_relative_difference=spread(times))
    return dict(runs=len(summaries), seed=summaries[0]['seed'], issued=summaries[0]['issued'],
                arms=arms, comparisons=comparisons)


def aggregate_lines(result):
    lines = [f"{result['runs']} runs  seed {result['seed']}  issued {result['issued']}"]
    for name, arm in result['arms'].items():
        lines.append(f"  {name:12s} held per run {arm['held_per_run']}  mean {arm['mean']:.1f}  "
                     f"range {arm['min']:.0f}-{arm['max']:.0f}")
    for planner, c in result['comparisons'].items():
        d = c['held_difference']
        lines.append(f"  SAC - {planner:12s} held difference mean {d['mean']:+.3f}  range "
                     f"[{d['min']:+.3f}, {d['max']:+.3f}]")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of one run of the final evaluation.')
    parser.add_argument('folder')
    parser.add_argument('--output', help='default: <folder>/summary-final.json')
    arguments = parser.parse_args()
    summary = summarise(arguments.folder)
    Path(arguments.output or Path(arguments.folder)/'summary-final.json').write_text(dumps(summary))
    print(headline(summary))
