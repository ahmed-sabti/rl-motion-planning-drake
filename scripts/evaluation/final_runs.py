from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'src'))
from planar_eval.splits import FINAL


def main(argv=None):
    p = argparse.ArgumentParser(description='The final comparison: the chosen policy against RRT, RRT* and '
                                            'RRT-Connect on the same requests, several runs in fresh processes.')
    p.add_argument('--candidate', required=True, help='checkpoint stem, e.g. policy/<dir>/<stem>')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=FINAL[0])
    p.add_argument('--count', type=int, default=FINAL[1])
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--requests', type=Path)
    a = p.parse_args(argv)
    if a.count < 1 or a.runs < 1:
        p.error('--count and --runs must be positive')
    from confirm import candidate_hashes, PIPELINE, SCENE_PROFILE
    identity = dict(candidate_hashes(a.candidate), scene_profile=SCENE_PROFILE, pipeline=PIPELINE,
                    requests_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest() if a.requests else None)
    out = a.output.resolve()
    for i in range(1, a.runs+1):
        run = out/f'run{i}'
        if not run.exists():
            continue
        freeze = json.loads((run/'freeze.json').read_text()) if (run/'freeze.json').exists() else {}
        if ((freeze.get('paths', {}).get('candidate'), freeze.get('seed'), freeze.get('count'))
                != (a.candidate, a.seed, a.count) or any(freeze.get(k) != v for k, v in identity.items())):
            raise SystemExit(f'{run} has different or unverified inputs; preserve it and choose a new output directory')
        if not (run/'done').is_file():
            raise SystemExit(f'{run} is incomplete; preserve it and choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO/'src'),
               OMP_NUM_THREADS='1')
    from summarise_final import aggregate, aggregate_lines, dumps, headline, summarise
    logs = out/'logs'
    logs.mkdir(exist_ok=True)
    awake = ['caffeinate', '-i'] if sys.platform == 'darwin' else []
    summaries = []
    for i in range(1, a.runs+1):
        run = out/f'run{i}'
        if not run.exists():
            with open(logs/f'run{i}.log', 'w') as log:
                extra = ['--requests', str(a.requests.resolve())] if a.requests else []
                code = subprocess.call([*awake, sys.executable, str(HERE/'confirm.py'), '--candidate', a.candidate,
                                        '--seed', str(a.seed), '--count', str(a.count), '--output', str(run), *extra],
                                       cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                raise SystemExit(f'run {i} failed; see {logs}/run{i}.log, then move {run} out of {out} and rerun')
        summary = summarise(run)
        (run/'summary-final.json').write_text(dumps(summary))
        print(headline(summary), flush=True)
        summaries.append(summary)
    result = aggregate(summaries)
    (out/'summary-runs.json').write_text(json.dumps(result, indent=2))
    print(aggregate_lines(result))


if __name__ == '__main__':
    main()
