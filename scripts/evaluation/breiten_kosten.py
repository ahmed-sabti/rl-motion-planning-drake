from pathlib import Path
import argparse, json, os, resource, subprocess, sys, time
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE/'src'))
QUERIES = 3000


def rss():
    return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())]))*1024


def measure(stem):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from planar_eval.sac_support import load_sac
    from planar_learn.step_policy import policy_action
    before = rss()
    started = time.perf_counter()
    model, config, payload = load_sac(stem)
    load_s = time.perf_counter()-started
    after = rss()
    obs = np.random.default_rng(0).uniform(-1., 1., (QUERIES, *model.observation_space.shape)).astype(np.float32)
    act = policy_action(model, deterministic=False)
    for o in obs[:200]:
        act(o)
    started = time.perf_counter()
    for o in obs:
        act(o)
    predict_s = (time.perf_counter()-started)/QUERIES
    with torch.no_grad():
        x = torch.as_tensor(obs[:1])
        started = time.perf_counter()
        for _ in range(QUERIES):
            model.actor(x, deterministic=False)
        network_s = (time.perf_counter()-started)/QUERIES
    return dict(width=payload['training']['hidden_width'], checkpoint=Path(stem).name,
                actor_parameters=sum(p.numel() for p in model.actor.parameters()),
                all_parameters=sum(p.numel() for p in model.policy.parameters()),
                checkpoint_bytes=Path(stem).with_suffix('.zip').stat().st_size,
                load_s=load_s, rss_before_bytes=before, rss_added_bytes=after-before,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                predict_s=predict_s, network_s=network_s)


def main():
    p = argparse.ArgumentParser(description='Cost of serving each network width: parameters, memory and time per policy query.')
    p.add_argument('--candidate', action='append', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--one', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.one:
        print(json.dumps(measure(a.one)))
        return
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS='1')
    for stem in a.candidate:
        for repeat in range(a.repeats):
            out = subprocess.check_output([sys.executable, __file__, '--candidate', stem, '--output', str(a.output),
                                           '--one', stem], env=env, stderr=subprocess.DEVNULL)
            rows.append(dict(json.loads(out.decode().strip().splitlines()[-1]), repeat=repeat))
    import platform, torch
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(queries=QUERIES, threads=1, python=platform.python_version(),
                                        torch=torch.__version__, machine=platform.machine(), rows=rows), indent=2))


if __name__ == '__main__':
    main()
