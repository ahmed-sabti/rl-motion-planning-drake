from functools import partial
from pathlib import Path
import argparse, hashlib, json, sys, time
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path[:0] = [str(CODE/'src'), str(HERE)]
import torch
from planar_eval import methods
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from planar_learn.step_env import StepConfig
from planar_learn.scenes import SCENE_PROFILE
from planar_learn.step_policy import save
from planar_dynamics.mission import KTO_PROFILE
from planar_physics.contract import VARIANT
from kto_env import ComparisonEnv
from planar_eval.sac_support import R, SETTINGS, WIDTHS, policy_kwargs, architecture
from planar_eval.step_bank import load_tasks


class TimedSAC(SAC):
    def _sample_action(self, *args, **kwargs):
        started = time.perf_counter()
        result = super()._sample_action(*args, **kwargs)
        self.env.env_method('charge_inference', time.perf_counter()-started)
        return result


def main():
    methods.pin_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True); p.add_argument('--name', required=True)
    p.add_argument('--steps', required=True, type=int); p.add_argument('--checkpoint-every', required=True, type=int)
    p.add_argument('--envs', type=int, default=10); p.add_argument('--torch-threads', type=int, default=8)
    p.add_argument('--width', type=int, choices=WIDTHS, default=1024); p.add_argument('--seed', type=int, default=963)
    p.add_argument('--out', default=str(CODE/'runs')); p.add_argument('--min-tasks', type=int, default=2000)
    p.add_argument('--preview-every', type=int, default=0); p.add_argument('--preview-dir')
    p.add_argument('--lineage', default='kto-ohne-kreis-20260930')
    a = p.parse_args()
    if a.envs % 5 or any(v % a.envs for v in (SETTINGS['learning_starts'], a.steps, a.checkpoint_every)):
        raise ValueError('envs a multiple of five, and steps, checkpoints and learning_starts of envs')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if list(out.glob(a.name+'*')):
        raise ValueError('Preserve existing experiment outputs')
    tasks, body, spec = load_tasks(Path(a.bank), R)
    assert body['split'] == 'train' and body['timing_profile'] == KTO_PROFILE and len(tasks) >= a.min_tasks
    config = StepConfig(scene_observed=False, tcp_observed=False, line_search=9, timing_profile=KTO_PROFILE,
                        joint_clip_cost=.05)
    preview = (a.preview_dir or str(out/(a.name+'-episodes')), a.preview_every) if a.preview_every > 0 else None
    env = SubprocVecEnv([partial(ComparisonEnv, tasks, config, a.seed+i, preview and (*preview, i))
                         for i in range(a.envs)])
    settings = dict(SETTINGS, gradient_steps=a.envs//5)
    try:
        torch.set_num_threads(a.torch_threads)
        model = TimedSAC('MlpPolicy', env, seed=a.seed, device='cpu', **settings, policy_kwargs=policy_kwargs(a.width))
        checked = architecture(model)
        started = time.perf_counter()
        metadata = dict(lineage=a.lineage, scene_profile=SCENE_PROFILE,
                        bank_sha256=hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
                        scene_distribution_parameters=body['scene_distribution'],
                        task_admission='unfiltered-cartesian' if VARIANT == '2r' else 'original-cartesian',
                        robot_variant=VARIANT, learning_algorithm='SAC', initialisation='random',
                        parent_checkpoint=None, validation_only=False, seed=a.seed,
                        settings=settings,
                        actual_network_inputs=checked['actor_inputs'], actual_model_envs=a.envs, architecture=checked,
                        no_aim=False, hidden_width=a.width, geometry_feature='wall_distance', joint_clip_cost=.05,
                        timing_profile=KTO_PROFILE, torch_threads=a.torch_threads, vec_env='SubprocVecEnv',
                        bank_file=Path(a.bank).name,
                        training_time_accounting='serving-only-v2',
                        initialization_entropy_coefficient=.1, scene_distribution=f'uniform{len(tasks)}-kto',
                        worker_seeds=list(range(a.seed, a.seed+a.envs)), classical_references=['rrt', 'rrt_connect'])
        (out/(a.name+'.launch.json')).write_text(json.dumps(metadata, indent=2))

        def checkpoint(suffix):
            path = out/(a.name+suffix)
            save(model, path, R, config, spec, training=dict(**metadata,
                 timesteps=model.num_timesteps, total_lineage_steps=model.num_timesteps,
                 optimizer_updates=model._n_updates, checkpoint_after_update=True,
                 replay_transitions=model.replay_buffer.size()*model.replay_buffer.n_envs,
                 environment_transitions=env.get_attr('transitions'), elapsed_s=time.perf_counter()-started),
                 parameters=sum(q.numel() for q in model.policy.parameters()))

        class Record(BaseCallback):
            def __init__(self):
                super().__init__(); self.next_checkpoint = a.checkpoint_every; self.next_log = 16384; self.outcomes = []
            def _on_rollout_start(self):
                if self.num_timesteps >= self.next_checkpoint:
                    checkpoint('-at'+str(self.num_timesteps)); self.next_checkpoint += a.checkpoint_every
            def _on_step(self):
                self.outcomes.extend(info['code'] for info, done in zip(self.locals['infos'], self.locals['dones']) if done)
                if self.num_timesteps >= self.next_log:
                    self.next_log += 16384
                    print(json.dumps(dict(stage='sac', steps=self.num_timesteps, seconds=time.perf_counter()-started,
                                          updates=model._n_updates,
                                          entropy_coefficient=float(model.log_ent_coef.detach().exp().item()),
                                          episodes={k: self.outcomes.count(k) for k in set(self.outcomes)})), flush=True)
                    self.outcomes.clear()
                return True
        print(json.dumps(dict(stage='start', name=a.name, envs=a.envs,
                              gradient_steps=settings['gradient_steps'], torch_threads=a.torch_threads)), flush=True)
        model.learn(total_timesteps=a.steps, callback=Record(), log_interval=None)
        assert model.num_timesteps == a.steps
        checkpoint('')
        print(json.dumps(dict(stage='complete', steps=model.num_timesteps, updates=model._n_updates,
                              seconds=time.perf_counter()-started)), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
