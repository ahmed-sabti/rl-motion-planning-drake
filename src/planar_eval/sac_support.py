from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy

from planar_eval import methods
from planar_physics.fixtures import ROBOT_EQUAL_030 as R
from planar_physics.contract import VARIANT
from planar_learn.step_env import RETIRED_OPTIONS, StepConfig, observation_fields
from planar_learn.step_policy import StepPlanner, load_manifest

methods.pin_threads(1)

SETTINGS = dict(learning_rate=.0003, buffer_size=500000, learning_starts=5120,
    batch_size=256, tau=.005, gamma=.99, train_freq=1, gradient_steps=1,
    ent_coef='auto_0.1', target_entropy='auto', target_update_interval=1)
WIDTHS = (64, 256, 1024)


def policy_kwargs(width=256):
    if width not in WIDTHS:
        raise ValueError('SAC hidden width must be 64, 256 or 1024')
    return dict(net_arch=dict(pi=[width, width], qf=[width, width]), activation_fn=torch.nn.Tanh)


def architecture(model):
    if not isinstance(model.policy, SACPolicy):
        raise ValueError('Expected an actual SAC policy')
    layers = list(model.actor.latent_pi)
    inputs, width = layers[0].in_features, layers[0].out_features
    n = R.n_joints
    linear = lambda net: [(l.in_features, l.out_features) for l in net if isinstance(l, torch.nn.Linear)]
    action, _ = model.predict(np.zeros(inputs, dtype=np.float32), deterministic=True)
    if not (len(layers) == 4 and all(isinstance(layers[i], torch.nn.Tanh) for i in (1, 3))
            and inputs == n+7 and width in WIDTHS
            and linear(model.actor.latent_pi) == [(inputs, width), (width, width)]
            and model.actor.mu.out_features == model.actor.log_std.out_features == n
            and model.observation_space.shape == (inputs,) and model.action_space.shape == (n,)
            and all(linear(net) == [(inputs+n, width), (width, width), (width, 1)] for net in
                    (model.critic.qf0, model.critic.qf1, model.critic_target.qf0, model.critic_target.qf1))
            and all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
            and action.shape == (n,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.):
        raise ValueError('SAC checkpoint outside the served architecture')
    return dict(actor_inputs=inputs, actor_hidden=[width, width], actions=n,
                training_critic_inputs=inputs+n, critic_action_components=n)


def load_sac(path):
    payload = load_manifest(Path(path), R)
    training = payload.get('training', {})
    if training.get('learning_algorithm') != 'SAC':
        raise ValueError('only explicitly identified SAC checkpoints are served')
    if training.get('robot_variant') != VARIANT:
        raise ValueError(f'Checkpoint must be for the {VARIANT} arm')
    model = SAC.load(Path(path).with_suffix('.zip'), device='cpu')
    checked = architecture(model)
    stored = payload['config']
    for name, default in RETIRED_OPTIONS.items():
        if stored.get(name, default) != default:
            raise ValueError(f'checkpoint uses the retired option {name}={stored[name]!r}')
    known = {f.name for f in fields(StepConfig)}
    config = StepConfig(**{k: v for k, v in stored.items() if k in known})
    declared = payload['training'].get('hidden_width')
    if ((declared is not None and declared != checked['actor_hidden'][0])
            or checked['actor_inputs'] != len(observation_fields(config))):
        raise ValueError('Actual network disagrees with checkpoint metadata')
    return model, config, payload


def load_candidate(path):
    model, config, payload = load_sac(path)
    return StepPlanner(model, config, payload=payload)
