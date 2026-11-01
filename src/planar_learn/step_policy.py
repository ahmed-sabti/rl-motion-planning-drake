from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from planar_physics.budget import BudgetExhausted, Ledger
from planar_physics.certificate import CertifierConfig, edge_verdicts
from planar_physics.contract import DEFAULT_CONTRACT
from planar_physics.kinematics import tcp
from planar_physics.plan import SolveStatus
from planar_physics.planners._common import finish, goals_for, validate_start

from .step_env import (ALGORITHM, VERSION, sense,
                       step_factors, action_step, advance)

FORMAT = 'step-sb3-v1'
LABEL = 'Step policy'


def save(model, path, robot, config, spec, **extra):
    archive = Path(path).with_suffix('.zip')
    model.save(archive)
    payload = {'format': FORMAT, 'config': asdict(config), 'scene_distribution': asdict(spec),
               'physical_contract': asdict(DEFAULT_CONTRACT), **extra}
    archive.with_suffix('.json').write_text(json.dumps(payload, indent=1)+'\n')
    return payload


def load_manifest(path, robot):
    path = Path(path)
    payload = json.loads(path.with_suffix('.json').read_text())
    if payload.get('format') != FORMAT:
        raise ValueError(f'{LABEL} checkpoint format mismatch; retrain historical policies')
    return payload


def policy_action(model, deterministic=True):
    import torch
    from stable_baselines3.sac.policies import SACPolicy
    policy = getattr(model, 'policy', None)
    if not isinstance(policy, SACPolicy):
        return lambda observation: model.predict(observation, deterministic=deterministic)[0]
    policy.set_training_mode(False)

    def act(observation):
        with torch.no_grad():
            action = policy.actor(torch.as_tensor(observation[None]), deterministic=deterministic)
        return policy.unscale_action(action.numpy())[0]
    return act


def served_chord(robot, scene, start, goals, *, margin_m=DEFAULT_CONTRACT.margin_m, ledger=None, strict=False):
    if goals is None or not len(goals):
        return None
    start = np.asarray(start, dtype=float)
    free = edge_verdicts(robot, scene, cfg=CertifierConfig(margin_m=margin_m), ledger=ledger, strict=strict)
    for goal_q in np.asarray(goals, dtype=float).reshape(-1, robot.n_joints):
        if free(start, goal_q):
            return np.vstack([start, goal_q])
    return None


def walk(model, robot, scene, start, goal_xy, config, *, margin_m=DEFAULT_CONTRACT.margin_m,
         ledger=None, deterministic=True, goals=None, try_chord=True, deadline=None):
    q = np.asarray(start, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)
    path, blocked, stuck, committed = [q.copy()], 0, 0, None
    if try_chord:
        line = served_chord(robot, scene, q, goals, margin_m=margin_m, ledger=ledger)
        if line is not None:
            return line, 'chord', 0, blocked
    factors = step_factors(config)
    free = edge_verdicts(robot, scene, cfg=CertifierConfig(margin_m=margin_m), ledger=ledger)
    act = policy_action(model, deterministic)
    tip = tcp(robot, q)
    for taken in range(config.horizon):
        if float(np.linalg.norm(tip-goal_xy)) <= config.goal_tol_m:
            return np.asarray(path), 'reached', taken, blocked
        observation, _, committed = sense(robot, scene, q, goal_xy, config.horizon-taken,
                                          config, committed, stuck, tip=tip)
        if deadline is not None and perf_counter() > deadline:
            raise BudgetExhausted('wall_clock_s')
        candidate = advance(robot, q, action_step(act(observation), config), factors, free)
        if candidate is None:
            blocked += 1
            stuck += 1
            continue
        q, stuck = candidate, 0
        tip = tcp(robot, q)
        path.append(q.copy())
    reached = float(np.linalg.norm(tip-goal_xy)) <= config.goal_tol_m
    return np.asarray(path), 'reached' if reached else 'horizon', config.horizon, blocked


class StepPlanner:
    name, version = ALGORITHM, VERSION
    RETRIES = 256

    def __init__(self, model, config, *, payload=None):
        self.model, self.config = model, config
        self.payload = dict(payload or {})
        self.physical_contract = DEFAULT_CONTRACT

    def solve(self, request, goals=None):
        import torch
        torch.manual_seed(int(request.seed.entropy) % (2**63 - 1))
        ledger = Ledger(request.budget)
        cap = request.budget.wall_clock_s
        deadline = None if cap is None else perf_counter() + float(cap)
        physical, runs, stats = self.physical_contract, 0, {}

        def done(status, reason='', waypoints=None):
            result = finish(request, ledger, self.name, status=status, reason=reason,
                            waypoints=waypoints)
            result.search_stats.update(stats, rungs_used=runs, rungs_available=self.RETRIES)
            return result
        if ((request.certifier.margin_m, request.certifier.numerical_slack_m, request.ik.margin_m, request.ik.tol_m)
                != (physical.margin_m, physical.numerical_slack_m, physical.margin_m, physical.goal_tol_m)):
            return done(SolveStatus.INVALID_INPUT, 'invalid:policy_contract')
        if self.config.goal_tol_m > request.ik.tol_m:
            return done(SolveStatus.INVALID_INPUT, 'invalid:policy_goal_tolerance')
        try:
            reason = validate_start(request, ledger)
            if not reason:
                goals, reason = goals_for(request, ledger, goals)
            if reason:
                return done(SolveStatus.INVALID_INPUT if reason.startswith('invalid:')
                            else SolveStatus.UNKNOWN, reason)
            for runs in range(1, self.RETRIES+1):
                path, outcome, steps, blocked = walk(
                    self.model, request.robot, request.scene, request.start_q_rad,
                    np.asarray(request.goal, dtype=float), self.config,
                    margin_m=request.certifier.margin_m, ledger=ledger, goals=goals,
                    deterministic=False, try_chord=runs == 1, deadline=deadline)
                if outcome in ('reached', 'chord'):
                    break
            stats.update(steps=steps, blocked=blocked, horizon=self.config.horizon)
            if outcome not in ('reached', 'chord') or len(path) < 2:
                return done(SolveStatus.UNKNOWN, f'walk:{outcome}')
            if np.linalg.norm(tcp(request.robot, path[-1])-np.asarray(request.goal)) > request.ik.tol_m:
                return done(SolveStatus.UNKNOWN, 'walk:outside_tolerance')
        except BudgetExhausted as exhausted:
            return done(SolveStatus.UNKNOWN, f'budget:{exhausted.field}')
        return done(SolveStatus.SOLVED, waypoints=path)
