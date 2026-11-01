from dataclasses import replace
from time import perf_counter
import os

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from ..budget import BudgetExhausted, Ledger
from ..plan import SolveStatus
from ._common import finish, goals_for, validate_start

KINDS = ("rrt", "rrt_connect", "rrt_star")
VERIFY_RESERVE_S = .2

ou.setLogLevel(getattr(ou.LogLevel, 'LOG_' + os.environ.get('PLANAR_OMPL_LOG', 'NONE').strip().upper(),
                       ou.LogLevel.LOG_NONE))


def _space(robot, cfg, validity_factory):
    n = int(robot.n_joints)
    space = ob.RealVectorStateSpace(n)
    bounds = ob.RealVectorBounds(n)
    for i in range(n):
        bounds.setLow(i, float(robot.lo_rad[i]))
        bounds.setHigh(i, float(robot.hi_rad[i]))
    space.setBounds(bounds)
    si = ob.SpaceInformation(space)
    validity = validity_factory(si)
    si.setStateValidityChecker(validity)
    extent = float(space.getMaximumExtent())
    si.setStateValidityCheckingResolution(float(np.clip(cfg.edge_step_rad / max(extent, 1e-9), 1e-6, .5)))
    si.setup()
    return si, validity


def _seed(entropy):
    seed = int(entropy) % (2 ** 32 - 1) or 1
    ou.RNG.setSeed(seed)
    return seed


class _DrakeValidity(ob.StateValidityChecker):
    def __init__(self, si, robot, scene, padding_m, ledger):
        super().__init__(si)
        from ..drake_model import get_checker
        self._checker = get_checker(robot, scene, padding=float(padding_m))
        self._free = self._checker.CheckConfigCollisionFree
        self._lo = tuple(float(v) for v in robot.lo_rad)
        self._hi = tuple(float(v) for v in robot.hi_rad)
        self._n = len(self._lo)
        self._q = np.zeros(self._n, dtype=float)
        self._ledger = ledger
        self._flushed = 0
        self.checks = 0

    def flush(self):
        if self.checks > self._flushed:
            self._ledger.charge("collision_queries", self.checks - self._flushed)
            self._flushed = self.checks

    def isValid(self, state):
        lo, hi, q = self._lo, self._hi, self._q
        if self._n == 3:
            a, b, c = state[0], state[1], state[2]
            if not (lo[0] <= a <= hi[0] and lo[1] <= b <= hi[1] and lo[2] <= c <= hi[2]):
                return False
            q[0], q[1], q[2] = a, b, c
        else:
            for i in range(self._n):
                v = state[i]
                if not lo[i] <= v <= hi[i]:
                    return False
                q[i] = v
        self.checks += 1
        try:
            return bool(self._free(q))
        except RuntimeError:
            return False


class _StoppingValidity(_DrakeValidity):
    def isValid(self, state):
        return not self._ledger.deadline_exceeded() and super().isValid(state)


class OmplPlanner:
    def __init__(self, kind="rrt_connect"):
        if kind not in KINDS:
            raise ValueError(f"unknown ompl planner: {kind}")
        self.kind = kind
        self.cfg = {'kind': kind, 'range': 'ompl_default', 'budget_s': None}
        self.name = f"ompl_{kind}"
        self.graph_capture = None
        self.version = 2

    def solve(self, request, goals=None):
        ledger = Ledger(request.budget)
        robot, scene = request.robot, request.scene
        path = search_planner = si = validity = None
        status, stats = SolveStatus.UNKNOWN, {}
        started = perf_counter()
        try:
            reason = ("invalid:uncapped_wall_clock" if request.budget.wall_clock_s is None
                      else validate_start(request, ledger))
            if not reason:
                goals, reason = goals_for(request, ledger, goals)
            if reason:
                if reason.startswith("invalid:"):
                    status = SolveStatus.INVALID_INPUT
            else:
                si, validity = _space(robot, request.certifier, lambda si: _DrakeValidity(
                    si, robot, scene, request.certifier.padding(robot), ledger))
                stats.update(resolution_rad=request.certifier.edge_step_rad,
                             rng_seed=_seed(request.seed.entropy), goals=int(len(goals)))
                path, search_planner = self._search(si, request, goals, started, stats)
        except BudgetExhausted as exhausted:
            reason = f"budget:{exhausted.field}"
        except RuntimeError:
            reason = "ompl:runtime_error"
        if path is not None:
            status, reason = SolveStatus.SOLVED, ""
        elif not reason:
            reason = "search:unsolved"
        if validity is not None:
            validity.flush()
            stats["validity_checks"] = validity.checks
        result = finish(request, ledger, self.name, status=status, reason=reason, waypoints=path)
        result.search_stats.update(stats)
        if search_planner is not None:
            captured = perf_counter()
            result = replace(result, graph=self.graph_capture(
                search_planner, si, lambda state: [state[i] for i in range(robot.n_joints)]))
            result.search_stats['graph_capture_s'] = perf_counter() - captured
        return result

    def _search(self, si, request, goals, started, stats):
        n = request.robot.n_joints
        start = si.allocState()
        for i in range(n):
            start[i] = float(request.start_q_rad[i])
        alive = [start]
        goal = ob.GoalStates(si)
        for goal_q in np.atleast_2d(np.asarray(goals, dtype=float)):
            state = si.allocState()
            for i in range(n):
                state[i] = float(goal_q[i])
            if si.isValid(state):
                goal.addState(state)
                alive.append(state)
        stats["goal_states"] = count = int(goal.getStateCount())
        if not count or not si.isValid(start):
            return None, None
        pdef = ob.ProblemDefinition(si)
        pdef.addStartState(start)
        pdef.setGoal(goal)
        planner = self._make(si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        stats["range_rad"] = float(planner.getRange())
        wall = float(request.budget.wall_clock_s)
        cap = max(1e-3, wall - min(VERIFY_RESERVE_S, .25*wall))
        ptc = ob.timedPlannerTerminationCondition(float(max(1e-3, cap - (perf_counter() - started))))
        solved = planner.solve(ptc)
        del ptc
        captured = planner if self.graph_capture is not None else None
        if not (bool(solved) and pdef.hasExactSolution()):
            return None, captured
        geometric = pdef.getSolutionPath()
        out = np.array([[s[i] for i in range(n)] for s in geometric.getStates()], dtype=float)
        del geometric, alive
        return (out, captured) if len(out) >= 2 else (None, captured)

    def _make(self, si):
        return {'rrt_connect': og.RRTConnect, 'rrt_star': og.RRTstar, 'rrt': og.RRT}[self.kind](si)


def simplify_path(robot, scene, path, *, cfg, ledger, seed_entropy=0):
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
        return None
    if len(points) == 2:
        return points.copy()
    if ledger is None:
        from ..budget import Budget
        ledger = Ledger(Budget(wall_clock_s=None))
    cap = ledger.budget.edge_certifications
    if ledger.deadline_exceeded() or (cap is not None and ledger.counts.get('edge_certifications', 0) >= cap):
        return None
    si, validity = _space(robot, cfg, lambda si: _StoppingValidity(si, robot, scene, cfg.padding(robot), ledger))
    _seed(seed_entropy)
    geometric = og.PathGeometric(si)
    states = []
    n = robot.n_joints
    for q in points:
        state = si.allocState()
        for i in range(n):
            state[i] = float(q[i])
        states.append(state)
        geometric.append(state)
    try:
        og.PathSimplifier(si).simplifyMax(geometric)
    finally:
        validity.flush()
    out = np.array([[s[i] for i in range(n)] for s in geometric.getStates()], dtype=float)
    del geometric, states
    if len(out) < 2 or not np.isfinite(out).all():
        return None
    out[0], out[-1] = points[0], points[-1]
    return out
