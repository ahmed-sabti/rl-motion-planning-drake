from planar_physics.contract import SCENARIO, VARIANT

GOALS = {**SCENARIO.goals, 'edge': [.80 if VARIANT == '3r' else .55, 0.]}


def goal_names(scene):
    return list(GOALS) if scene == 'nominal' else list(SCENARIO.goals)
