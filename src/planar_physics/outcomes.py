from enum import StrEnum

class MissionOutcome(StrEnum):
    HELD = 'solved_and_held'
    NOT_HELD = 'not_held'
    PLAN_FAILED = 'plan_failed'
    TRAJECTORY_REJECTED = 'trajectory_rejected'
    DEADLINE = 'deadline_exceeded'
