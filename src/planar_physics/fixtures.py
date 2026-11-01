from .contract import SCENARIO
from .robot import Robot
from .scene import Scene

ROBOT_EQUAL_030 = Robot()
SCENES = {name: Scene() if scene is None else scene for name, scene in SCENARIO.scenes.items()}
