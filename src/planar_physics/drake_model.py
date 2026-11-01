from collections import OrderedDict
from types import SimpleNamespace
import logging
import os
import threading
import numpy as np
from pydrake.common import Parallelism
from pydrake.geometry import Capsule, CollisionFilterDeclaration, GeometrySet, Sphere
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import CoulombFriction
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
from .robot import _readonly


logging.getLogger('drake').setLevel(
    getattr(logging, os.environ.get('PLANAR_DRAKE_LOG', 'WARNING').upper(), logging.WARNING))

def parse_arm(plant, robot):
    instance = Parser(plant).AddModelsFromString(robot._urdf_text, "urdf")[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName('base', instance),
                     RigidTransform([*robot.base_xy_m, 0]))
    plant.mutable_gravity_field().set_gravity_vector([*robot.gravity_m_s2, 0])
    return instance

def _parameters(plant, scene_graph, instance):
    n = plant.num_positions(instance)
    links = tuple(plant.GetBodyByName(f'link{k+1}', instance) for k in range(n))
    joints = tuple(plant.GetJointByName(f'j{k+1}', instance) for k in range(n))
    tip = plant.GetFrameByName('tip', instance)
    lengths = _readonly([j.frame_on_parent().GetFixedPoseInBodyFrame().translation()[0]
                         for j in joints[1:]] + [tip.GetFixedPoseInBodyFrame().translation()[0]])
    radii = [scene_graph.model_inspector().GetShape(plant.GetCollisionGeometriesForBody(b)[0]).radius() for b in links]
    return SimpleNamespace(n_joints=n, lengths_m=lengths, reach_m=float(sum(lengths)),
        lo_rad=_readonly(plant.GetPositionLowerLimits()), hi_rad=_readonly(plant.GetPositionUpperLimits()),
        joint_limits_rad=tuple(zip(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())),
        max_joint_speed_rad_s=tuple(plant.GetVelocityUpperLimits()),
        max_joint_torque_nm=tuple(plant.GetJointActuatorByName(f'motor{k+1}', instance).effort_limit() for k in range(n)),
        joint_viscous_damping_nms_rad=tuple(j.default_damping() for j in joints),
        link_masses_kg=tuple(b.default_mass() for b in links), link_radius_m=max(radii),
        clearance_lipschitz_m=float(np.linalg.norm(np.cumsum(lengths[::-1]))))

def build_model(robot, scene=None, *, time_step=0.):
    builder = RobotDiagramBuilder(time_step=time_step)
    plant, scene_graph = builder.plant(), builder.scene_graph()
    instance = parse_arm(plant, robot)
    extent = 0.
    def world_shape(shape, pose, name, color):
        plant.RegisterCollisionGeometry(plant.world_body(), pose, shape, name, CoulombFriction(.5, .5))
        plant.RegisterVisualGeometry(plant.world_body(), pose, shape, name, color)
    if scene is not None:
        extent = max((float(np.linalg.norm(point - robot.base_xy_m))
                      for piece in scene.wall_pieces_m for point in piece.reshape(2, 2)), default=0.)
        for k, (x1, y1, x2, y2) in enumerate(scene.wall_pieces_m):
            length = float(np.hypot(x2-x1, y2-y1))
            shape = Capsule(scene.wall.thickness_m/2, length) if length else Sphere(scene.wall.thickness_m/2)
            pose = RigidTransform(RotationMatrix.MakeXRotation(np.pi/2), [(x1+x2)/2, (y1+y2)/2, 0])
            world_shape(shape, pose, f'wall{k}', [.5, .5, .55, 1])
    plant.Finalize()
    parameters = _parameters(plant, scene_graph, instance)
    if not robot.self_collision:
        bodies = [plant.GetBodyByName(f'link{k+1}', instance) for k in range(parameters.n_joints)]
        scene_graph.collision_filter_manager().Apply(CollisionFilterDeclaration().ExcludeWithin(
            GeometrySet([g for body in bodies for g in plant.GetCollisionGeometriesForBody(body)])))
    checker = SceneGraphCollisionChecker(model=builder.Build(), robot_model_instances=[instance],
        edge_step_size=.01, env_collision_padding=0., self_collision_padding=0., implicit_context_parallelism=Parallelism(1))
    diagram = checker.model()
    plant, scene_graph = diagram.plant(), diagram.scene_graph()
    links = tuple(plant.GetBodyByName(f'link{k+1}', instance) for k in range(parameters.n_joints))
    context = diagram.CreateDefaultContext()
    pc, sc = plant.GetMyMutableContextFromRoot(context), scene_graph.GetMyMutableContextFromRoot(context)
    plant.get_actuation_input_port().FixValue(pc, np.zeros(parameters.n_joints))
    return SimpleNamespace(diagram=diagram, plant=plant, scene_graph=scene_graph, root_context=context,
        plant_context=pc, scene_context=sc, model_instance=instance, link_bodies=links,
        tip_frame=plant.GetFrameByName('tip', instance),
        checker=checker, checkers=OrderedDict(), parameters=parameters,
        influence_distance_m=2*parameters.reach_m+extent+parameters.link_radius_m+1.)

_local = threading.local()

def get_model(robot, scene=None):
    key = (robot.digest(), scene.identity if scene is not None else None)
    if not hasattr(_local, 'models'):
        _local.models = OrderedDict()
    cache = _local.models
    if key not in cache:
        cache[key] = build_model(robot, scene)
        if len(cache) > 32:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]

def get_checker(robot, scene=None, *, padding=0., step=.004):
    model = get_model(robot, scene)
    key, cache = (padding, step), model.checkers
    if key not in cache:
        checker = model.checker.Clone()
        checker.SetPaddingAllRobotEnvironmentPairs(padding)
        checker.SetPaddingAllRobotRobotPairs(padding)
        checker.set_edge_step_size(step)
        cache[key] = checker
        if len(cache) > 8:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]
