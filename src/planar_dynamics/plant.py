from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import MultibodyForces
from pydrake.systems.analysis import ApplySimulatorConfig, Simulator, SimulatorConfig
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, TriggerType
from pydrake.systems.primitives import Adder, Demultiplexer, LogVectorOutput, MatrixGain, Multiplexer, Saturation, TrajectorySource, ZeroOrderHold
from pydrake.trajectories import PiecewisePolynomial
from planar_physics.drake_model import get_model, parse_arm
from planar_physics.contract import DEFAULT_CONTRACT

def inverse_dynamics(robot, q, dq, ddq):
    model = get_model(robot)
    plant, context = model.plant, model.plant_context
    qs, dqs, ddqs = np.atleast_2d(q), np.atleast_2d(dq), np.atleast_2d(ddq)
    n = qs.shape[1]
    forces = MultibodyForces(plant)
    state = np.empty(2*n)
    torques = np.empty((len(qs), n))
    for k in range(len(qs)):
        state[:n], state[n:] = qs[k], dqs[k]
        plant.SetPositionsAndVelocities(context, state)
        forces.SetZero()
        plant.CalcForceElementsContribution(context, forces)
        torques[k] = plant.CalcInverseDynamics(context, ddqs[k], forces)
    return torques if np.ndim(q) > 1 else torques[0]

@dataclass(frozen=True, slots=True)
class PlantConfig:
    timestep_s: float = DEFAULT_CONTRACT.dt_s

class PlanarPlant:
    @staticmethod
    def build(robot, config=PlantConfig()):
        n = robot.n_joints
        gains = np.array([DEFAULT_CONTRACT.kp, DEFAULT_CONTRACT.kd])
        builder = DiagramBuilder()
        plant = builder.AddSystem(MultibodyPlant(time_step=0.0))
        instance = parse_arm(plant, robot)
        for j in range(n):
            plant.GetJointByName(f"j{j+1}").set_position_limits([-np.inf], [np.inf])
        plant.Finalize()
        source = builder.AddSystem(TrajectorySource(PiecewisePolynomial(np.zeros(n)), 2))
        desired = builder.AddSystem(Demultiplexer([2*n, n]))
        controller = builder.AddSystem(InverseDynamicsController(plant, np.full(n, gains[0]), np.zeros(n), np.full(n, gains[1]), True))
        builder.Connect(plant.get_state_output_port(), controller.get_input_port_estimated_state())
        builder.Connect(source.get_output_port(), desired.get_input_port())
        builder.Connect(desired.get_output_port(0), controller.get_input_port_desired_state())
        builder.Connect(desired.get_output_port(1), controller.get_input_port_desired_acceleration())
        caps = np.asarray(robot.max_joint_torque_nm)
        magnitude = builder.AddSystem(Saturation(-caps, caps))
        memory = builder.AddSystem(ZeroOrderHold(config.timestep_s, 2*n))
        sampled = builder.AddSystem(Demultiplexer([n, n]))
        builder.Connect(controller.get_output_port_control(), magnitude.get_input_port())
        builder.Connect(memory.get_output_port(), sampled.get_input_port())
        pair = builder.AddSystem(Multiplexer([n, n]))
        difference = builder.AddSystem(MatrixGain(np.c_[np.eye(n), -np.eye(n)]))
        increment = np.asarray(robot.max_joint_torque_rate_nm_s)*config.timestep_s
        slew = builder.AddSystem(Saturation(-increment, increment))
        add = builder.AddSystem(Adder(2, n))
        builder.Connect(magnitude.get_output_port(), pair.get_input_port(0))
        builder.Connect(sampled.get_output_port(0), pair.get_input_port(1))
        builder.Connect(pair.get_output_port(), difference.get_input_port())
        builder.Connect(difference.get_output_port(), slew.get_input_port())
        builder.Connect(slew.get_output_port(), add.get_input_port(0))
        builder.Connect(sampled.get_output_port(0), add.get_input_port(1))
        actuator = builder.AddSystem(Multiplexer([n, n]))
        builder.Connect(add.get_output_port(), actuator.get_input_port(0))
        builder.Connect(controller.get_output_port_control(), actuator.get_input_port(1))
        builder.Connect(actuator.get_output_port(), memory.get_input_port())
        builder.Connect(sampled.get_output_port(0), plant.get_actuation_input_port())
        mux = builder.AddSystem(Multiplexer([2*n, 2*n, 2*n]))
        builder.Connect(plant.get_state_output_port(), mux.get_input_port(0))
        builder.Connect(memory.get_output_port(), mux.get_input_port(1))
        builder.Connect(desired.get_output_port(0), mux.get_input_port(2))
        logger = LogVectorOutput(mux.get_output_port(), builder, {TriggerType.kPeriodic, TriggerType.kForced}, config.timestep_s)
        diagram = builder.Build()
        native = SimpleNamespace(diagram=diagram, plant=plant, source=source, limiter=memory,
                                 logger=logger, model_instance=instance)
        return PlanarPlant(robot, native, config, diagram.CreateDefaultContext())

    def __init__(self, robot, native, config, context):
        self.robot, self.native, self.config = robot, native, config
        self.model = native.plant
        self.root_context = context
        self.context = self.model.GetMyMutableContextFromRoot(context)
        self.simulator = Simulator(native.diagram, context)
        ApplySimulatorConfig(SimulatorConfig(integration_scheme="runge_kutta3", max_step_size=config.timestep_s, use_error_control=False, start_time=context.get_time()), self.simulator)

    def _vector(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (self.robot.n_joints,) or not np.isfinite(value).all():
            raise ValueError("joint vectors must have one finite value per joint")
        return value

    def reset(self, q, dq=None, applied_torque_nm=None):
        q = self._vector(q)
        dq = np.zeros(self.robot.n_joints) if dq is None else self._vector(dq)
        self.root_context.SetTime(0.)
        self.model.SetPositionsAndVelocities(self.context, np.r_[q, dq])
        caps = np.asarray(self.robot.max_joint_torque_nm)
        tau = np.clip(self.holding_torque(q) if applied_torque_nm is None else self._vector(applied_torque_nm), -caps, caps)
        self.native.limiter.GetMyMutableContextFromRoot(self.root_context).get_mutable_discrete_state_vector().SetFromVector(np.r_[tau, tau])
        self.set_reference(PiecewisePolynomial(q))
        self.native.logger.FindMutableLog(self.root_context).Clear()
        self.simulator.clear_monitor()
        self.simulator.Initialize()

    def set_reference(self, trajectory):
        self.native.source.UpdateTrajectory(trajectory)
        self.root_context.SetAllCacheEntriesOutOfDate()

    @property
    def q(self):
        return self.model.GetPositions(self.context).copy()

    @property
    def dq(self):
        return self.model.GetVelocities(self.context).copy()

    @property
    def time_s(self):
        return float(self.root_context.get_time())

    def holding_torque(self, q):
        zero = np.zeros(self.robot.n_joints)
        return inverse_dynamics(self.robot, q, zero, zero)

    def log(self):
        return self.native.logger.FindLog(self.root_context)
