import sys
from os.path import dirname, abspath, join
from os import listdir

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.trunk.model import trunkRobot
from sofacontrol.open_loop_sequences import TrunkRobotSequences

from sofacontrol.scp.models.ssm import SSMGuSTO
from sofacontrol.measurement_models import linearModel, OutputModel
from sofacontrol.scp.ros import runGuSTOSolverNode
from sofacontrol.utils import QuadraticCost, HyperRectangle, load_data, qv2x, Polyhedron
from sofacontrol.SSM.observer import SSMObserver, DiscreteEKFObserver
from sofacontrol.SSM import adiabatic_ssm
from sofacontrol.SSM.controllers import scp
import pickle

import matplotlib.pyplot as plt

# Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51
N_NODES = 709

MODEL_NAMES = ["north", "west", "origin", "east", "south", "northwest", "northeast", "southwest", "southeast"] # 

useTimeDelay = True

# Load equilibrium point
rest_file = join(path, 'rest_qv.pkl')
rest_data = load_data(rest_file)
q_equilibrium = np.array(rest_data['q'][0])

# Setup equilibrium point (no time delay and observed position and velocity of tip)
x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

# Set directory for SSM Models
pathToModel = "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms"  # join(path, "SSMmodels", "model_004")

# load local SSM models
raw_params = {}
raw_models = []
model_names = MODEL_NAMES # listdir(pathToModel)
print(model_names)
for model_name in model_names:
    with open(join(pathToModel, model_name, "SSMmodel_delay-embedding", "SSM_model.pkl"), 'rb') as f:
        SSM_data = pickle.load(f)
    raw_models.append(SSM_data['model'])
    raw_params = SSM_data['params']

if raw_params['delay_embedding']:
    outputModel = linearModel([TIP_NODE], N_NODES, vel=False)
    z_eq_point = outputModel.evaluate(x_eq, qv=False)
    if useTimeDelay:
        # obs are pos of tip + n_delay time-delayed versions of it
        outputSSMModel = OutputModel(15, 3) # TODO: hardcoded n_delay = 4
    else:
        # obs are pos and vel of tip
        outputSSMModel = OutputModel(6, 3)
    Cout = outputSSMModel.C
else:
    outputModel = linearModel([TIP_NODE], N_NODES)
    z_eq_point = outputModel.evaluate(x_eq, qv=True)
    Cout = None

model = adiabatic_ssm.AdiabaticSSMDynamics(z_eq_point, raw_models, raw_params, discrete=False, discr_method='be', debug=True, C=Cout)

# This dt for when to recalculate control
dt = 0.02

cost = QuadraticCost()
Qz = np.zeros((model.output_dim, model.output_dim))
Qz[0, 0] = 100.  # corresponding to x position of end effector
Qz[1, 1] = 100.  # corresponding to y position of end effector
Qz[2, 2] = 0 # 100.  # corresponding to z position of end effector
R = 0.001 * np.eye(model.input_dim)
cost.R = R
cost.Q = model.H.T @ Qz @ model.H

# Define target trajectory for optimization
# === figure8 ===
M = 1
T = 10
N = 1000
radius = 45.
t = np.linspace(0, M * T, M * N + 1)
th = np.linspace(0, M * 2 * np.pi, M * N + 1)
zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.output_dim - len(z_eq_point)))), (M * N + 1, 1))
# zf_target = np.zeros((M*N+1, 6))
zf_target[:, 0] += -radius * np.sin(th)
zf_target[:, 1] += radius * np.sin(2 * th)
# zf_target[:, 2] += -np.ones(len(t)) * 20

# === circle with constant z ===
# M = 1
# T = 10
# N = 1000
# radius = 20.
# t = np.linspace(0, M * T, M * N + 1)
# th = np.linspace(0, M * 2 * np.pi, M * N + 1) # + np.pi / 2
# zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.output_dim - len(z_eq_point)))), (M * N + 1, 1))
# # zf_target = np.zeros((M*N+1, 6))
# zf_target[:, 0] += radius * np.cos(th)
# zf_target[:, 1] += radius * np.sin(th)
# zf_target[:, 2] += -np.ones(len(t)) * 20

model.z_target = model.zfyf_to_zy(zf=zf_target)


def run_scp():
    """
     In problem_specification add:

     from examples.trunk import trunk_SSM
     problem = trunk_SSM.run_scp

     then run:

     python3 launch_sofa.py
     """
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.measurement_models import linearModel, OutputModel
    from sofacontrol.utils import QuadraticCost, qv2x, load_data, Polyhedron
    from sofacontrol.SSM import adiabatic_ssm
    from sofacontrol.SSM.observer import SSMObserver, DiscreteEKFObserver
    from sofacontrol.SSM.controllers import scp
    import pickle

    prob = Problem()
    prob.Robot = trunkRobot()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3)
    cov_v = 0.0 * np.eye(3) # * len(DEFAULT_OUTPUT_NODES))
    prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE])

    if raw_params['delay_embedding']:
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=False, S_q=cov_q)
    else:
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=True, S_q=cov_q, S_v=cov_v)

    # Pure SSM Manifold Observer
    observer = SSMObserver(model)

    # Define controller (wait 3 seconds of simulation time to start)
    prob.controller = scp(model, cost, dt, N_replan=2, delay=1, feedback=False, EKF=observer)

    # Saving paths
    prob.opt['sim_duration'] = 11
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'ssmr'

    return prob


def run_gusto_solver():
    """
    python3 trunk_SSM.py run_gusto_solver
    """
    # Define initial condition to be x_ref for initial solve
    x0 = np.zeros(model.state_dim)

    z = model.zfyf_to_zy(zf=zf_target)

    N = 3

    # Control constraints
    u_min, u_max = 0.0, 800.0
    U = HyperRectangle([u_max] * model.input_dim, [u_min] * model.input_dim)
    # input rate constraints
    dU = HyperRectangle([10] * model.input_dim, [-10] * model.input_dim) # None # 
    # dU = None

    # State constraints
    X = None

    # Define GuSTO model
    gusto_model = SSMGuSTO(model)

    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
                       verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI',
                       max_gusto_iters=0, input_nullspace=None, dU=dU, jit=False)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'module_test':
        module_test()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    else:
        raise RuntimeError('Not a valid function argument')