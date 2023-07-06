import sys
from os.path import dirname, abspath, join, isdir, exists
from os import listdir

import numpy as np
from matplotlib import pyplot as plt

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.trunk.model import trunkRobot

from sofacontrol.scp.models.ssm import SSMGuSTO
from sofacontrol.measurement_models import linearModel, OutputModel
from sofacontrol.scp.ros import runGuSTOSolverNode
from sofacontrol.utils import QuadraticCost, HyperRectangle, load_data, qv2x
from sofacontrol.SSM import adiabatic_ssm
import pickle

# Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51
N_NODES = 709
# Set directory for SSM Models
PATH_TO_MODEL = "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms_N=100_sparsity=0.95" # 33_handcrafted" # 147" # 
MODEL_NAMES = [name for name in sorted(listdir(PATH_TO_MODEL)) if isdir(join(PATH_TO_MODEL, name))]
# if exists(join(PATH_TO_MODEL, "use_models.pkl")):
#     with open(join(PATH_TO_MODEL, "use_models.pkl"), "rb") as f:
#         USE_MODELS = pickle.load(f)
# else:
#     raise FileNotFoundError("No use_models.pkl file found in model directory")
# USE_MODELS = [int(i) for i in []] # ['096', '077', '098', '053', '032', '018', '064', '082', '031', '061']]
USE_MODELS = list(range(len(MODEL_NAMES)))
# USE_MODELS = [0, 52, 40, 17, 92, 28, 74, 15, 77, 83, 82, 88, 11, 93, 98, 84, 56, 62, 21, 16, 31, 38, 68, 14, 23, 67, 18, 75, 46, 81, 34, 44, 37, 99, 85, 95, 35, 64, 9, 79, 90, 59, 50, 97, 51, 20, 27, 32, 25, 49, 39, 43, 55, 29]
MODEL_NAMES = [MODEL_NAMES[i] for i in USE_MODELS]
print("Using models: ", MODEL_NAMES)

useTimeDelay = True
useDefaultModels = False

# Load equilibrium point
rest_file = join(path, 'rest_qv.pkl')
rest_data = load_data(rest_file)
q_equilibrium = np.array(rest_data['q'][0])

# Setup equilibrium point (no time delay and observed position and velocity of tip)
x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

# load local SSM models
raw_params = {}
raw_models = []

if useDefaultModels:
    PATH_TO_DEFAULT_MODELS = "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms_N=9_old"
    for model_name in [name for name in sorted(listdir(PATH_TO_DEFAULT_MODELS)) if isdir(join(PATH_TO_DEFAULT_MODELS, name))]:
        with open(join(PATH_TO_DEFAULT_MODELS, model_name, "SSMmodel_delay-embedding_ROMOrder=3_globalV_fixed-delay", "SSM_model.pkl"), 'rb') as f:
            SSM_data = pickle.load(f)
        with open(join(PATH_TO_DEFAULT_MODELS, model_name, "rest_q.pkl"), "rb") as f:
            q_eq = pickle.load(f)
        SSM_data['model']['q_eq'] = (q_eq - q_equilibrium)[TIP_NODE*3:(TIP_NODE+1)*3]
        raw_models.append(SSM_data['model'])
        raw_params = SSM_data['params']

for model_name in MODEL_NAMES:
    with open(join(PATH_TO_MODEL, model_name, "SSMmodel_delay-embedding_ROMOrder=3_globalV_fixed-delay", "SSM_model.pkl"), 'rb') as f:
        SSM_data = pickle.load(f)
    with open(join(PATH_TO_MODEL, model_name, "rest_q.pkl"), "rb") as f:
        q_eq = pickle.load(f)
    SSM_data['model']['q_eq'] = (q_eq - q_equilibrium)[TIP_NODE*3:(TIP_NODE+1)*3]
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
Qz[2, 2] = 1000.  # corresponding to z position of end effector
R = 0.0001 * np.eye(model.input_dim)
cost.R = R
cost.Q = model.H.T @ Qz @ model.H

# Define target trajectory for optimization
# === figure8 (2D) ===
# M = 1
# T = 10
# N = 1000
# radius = 30.
# tf = np.linspace(0, M * T, M * N + 1)
# th = np.linspace(0, M * 2 * np.pi, M * N + 1) # + np.pi
# zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.output_dim - len(z_eq_point)))), (M * N + 1, 1))
# # zf_target = np.zeros((M * N, model.output_dim))
# zf_target[:, 0] += -radius * np.sin(th)
# zf_target[:, 1] += radius * np.sin(2 * th)
# # zf_target[:, 2] += -np.ones(len(t)) * 10

# === circle with constant z (3D) ===
M = 1
T = 10
N = 1000
radius = 20.
tf = np.linspace(0, M * T, M * N + 1)
th = np.linspace(0, M * 2 * np.pi, M * N + 1) # + 3 * np.pi / 2
zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.output_dim - len(z_eq_point)))), (M * N + 1, 1))
# zf_target = np.zeros((M * N, model.output_dim))
zf_target[:, 0] += radius * np.cos(th)
zf_target[:, 1] += radius * np.sin(th)
zf_target[:, 2] += -np.ones(len(tf)) * 10

model.z_target = model.zfyf_to_zy(zf=zf_target)


def run_scp(z=None, T=11.):
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

    if z is not None:
        model.z_target = z

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

    # Define controller (wait 1 seconds of simulation time to start)
    prob.controller = scp(model, cost, dt, N_replan=2, delay=1, feedback=False, EKF=observer)

    # Saving paths
    prob.opt['sim_duration'] = T
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'ssmr'

    return prob


def run_gusto_solver(t=None, z=None):
    """
    python3 trunk_SSM.py run_gusto_solver
    """
    # Define initial condition to be x_ref for initial solve
    x0 = np.zeros(model.state_dim)

    if z is None:
        t = tf
        z = model.zfyf_to_zy(zf=zf_target)
    else:
        model.z_target = z

    N = 3

    # Control constraints
    u_min, u_max = 0.0, 800.0
    U = HyperRectangle([u_max] * model.input_dim, [u_min] * model.input_dim)
    # input rate constraints
    # dU = HyperRectangle([10] * model.input_dim, [-10] * model.input_dim) # None # 
    dU = None
    # State constraints
    X = None

    # Define GuSTO model
    gusto_model = SSMGuSTO(model)

    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
                    verbose=0, warm_start=True, convg_thresh=0.001, solver='GUROBI',
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