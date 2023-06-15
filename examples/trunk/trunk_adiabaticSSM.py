import sys
from os.path import dirname, abspath, join, isdir
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

# Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51
N_NODES = 709
# Set directory for SSM Models
pathToModel = "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms_N=33_handcrafted" # 100_sparsity=0.95" # 9" # 

MODEL_NAMES = [name for name in sorted(listdir(pathToModel)) if isdir(join(pathToModel, name))]
# remove models with opposite side strings and by visually inspection of eigenvalues (autonomous linear part)
# use_models = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 26, 27, 29, 30, 31, 32, 39, 41, 42, 45, 46, 47, 48, 51, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 89, 91, 93, 94, 96, 97, 98, 99]
# remove models automatically:
# use_models = [0, 1, 2, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 19, 21, 22, 23, 25, 26, 27, 31, 33, 36, 42, 43, 45, 47, 48, 49, 50, 53, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 76, 77, 80, 81, 87, 89, 91, 93, 96, 98]
# MODEL_NAMES = [MODEL_NAMES[i] for i in use_models]

useTimeDelay = True

# Load equilibrium point
rest_file = join(path, 'rest_qv.pkl')
rest_data = load_data(rest_file)
q_equilibrium = np.array(rest_data['q'][0])

# Setup equilibrium point (no time delay and observed position and velocity of tip)
x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

# load local SSM models
raw_params = {}
raw_models = []
model_names = MODEL_NAMES # listdir(pathToModel)
print(model_names)
for model_name in model_names:
    with open(join(pathToModel, model_name, "SSMmodel_delay-embedding_ROMOrder=3_globalV", "SSM_model.pkl"), 'rb') as f: # 
        SSM_data = pickle.load(f)
    with open(join(pathToModel, model_name, "rest_q.pkl"), "rb") as f:
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
Qz[2, 2] = 0.  # corresponding to z position of end effector
R = 0.001 * np.eye(model.input_dim)
cost.R = R
cost.Q = model.H.T @ Qz @ model.H

# Define target trajectory for optimization
# === figure8 ===
M = 1
T = 10
N = 1000
radius = 30.
tf = np.linspace(0, M * T, M * N + 1)
th = np.linspace(0, M * 2 * np.pi, M * N + 1) # + np.pi
zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.output_dim - len(z_eq_point)))), (M * N + 1, 1))
# zf_target = np.zeros((M*N+1, 3))
zf_target[:, 0] += -radius * np.sin(th)
zf_target[:, 1] += radius * np.sin(2 * th)
# zf_target[:, 2] += -np.ones(len(t)) * 10

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
# zf_target[:, 2] += -np.ones(len(t)) * 10

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
    dU = HyperRectangle([1] * model.input_dim, [-1] * model.input_dim) # None # 
    # dU = None

    # State constraints
    X = None

    # Define GuSTO model
    gusto_model = SSMGuSTO(model)

    print(z)

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