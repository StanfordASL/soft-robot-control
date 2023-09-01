"""
This file integrates open-source code (https://github.com/ramvasudevan/soft-robot-koopman) with Sofa.

Open-source code accompanies following papers:https://arxiv.org/abs/1902.02827 (Modeling and Control of Soft Robots
                                                                                Using the Koopman Operator and MPC)

Non-listed MATLAB dependencies for soft-robot-koopman:
- MATLAB 2019a (9.6)
- Control System Toolbox 10.6
- DSP System Toolbox 9.8
- Model Predictive Control Toolbox 6.3
- Optimization Toolbox 8.3
- Signal Processing Toolbox 8.2
- Statistics and Machine Learning Toolbox 11.5
- Symbolic Math Toolbox 8.3


Baseline comparison for Trunk robot consists of multiple scripts that interface with Sofa, namely :
- Collection of training data for Koopman: train_koopman_collection()
- Collection of validation data for Koopman: validation_data_koopman_collection()
Next, a file which uses the built model for control: test_koopman_mpc()

To build the model, run generate_koopman_model.m by adding this file to the soft-robot-koopman repo

"""

import sys
from os.path import dirname, abspath, join

import numpy as np

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.trunk.model import trunkRobot

from sofacontrol.utils import QuadraticCost, HyperRectangle, Polyhedron, qv2x, load_data
from sofacontrol.measurement_models import MeasurementModel
from scipy.io import loadmat

from sofacontrol.tpwl.tpwl_utils import Target

from sofacontrol.baselines.koopman import koopman_utils

# Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51
N_NODES = 709

# Load equilibrium point
rest_file = join(path, 'rest_qv.pkl')
rest_data = load_data(rest_file)
q_equilibrium = np.array(rest_data['q'][0])

# Setup equilibrium point (no time delay and observed position and velocity of tip)
x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

koopman_data = loadmat(join(path, 'koopman_model.mat'))['py_data'][0, 0]
raw_model = koopman_data['model']
raw_params = koopman_data['params']
model = koopman_utils.KoopmanModel(raw_model, raw_params)
scaling = koopman_utils.KoopmanScaling(scale=model.scale)

print(scaling.y_offset)
print(scaling.u_offset)
print(scaling.y_factor)
print(scaling.u_factor)

cov_q = 0.0 * np.eye(3)
outputModel = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=False, S_q=cov_q)
z_eq_point = outputModel.evaluate(x_eq)

cost = QuadraticCost()
target = Target()

# Define target trajectory for optimization
# === figure8 ===
M = 1
T = 10
N = 1000
radius = 30.
t = np.linspace(0, M * T, M * N)
th = np.linspace(0, M * 2 * np.pi, M * N)
zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.n - len(z_eq_point)))), (M * N, 1))
# zf_target = np.zeros((M * N, model.n))
zf_target[:, 0] += -radius * np.sin(th)
zf_target[:, 1] += radius * np.sin(2 * th)
# zf_target[:, 2] += -np.ones(len(t)) * 10

# # === circle with constant z ===
# M = 1
# T = 10
# N = 1000
# radius = 20.
# t = np.linspace(0, M * T, M * N )
# th = np.linspace(0, M * 2 * np.pi, M * N)
# zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.n - len(z_eq_point)))), (M * N, 1))
# # zf_target = np.zeros((M * N, model.n))
# zf_target[:, 0] += radius * np.cos(th)
# zf_target[:, 1] += radius * np.sin(th)
# zf_target[:, 2] += -np.ones(len(t)) * 10

# === Pac-Man (3D) ===
# M = 1
# T = 10
# N = 1000
# radius = 20.
# t = np.linspace(0, M * T, M * N + 1)
# th = np.linspace(0, M * 2 * np.pi, M * N + 1)
# zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.n - len(z_eq_point)))), (M * N + 1, 1))
# # zf_target = np.zeros((M * N, model.output_dim))
# print("z_eq_point:", z_eq_point)
# zf_target[:, 0] += radius * np.cos(th)
# zf_target[:, 1] += radius * np.sin(th)
# zf_target[:, 2] += -np.ones(len(t)) * 10
# t_in_pacman, t_out_pacman = 1., 1.
# zf_target[t < t_in_pacman, :] = z_eq_point + (zf_target[t < t_in_pacman][-1, :] - z_eq_point) * (t[t < t_in_pacman] / t_in_pacman)[..., None]
# zf_target[t > T - t_out_pacman, :] = z_eq_point + (zf_target[t > T - t_out_pacman][0, :] - z_eq_point) * (1 - (t[t > T - t_out_pacman] - (T - t_out_pacman)) / t_out_pacman)[..., None]

# Cost
cost.R = .0001 * np.eye(model.m)
cost.Q = np.zeros((model.n, model.n))
cost.Q[0, 0] = 100  # corresponding to x position of end effector
cost.Q[1, 1] = 100  # corresponding to y position of end effector
# cost.Q[2, 2] = 100  # corresponding to z position of end effector


def generate_koopman_data():
    """
    Prepare Koopman data in .mat format for MATLAB Koopman code (generate_koopman_model.m)
    """
    from scipy.io import savemat
    from sofacontrol.utils import load_data, qv2x
    from sofacontrol.measurement_models import linearModel

    koopman_data_name = 'pod_snapshots'
    num_nodes = N_NODES
    ee_node = [TIP_NODE]
    koopman_data = load_data(join(path, '{}.pkl'.format(koopman_data_name)))

    state = qv2x(q=koopman_data['q'], v=koopman_data['v'])
    names = ['ee_pos']
    measurement_models = [linearModel(nodes=ee_node, num_nodes=num_nodes, pos=True, vel=False)]

    for i, name in enumerate(names):
        mat_data_file = join(path, '{}.mat'.format(name))
        y = measurement_models[i].evaluate(x=state.T)
        matlab_data = dict()
        matlab_data['y'] = y.T
        matlab_data['u'] = np.asarray(koopman_data['u'])
        matlab_data['t'] = np.atleast_2d(koopman_data['dt'] * np.asarray(range(len(matlab_data['u']))))

        savemat(mat_data_file, matlab_data)


def run_koopman():
    """
    In problem_specification add:

    from examples.trunk import trunk_koopman
    problem = trunk_koopman.run_koopman

    then run:

    python3 launch_sofa.py
    """
    from sofacontrol.baselines.koopman import koopman
    from robots import environments
    from sofacontrol.closed_loop_controller import ClosedLoopController

    prob = Problem()
    prob.Robot = environments.Trunk()
    prob.ControllerClass = ClosedLoopController

    cov_q = 0.0 * np.eye(3)
    prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=False, S_q=cov_q)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE])

    print("Koopman controller dt:", model.Ts)
    prob.controller = koopman.KoopmanMPC(dyn_sys=model, dt=model.Ts, delay=1, rollout_horizon=1)

    prob.opt['sim_duration'] = 11.  # Simulation time, optional
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'koopman'

    return prob


def run_koopman_solver():
    """
    python3 diamond_koopman.py run_koopman_solver
    """
    from sofacontrol.baselines.ros import runMPCSolverNode
   
    # Define target trajectory for optimization
    target.t = t
    target.z = scaling.scale_down(y=zf_target)
    target.u = scaling.scale_down(u=np.zeros(model.m)).reshape(-1)

    # Consider same "scaled" cost parameters as other models
    cost.R *= np.diag(scaling.u_factor[0])
    cost.Q *= np.diag(scaling.y_factor[0])

    # Control constraints
    u_ub = 800. * np.ones(model.m)
    u_lb = 0. * np.ones(model.m)
    u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
    u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
    print(u_ub_norm)
    print(u_lb_norm)
    U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)
    # dU = HyperRectangle(scaling.scale_down(u=[10] * model.m).reshape(-1), scaling.scale_down(u=[-10] * model.m).reshape(-1))
    dU = None

    N = 3

    runMPCSolverNode(model=model, N=N, cost_params=cost, target=target, dt=model.Ts, verbose=1,
                     warm_start=True, U=U, dU=dU) # , solver='GUROBI')



if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'generate_koopman_data':
        generate_koopman_data()
    elif sys.argv[1] == 'run_koopman_solver':
        run_koopman_solver()
    else:
        raise RuntimeError('Not a valid function argument')
