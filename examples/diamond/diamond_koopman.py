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
root = dirname(dirname(path))  # added another layer of dirname because pathing wasnt working,
# I suspect someone changed file structure but not this section
sys.path.append(root)

# Default nodes are the "end effector (1354)" and the "elbows (726, 139, 1445, 729)"
DEFAULT_OUTPUT_NODES = [1354, 726, 139, 1445, 729]

from examples import Problem


def generate_koopman_data():
    """
    Prepare Koopman data in .mat format for MATLAB Koopman code (generate_koopman_model.m)
    """
    from scipy.io import savemat
    from sofacontrol.utils import load_data, qv2x
    from sofacontrol.measurement_models import linearModel

    pod_data_name = 'pod_snapshots'
    num_nodes = 1628
    ee_node = [1354]
    five_nodes = DEFAULT_OUTPUT_NODES
    three_nodes = [1354, 726, 139]
    three_nodes_row = [1354, 726, 1445]
    pod_data = load_data(join(path, '{}.pkl'.format(pod_data_name)))

    state = qv2x(q=pod_data['q'], v=pod_data['v'])
    names = ['ee_pos']
    measurement_models = [linearModel(nodes=ee_node, num_nodes=num_nodes, pos=True, vel=False)]

    for i, name in enumerate(names):
        mat_data_file = join(path, '{}.mat'.format(name))
        y = measurement_models[i].evaluate(x=state.T)
        matlab_data = dict()
        matlab_data['y'] = y.T
        matlab_data['u'] = np.asarray(pod_data['u'])
        matlab_data['t'] = np.atleast_2d(pod_data['dt'] * np.asarray(range(len(matlab_data['u']))))

        savemat(mat_data_file, matlab_data)


def run_koopman():
    """
    In problem_specification add:

    from examples.diamond import diamond_koopman
    problem = diamond_koopman.run_koopman

    then run:

    python3 launch_sofa.py
    """
    from robots import environments
    from examples.hardware.model import diamondRobot
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.baselines.koopman import koopman_utils
    from scipy.io import loadmat
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.utils import Polyhedron
    from sofacontrol.baselines.koopman import koopman
    koopman_data = loadmat(join(path, 'koopman_model.mat'))['py_data'][0, 0]
    raw_model = koopman_data['model']
    raw_params = koopman_data['params']
    model = koopman_utils.KoopmanModel(raw_model, raw_params)
    scaling = koopman_utils.KoopmanScaling(scale=model.scale)

    prob = Problem()
    # prob.Robot = environments.Diamond()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    cov_q = 0.1 * np.eye(3)
    prob.measurement_model = MeasurementModel(nodes=[1354], num_nodes=1628, pos=True, vel=False, S_q=cov_q)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Building A matrix
    Hz = np.zeros((2, 3))
    Hz[0, 0] = 1
    Hz[1, 1] = 1
    Hzfull = np.vstack([-Hz, Hz])

    b_z_lb = np.array([-17.5 - 5.5, -20 + 1.5, 0])[0:2]
    b_z_ub = np.array([17.5 - 5.5, 20 + 1.5, 0])[0:2]

    b_z = np.hstack([-b_z_lb, b_z_ub])
    Y = Polyhedron(A=Hzfull, b=b_z, with_reproject=True)

    prob.controller = koopman.KoopmanMPC(dyn_sys=model, dt=model.Ts, delay=2, rollout_horizon=1)

    prob.opt['sim_duration'] = 12.  # Simulation time, optional
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'koopman'

    return prob


def run_koopman_solver():
    """
    python3 diamond_koopman.py run_koopman_solver
    """
    from scipy.io import loadmat
    from sofacontrol.baselines.koopman import koopman_utils
    from sofacontrol.baselines.ros import runMPCSolverNode
    from sofacontrol.tpwl.tpwl_utils import Target
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.utils import HyperRectangle, Polyhedron

    koopman_data = loadmat(join(path, 'koopman_model.mat'))['py_data'][0, 0]
    raw_model = koopman_data['model']
    raw_params = koopman_data['params']
    model = koopman_utils.KoopmanModel(raw_model, raw_params)

    Ts = model.Ts
    cost_params = QuadraticCost()
    # Define target trajectory for optimization
    T = 10
    target = Target()
    target.t = np.linspace(0, T, 1000)
    th = np.linspace(0, 2 * np.pi, 1000)
    zf_target = np.zeros((1000, model.n))
    zf_target[:, 0] = -15. * np.sin(th)
    zf_target[:, 1] = 15. * np.sin(2 * th)
    zf_target[:, 2] -= 114

    scaling = koopman_utils.KoopmanScaling(scale=model.scale)
    zf_target_norm = scaling.scale_down(y=zf_target)

    target.z = zf_target_norm

    u_target = np.zeros(model.m)
    u_target_norm = scaling.scale_down(u=u_target).reshape(-1)
    target.u = u_target_norm
    cost_params.R = .00001 * np.eye(model.m)
    cost_params.Q = np.zeros((model.n, model.n))
    cost_params.Q[0, 0] = 100  # corresponding to x position of end effector
    cost_params.Q[1, 1] = 100  # corresponding to y position of end effector

    # Consider same "scaled" cost parameters as other models
    cost_params.R *= np.diag(scaling.u_factor[0])
    cost_params.Q *= np.diag(scaling.y_factor[0])

    # Constraints
    u_ub = 1500. * np.ones(model.m)
    u_lb = 200. * np.ones(model.m)
    u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
    u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
    U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)

    # Constraints - State constraints
    # Building A matrix
    # Hz = np.zeros((2, 3))
    # Hz[0, 0] = 1
    # Hz[1, 1] = 1
    # H = Hz @ model.H
    # H_full = np.vstack([-H, H])
    # b_z_lb = np.array([-17.5 - 5.5, -20 + 1.5, 0])
    # b_z_ub = np.array([17.5 - 5.5, 20 + 1.5, 0])
    # b_z_lb_norm = scaling.scale_down(y=b_z_lb).reshape(-1)[0:2]
    # b_z_ub_norm = scaling.scale_down(y=b_z_ub).reshape(-1)[0:2]
    #
    # b_z = np.hstack([-b_z_lb_norm, b_z_ub_norm])
    # X = Polyhedron(A=H_full, b=b_z)

    planning_horizon = 5

    # runMPCSolverNode(model=model, N=planning_horizon, cost_params=cost_params, target=target, dt=Ts, verbose=1,
    #                  warm_start=True, U=U, solver='GUROBI')

    runMPCSolverNode(model=model, N=planning_horizon, cost_params=cost_params, target=target, dt=Ts, verbose=2,
                     warm_start=True, U=U, solver='GUROBI')

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'generate_koopman_data':
        generate_koopman_data()
    elif sys.argv[1] == 'run_koopman_solver':
        run_koopman_solver()
    else:
        raise RuntimeError('Not a valid function argument')
