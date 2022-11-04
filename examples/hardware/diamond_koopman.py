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
from examples.hardware.model import diamondRobot

# Default nodes are the "end effector (1354)" and the "elbows (726, 139, 1445, 729)"
DEFAULT_OUTPUT_NODES = [1354]


def generate_koopman_data():
    """
    Prepare Koopman data in .mat format for MATLAB Koopman code (generate_koopman_model.m)
    """
    from scipy.io import savemat, loadmat
    from sofacontrol.utils import load_data, qv2x, save_data, vq2qv
    from sofacontrol.measurement_models import linearModel

    useVel = True
    # koopman_data_name = 'pod_snapshots'
    # koopman_data_name = 'koopman_train_data_full'
    koopman_data_name = 'koopman_train_data_full_extra'
    names = ['diamond_train_full_extra']

    # koopman_data_name = 'koopman_val_data'
    # names = ['diamond_val_full']

    num_nodes = 1628
    ee_node = [1354]
    koopman_data = load_data(join(path, '{}.pkl'.format(koopman_data_name)))

    state = qv2x(q=koopman_data['q'], v=koopman_data['v'])
    measurement_models = [linearModel(nodes=ee_node, num_nodes=num_nodes, pos=True, vel=useVel)]

    for i, name in enumerate(names):
        mat_data_file = join(path, '{}.mat'.format(name))
        y = measurement_models[i].evaluate(x=state.T)
        matlab_data = dict()
        if useVel:
            matlab_data['y'] = vq2qv(y.T)
        else:
            matlab_data['y'] = y.T
        matlab_data['u'] = np.asarray(koopman_data['u'])
        matlab_data['t'] = np.atleast_2d(koopman_data['dt'] * np.asarray(range(len(matlab_data['u']))))

        savemat(mat_data_file, matlab_data)

        # # Also save dataset as pkl file
        # koopman_dataset = dict()
        # koopman_dataset['x'] = y.T
        # koopman_dataset['u'] = np.asarray(koopman_data['u'])
        # koopman_dataset['t'] = np.atleast_2d(koopman_data['dt'] * np.asarray(range(len(matlab_data['u']))))
        #
        # save_data(join(path, '{}.pkl'.format('koopman_diamond_dataset')), koopman_dataset)
        #
        # # Load old mat and save to pkl
        # koopman_old_model = loadmat(join(path, 'koopman_model.mat'))['py_data'][0, 0]
        # save_data(join(path, '{}.pkl'.format('koopman_diamond_old_model')), koopman_old_model['model'])


def run_koopman():
    """
    In problem_specification add:

    from examples.diamond import diamond_koopman
    problem = diamond_koopman.run_koopman

    then run:

    python3 launch_sofa.py
    """
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.baselines.koopman import koopman_utils, koopman
    from scipy.io import loadmat
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.utils import Polyhedron

    useVel = False
    koopman_data = loadmat(join(path, 'koopman_model.mat'))['py_data'][0, 0]
    raw_model = koopman_data['model']
    raw_params = koopman_data['params']
    model = koopman_utils.KoopmanModel(raw_model, raw_params, DMD=False)
    scaling = koopman_utils.KoopmanScaling(scale=model.scale)

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model vel=useVel, qv=useVel)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    cov_q = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    if useVel:
        cov_v = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    else:
        cov_v = None

    # if velocity is used, then return in qv format
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v,
                                              vel=useVel, qv=useVel)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # This is a hack to make constraint "feasible" - esp during long horizons
    # Activate this if optimization infeasible due to constraint
    # Building A matrix
    # Hz = np.zeros((1, model.n))
    # Hz[0, 1] = 1
    # b_z = np.array([5])
    # Y = Polyhedron(A=Hz, b=b_z, with_reproject=True)
    Y = None

    prob.controller = koopman.KoopmanMPC(dyn_sys=model, dt=model.Ts, delay=3, rollout_horizon=1, Y=Y)

    prob.opt['sim_duration'] = 13.  # Simulation time, optional
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
    model = koopman_utils.KoopmanModel(raw_model, raw_params, DMD=False)
    scaling = koopman_utils.KoopmanScaling(scale=model.scale)

    cost = QuadraticCost()
    target = Target()

    #############################################
    # Problem 1, Figure 8 with constraints
    #############################################
    # M = 3
    # T = 10
    # N = 500
    # t = np.linspace(0, M*T, M*N)
    # th = np.linspace(0, M * 2 * np.pi, M*N)
    # zf_target = np.zeros((M*N, model.n))
    #
    # zf_target[:, 0] = -15. * np.sin(th) - 7.1
    # zf_target[:, 1] = 15. * np.sin(2 * th)

    # zf_target[:, 0] = -25. * np.sin(th) + 13.
    # zf_target[:, 1] = 25. * np.sin(2 * th) + 20.

    # zf_target[:, 0] = -35. * np.sin(th) - 7.1
    # zf_target[:, 1] = 35. * np.sin(2 * th)

    # zf_target[:, 0] = -5. * np.sin(th) - 7.1
    # zf_target[:, 1] = 5. * np.sin(2 * th)
    #
    # # Offset with constraints
    # zf_target[:, 0] = -15. * np.sin(th)
    # zf_target[:, 1] = 15. * np.sin(2 * th)
    #
    # zf_target[:, 0] = -15. * np.sin(8 * th) - 7.1
    # zf_target[:, 1] = 15. * np.sin(16 * th)

    # Cost
    # cost.R = .00001 * np.eye(model.m)
    # cost.Q = np.zeros((model.n, model.n))
    # cost.Q[0, 0] = 100  # corresponding to x position of end effector
    # cost.Q[1, 1] = 100  # corresponding to y position of end effector
    # cost.Q[2, 2] = 0.0  # corresponding to z position of end effector
    #
    # # Control constraints
    # u_ub = 2500. * np.ones(model.m)
    # u_lb = 200. * np.ones(model.m)
    # u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
    # u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
    # U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)

    # State constraints
    # Hz = np.zeros((1, model.n))
    # Hz[0, 1] = 1
    # H = Hz @ model.H
    # b_z = np.array([5])
    # b_z_ub_norm = scaling.scale_down(y=b_z).reshape(-1)[1]
    # X = Polyhedron(A=H, b=b_z_ub_norm)

    # X = None

    #####################################################
    # Problem 2, Circle on side (2pi/T = frequency rad/s)
    #####################################################
    # Multiply 'th' in sine terms to factor rad/s frequency
    M = 3
    T = 5
    N = 1000
    t = np.linspace(0, M * T, M * N)
    th = np.linspace(0, M * 2 * np.pi, M * N)
    x_target = np.zeros(M * N)

    r = 15
    phi = 17
    y_target = r * np.sin(phi * T / (2 * np.pi) * th)
    z_target = r - r * np.cos(phi * T / (2 * np.pi) * th) + 107.0

    # r = 15
    # y_target = r * np.sin(th)
    # z_target = r - r * np.cos(th) + 107.0

    # TODO: Modify number of observables
    zf_target = np.zeros((M * N, model.n))
    zf_target[:, 0] = x_target
    zf_target[:, 1] = y_target
    zf_target[:, 2] = z_target

    # Cost
    cost.R = .00001 * np.eye(model.m)
    cost.Q = np.zeros((model.n, model.n))
    cost.Q[0, 0] = 0.0  # corresponding to x position of end effector
    cost.Q[1, 1] = 100.0  # corresponding to y position of end effector
    cost.Q[2, 2] = 100.0  # corresponding to z position of end effector

    # Constraints
    u_ub = 2500. * np.ones(model.m)
    u_lb = 200. * np.ones(model.m)
    u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
    u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
    U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)
    X = None

    # Define target trajectory for optimization
    target.t = t
    target.z = scaling.scale_down(y=zf_target)
    target.u = scaling.scale_down(u=np.zeros(model.m)).reshape(-1)

    # Consider same "scaled" cost parameters as other models
    cost.R *= np.diag(scaling.u_factor[0])
    cost.Q *= np.diag(scaling.y_factor[0])

    N = 3

    runMPCSolverNode(model=model, N=N, cost_params=cost, X=X, target=target, dt=model.Ts, verbose=1,
                     warm_start=False, U=U, solver='GUROBI')


def run_MPC_OL():
    """
     In problem_specification add:

     from examples.hardware import diamond
     problem = diamond.run_MPC_OL

     then run:

     python3 launch_sofa.py
     """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.baselines.koopman import koopman_utils
    from scipy.io import loadmat
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.open_loop_sequences import DiamondRobotSequences
    from sofacontrol.utils import QuadraticCost, HyperRectangle, load_data, qv2x
    from sofacontrol.tpwl.tpwl_utils import Target
    from sofacontrol.baselines.ros import runMPCSolver
    from sofacontrol.baselines.koopman.koopman_utils import KoopmanData

    t0 = 3.0
    dt = 0.05
    useVel = False
    koopman_data = loadmat(join(path, 'koopman_model_OL.mat'))['py_data'][0, 0]
    raw_model = koopman_data['model']
    raw_params = koopman_data['params']
    model = koopman_utils.KoopmanModel(raw_model, raw_params, DMD=False)
    data = KoopmanData(model.scale, model.delays)

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController
    Sequences = DiamondRobotSequences(t0=t0, dt=dt)

    # Specify a measurement and output model
    cov_q = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    if useVel:
        cov_v = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    else:
        cov_v = None
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v,
                                              vel=useVel)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354], vel=useVel, qv=True)
    scaling = koopman_utils.KoopmanScaling(scale=model.scale)

    # Load initial state and transform to lifted state
    rest_file = join(path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    qv_equilibrium = np.array(rest_data['rest'])

    # Setup initial condition
    x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
    x0 = prob.output_model.evaluate(x_eq, qv=True)

    cost = QuadraticCost()
    target = Target()
    #############################################
    # Problem 1, Figure 8 with constraints
    #############################################
    # M = 3
    # T = 10
    # N = 500
    # t = np.linspace(0, M*T, M*N)
    # th = np.linspace(0, M * 2 * np.pi, M*N)
    # zf_target = np.zeros((M*N, model.n))
    #
    # zf_target[:, 0] = -15. * np.sin(2 * th) - 7.1
    # zf_target[:, 1] = 15. * np.sin(4 * th)

    # zf_target[:, 0] = -25. * np.sin(th) + 13.
    # zf_target[:, 1] = 25. * np.sin(2 * th) + 20.
    #
    # zf_target[:, 0] = -40. * np.sin(th) - 7.1
    # zf_target[:, 1] = 40. * np.sin(2 * th)
    #
    # zf_target[:, 0] = -5. * np.sin(th) - 7.1
    # zf_target[:, 1] = 5. * np.sin(2 * th)
    #
    # # Offset with constraints
    # zf_target[:, 0] = -15. * np.sin(th)
    # zf_target[:, 1] = 15. * np.sin(2 * th)
    #
    # zf_target[:, 0] = -15. * np.sin(8 * th) - 7.1
    # zf_target[:, 1] = 15. * np.sin(16 * th)

    # Cost
    # cost.R = .00001 * np.eye(model.m)
    # cost.Q = np.zeros((model.n, model.n))
    # cost.Q[0, 0] = 100  # corresponding to x position of end effector
    # cost.Q[1, 1] = 100  # corresponding to y position of end effector
    # cost.Q[2, 2] = 0.0  # corresponding to z position of end effector

    # # Control constraints
    # u_ub = 4000. * np.ones(model.m)
    # u_lb = 200. * np.ones(model.m)
    # u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
    # u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
    # U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)

    # State constraints
    # Hz = np.zeros((1, 3))
    # Hz[0, 1] = 1
    # H = Hz @ model.H
    # b_z = np.array([5])
    # b_z_ub_norm = scaling.scale_down(y=b_z).reshape(-1)[1]
    # X = Polyhedron(A=H, b=b_z_ub_norm)

    # X = None

    ##############################################
    # Problem 2, Circle on side
    ##############################################
    M = 3
    T = 5
    N = 1000
    t = np.linspace(0, M * T, M * N)
    th = np.linspace(0, M * 2 * np.pi, M * N)
    x_target = np.zeros(M * N)

    r = 10
    y_target = r * np.sin(th)
    z_target = r - r * np.cos(th) + 107.0

    # r = 20
    # y_target = r * np.sin(17 * th)
    # z_target = r - r * np.cos(17 * th) + 107.0

    zf_target = np.zeros((M * N, 3))
    zf_target[:, 0] = x_target
    zf_target[:, 1] = y_target
    zf_target[:, 2] = z_target

    # Cost
    cost.R = .00001 * np.eye(model.m)
    cost.Q = np.zeros((3, 3))
    cost.Q[0, 0] = 50.0  # corresponding to x position of end effector
    cost.Q[1, 1] = 100.0  # corresponding to y position of end effector
    cost.Q[2, 2] = 100.0  # corresponding to z position of end effector

    # Constraints
    u_ub = 1500. * np.ones(model.m)
    u_lb = 200. * np.ones(model.m)
    u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
    u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
    U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)
    X = None

    # Define target trajectory for optimization
    target.t = t
    target.z = scaling.scale_down(y=zf_target)
    target.u = scaling.scale_down(u=np.zeros(model.m)).reshape(-1)

    # Consider same "scaled" cost parameters as other models
    cost.R *= np.diag(scaling.u_factor[0])
    cost.Q *= np.diag(scaling.y_factor[0])

    # If there are delays, replicate initial conditions
    for j in range(model.delays + 1):
        data.add_measurement(x0, u_lb)

    zeta0 = data.get_zeta()
    x0_lifted = np.asarray(model.lift_data(*zeta0))

    N = 200

    # Using osqp instead of Gurobi because Gurobi had some numerical issues
    xopt, uopt, zopt, topt = runMPCSolver(model=model, N=N, cost_params=cost, x0=x0_lifted, target=target, dt=model.Ts,
                                          verbose=1,
                                          warm_start=False, U=U, X=X, solver='OSQP')

    uopt_rescaled = data.scaling.scale_up(u=uopt)

    u, save, t = Sequences.augment_input_with_base(uopt_rescaled.T, save_data=True)
    prob.controller = OpenLoop(u.shape[0], t, u, save, dt=dt)

    # prob.snapshots = SnapshotData(save_dynamics=False)

    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'mpc_OL_Koopman'
    return prob


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'generate_koopman_data':
        generate_koopman_data()
    elif sys.argv[1] == 'run_koopman_solver':
        run_koopman_solver()
    else:
        raise RuntimeError('Not a valid function argument')
