import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.hardware.model import diamondRobot
from sofacontrol.open_loop_sequences import DiamondRobotSequences

#DEFAULT_OUTPUT_NODES = [1354, 726, 139, 1445, 729]
DEFAULT_OUTPUT_NODES = [1354]

TIP_NODE = 1354

def module_test_continuous():
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.SSM import ssm
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.SSM.controllers import scp
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.utils import load_data, qv2x, x2qv
    from sofacontrol.measurement_models import linearModel

    # Load SSM Models
    # Continuous time models
    from examples.hardware.SSMmodels.diamond_softrobot_Amat_O3_cont import diamond_softrobot_Amat_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_Bmat_O3_cont import diamond_softrobot_Bmat_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_C_O3_cont import diamond_softrobot_C_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_W_O3_cont import diamond_softrobot_W_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_f_reduced_O3_cont import diamond_softrobot_f_reduced_O3_cont

    dt = 0.1
    prob = Problem()
    prob.Robot = diamondRobot(dt=dt)
    prob.ControllerClass = ClosedLoopController

    # 1) Setup model
    # TODO: 1) Grab equilibrium point and 2) extract maps into dictionary (store as d for now)
    # TODO: Need to extract rest.pkl with velocity
    rest_file = join(path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    qv_equilibrium = np.array(rest_data['rest'])

    x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
    outputModel = linearModel([TIP_NODE], 1628)
    z_eq_point = outputModel.evaluate(x_eq, qv=True)

    # Setup the (discrete) maps: TODO: Automate generation of this via symbolic and differentiation
    # TODO: Setup continuous maps as well
    maps = dict()
    maps['A'] = diamond_softrobot_Amat_O3_cont
    maps['B'] = diamond_softrobot_Bmat_O3_cont
    maps['C'] = diamond_softrobot_C_O3_cont
    maps['W'] = diamond_softrobot_W_O3_cont
    maps['f_nl'] = diamond_softrobot_f_reduced_O3_cont

    n, m, o = 6, 4, 6
    model = ssm.SSMDynamics(z_eq_point, maps, n, m, o, discrete=False, discr_method='be')

    # 2) Test various functions here (compute_RO_state, get_ref_point, get_jacobians (check assertion and subfunctions)
    # update_state, update_dynamics). Test functions by perturbing z_ref slightly
    p0 = np.ones((n, 1))
    z0 = np.ones((o, 1))
    u0 = np.ones((m, 1))
    p_test = model.compute_RO_state(z0)
    z_ref = model.get_ref_point()
    A_d, B_d, d_d = model.get_jacobians(p0, u=u0, dt=dt)
    assert(p_test == maps['W'](z0 - z_eq_point)).all(), 'Something wrong with observed to reduced state map'

    x_next = model.update_state(p0, u0, dt)
    x_next_true = p0 + dt*maps['f_nl'](p0, u0)
    # assert (np.round(x_next,4) == np.round(np.squeeze(x_next_true),4)).all(), 'Update state result does not match discrete transition map'


    # 3) Reuse inputs from TPWL model. Test rollout function and compare figure of rollout figure 8
    # with true response of system (need to extract z from simulation and load here). Plot comparison

    # Load files to run tests
    pathToTraj = path + '/checkModel/'
    z_true_file = join(pathToTraj, 'z_big.csv')
    z_true = np.genfromtxt(z_true_file, delimiter=',')
    zq_true, zv_true = x2qv(z_true)
    u_true_file = join(pathToTraj, 'u_big.csv')
    u_true = np.genfromtxt(u_true_file, delimiter=',')

    T = 10.01
    N = int(T / dt)
    t_original = np.linspace(0, T, int(T/0.01)+1)
    t_interp = np.linspace(0, T, N+1)
    u_interp = interp1d(t_original, u_true, axis=0)(t_interp)
    p0 = np.zeros((n,))
    p_traj, z_traj = model.rollout(p0, u_interp, dt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(z_traj[:, 0], z_traj[:, 1], z_traj[:, 2], label='Predicted Trajectory')
    ax.plot3D(zq_true[:, 0], zq_true[:, 1], zq_true[:, 2], label='Actual Trajectory')
    plt.legend()

    z_true_qv = interp1d(t_original, np.hstack((zq_true, zv_true)), axis=0)(t_interp)
    err_SSM = z_true_qv - z_traj[:-1]
    SSM_RMSE = np.linalg.norm(np.linalg.norm(err_SSM, axis=1))**2 / err_SSM.shape[0]
    print('------ Mean Squared Errors (MSEs)----------')
    print('Ours (SSM): {}'.format(SSM_RMSE))
    plt.show()

    # TODO: Add function here that plots the individual trajectories
    print('Testing rollout functions')

def module_test():
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.SSM import ssm
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.SSM.controllers import scp
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.utils import load_data, qv2x, x2qv
    from sofacontrol.measurement_models import linearModel

    # Load SSM Models
    # Discrete time models (dt = 0.001)
    from examples.hardware.SSMmodels.diamond_softrobot_Amat_O3 import diamond_softrobot_Amat_O3
    from examples.hardware.SSMmodels.diamond_softrobot_Bmat_O3 import diamond_softrobot_Bmat_O3
    from examples.hardware.SSMmodels.diamond_softrobot_C_O3 import diamond_softrobot_C_O3
    from examples.hardware.SSMmodels.diamond_softrobot_W_O3 import diamond_softrobot_W_O3
    from examples.hardware.SSMmodels.diamond_softrobot_f_reduced_O3 import diamond_softrobot_f_reduced_O3

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    # 1) Setup model
    # TODO: 1) Grab equilibrium point and 2) extract maps into dictionary (store as d for now)
    # TODO: Need to extract rest.pkl with velocity
    rest_file = join(path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    qv_equilibrium = np.array(rest_data['rest'])

    x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
    outputModel = linearModel([TIP_NODE], 1628)
    z_eq_point = outputModel.evaluate(x_eq, qv=True)

    # Setup the (discrete) maps: TODO: Automate generation of this via symbolic and differentiation
    # TODO: Setup continuous maps as well
    maps = dict()
    maps['A_d'] = diamond_softrobot_Amat_O3
    maps['B_d'] = diamond_softrobot_Bmat_O3
    maps['C'] = diamond_softrobot_C_O3
    maps['W'] = diamond_softrobot_W_O3
    maps['f_nl_d'] = diamond_softrobot_f_reduced_O3

    n, m, o = 6, 4, 6
    model = ssm.SSMDynamics(z_eq_point, maps, n, m, o, discrete=True)

    # 2) Test various functions here (compute_RO_state, get_ref_point, get_jacobians (check assertion and subfunctions)
    # update_state, update_dynamics). Test functions by perturbing z_ref slightly
    p0 = np.ones((n, 1))
    z0 = np.ones((o, 1))
    u0 = np.ones((m, 1))
    p_test = model.compute_RO_state(z0)
    z_ref = model.get_ref_point()
    A_d, B_d, d_d = model.get_jacobians(p0, u=u0)
    assert (A_d == maps['A_d'](p0)).all() and (B_d == maps['B_d'](u0)).all(), 'Maps do not agree. Something wrong with Jacobian function'
    assert(p_test == maps['W'](z0 - z_eq_point)).all(), 'Something wrong with observed to reduced state map'

    dt = 0.001
    x_next = model.update_state(p0, u0, dt)
    x_next_true = maps['f_nl_d'](p0, u0)
    assert (x_next == np.squeeze(x_next_true)).all(), 'Update state result does not match discrete transition map'


    # 3) Reuse inputs from TPWL model. Test rollout function and compare figure of rollout figure 8
    # with true response of system (need to extract z from simulation and load here). Plot comparison

    # Load files to run tests
    pathToTraj = path + '/checkModel/'
    z_true_file = join(pathToTraj, 'z_big.csv')
    z_true = np.genfromtxt(z_true_file, delimiter=',')
    zq_true, zv_true = x2qv(z_true)
    u_true_file = join(pathToTraj, 'u_big.csv')
    u_true = np.genfromtxt(u_true_file, delimiter=',')

    T = 10.01
    N = int(T / dt)
    t_original = np.linspace(0, T, int(T/0.01)+1)
    t_interp = np.linspace(0, T, N+1)
    u_interp = interp1d(t_original, u_true, axis=0)(t_interp)
    p0 = np.zeros((n,))
    p_traj, z_traj = model.rollout(p0, u_interp, dt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(z_traj[:, 0], z_traj[:, 1], z_traj[:, 2], label='Predicted Trajectory')
    ax.plot3D(zq_true[:, 0], zq_true[:, 1], zq_true[:, 2], label='Actual Trajectory')
    plt.legend()

    z_true_qv = interp1d(t_original, np.hstack((zq_true, zv_true)), axis=0)(t_interp)
    err_SSM = z_true_qv - z_traj[:-1]
    SSM_RMSE = np.linalg.norm(np.linalg.norm(err_SSM, axis=1))**2 / err_SSM.shape[0]
    print('------ Mean Squared Errors (MSEs)----------')
    print('Ours (SSM): {}'.format(SSM_RMSE))
    plt.show()

    # TODO: Add function here that plots the individual trajectories
    print('Testing rollout functions')



# 4) Test controller functions. Let's try iLQR first
def run_ilqr():
    from examples.hardware.model import diamondRobot
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.SSM.controllers import ilqr
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.SSM import ssm
    from sofacontrol.utils import load_data, qv2x, x2qv

    # Load SSM Models
    from examples.hardware.SSMmodels.diamond_softrobot_Amat_O3_cont import diamond_softrobot_Amat_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_Bmat_O3_cont import diamond_softrobot_Bmat_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_C_O3_cont import diamond_softrobot_C_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_W_O3_cont import diamond_softrobot_W_O3_cont
    from examples.hardware.SSMmodels.diamond_softrobot_f_reduced_O3_cont import diamond_softrobot_f_reduced_O3_cont

    n, m, o = 6, 4, 6
    dt = 0.01
    prob = Problem()
    prob.Robot = diamondRobot(dt=dt)
    prob.ControllerClass = ClosedLoopController

    # Specify measurement model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Specify output model
    output_model = linearModel(nodes=[1354], num_nodes=1628)

    # Setup model
    rest_file = join(path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    qv_equilibrium = np.array(rest_data['rest'])

    # Necessary since rest position above is in full state space
    # I extracted the equilibrium in qv so need to switch to vq format.
    # Then I extract both configuration and velocity and return in qv format
    x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
    outputModel = linearModel([TIP_NODE], 1628)
    z_eq_point = outputModel.evaluate(x_eq, qv=True)

    maps = dict()
    maps['A'] = diamond_softrobot_Amat_O3_cont
    maps['B'] = diamond_softrobot_Bmat_O3_cont
    maps['C'] = diamond_softrobot_C_O3_cont
    maps['W'] = diamond_softrobot_W_O3_cont
    maps['f_nl'] = diamond_softrobot_f_reduced_O3_cont

    model = ssm.SSMDynamics(z_eq_point, maps, n, m, o, discrete=False, discr_method='be')

    # Define cost functions and trajectory
    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[0, 0] = 100  # corresponding to x position of end effector
    Qz[1, 1] = 100  # corresponding to y position of end effector
    Qz[2, 2] = 0.0  # corresponding to z position of end effector
    cost.Q = Qz
    cost.Qf = np.zeros_like(Qz)
    cost.R = .00001 * np.eye(model.input_dim)

    M = 3
    T = 10
    N = 1000
    t = np.linspace(0, M * T, M * N)
    th = np.linspace(0, M * 2 * np.pi, M * N)
    zf_target = np.zeros((M * N, model.output_dim))
    zf_target[:, 0] = -15. * np.sin(th)
    zf_target[:, 1] = 15. * np.sin(2 * th)
    z = model.zfyf_to_zy(zf=zf_target)

    # Define controller (wait 3 seconds of simulation time to start)
    from types import SimpleNamespace
    target = SimpleNamespace(z=z, Hf=output_model.C, t=t)

    dt = 0.1
    prob.controller = ilqr(model, cost, target, dt, observer=None, delay=3)

    # Saving paths
    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'ilqr_ssm'

    return prob


def run_scp():
    """
     In problem_specification add:

     from examples.hardware import diamond
     problem = diamond.run_scp

     then run:

     python3 launch_sofa.py
     """
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.tpwl.controllers import scp
    from sofacontrol.tpwl.observer import DiscreteEKFObserver
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import QuadraticCost

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Load and configure the TPWL model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=prob.output_model.C,
                         Cf=prob.measurement_model.C)

    dt = 0.01
    model.pre_discretize(dt=dt)

    # Set up an EKF observer
    dt_char = model.get_characteristic_dx(dt)
    W = np.diag(dt_char)
    V = 0.0 * np.eye(model.get_meas_dim())
    EKF = DiscreteEKFObserver(model, W=W, V=V)

    ##############################################
    # Problem 1, Figure 8 with constraints
    ##############################################
    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector
    cost.Q = model.H.T @ Qz @ model.H
    cost.R = .003 * np.eye(model.input_dim)

    ##############################################
    # Problem 2, Circle on side
    ##############################################
    # cost = QuadraticCost()
    # Qz = np.zeros((model.output_dim, model.output_dim))
    # Qz[3, 3] = 0.0  # corresponding to x position of end effector
    # Qz[4, 4] = 100.0  # corresponding to y position of end effector
    # Qz[5, 5] = 100.0  # corresponding to z position of end effector
    # cost.Q = model.H.T @ Qz @ model.H
    # cost.R = .003 * np.eye(model.input_dim)

    # Define controller (wait 3 seconds of simulation time to start)
    prob.controller = scp(model, cost, dt, N_replan=30, observer=EKF, delay=3)

    # Saving paths
    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'scp'

    return prob


def run_gusto_solver():
    """
    python3 diamond.py run_gusto_solver
    """
    from sofacontrol.scp.models.tpwl import TPWLGuSTO
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.scp.ros import runGuSTOSolverNode
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import HyperRectangle, Polyhedron

    output_model = linearModel(nodes=[1354], num_nodes=1628)

    # Load and configure the TPWL model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=output_model.C)

    #############################################
    # Problem 1, Figure 8 with constraints
    #############################################
    M = 3
    T = 10
    N = 500
    t = np.linspace(0, M * T, M * N)
    th = np.linspace(0, M * 2 * np.pi, M * N)
    zf_target = np.zeros((M * N, model.output_dim))
    zf_target[:, 3] = -15. * np.sin(th)
    zf_target[:, 4] = 15. * np.sin(2 * th)
    z = model.zfyf_to_zy(zf=zf_target)

    # Cost
    R = .00001 * np.eye(model.input_dim)
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector

    # Control constraints
    low = 200.0
    high = 1500.0
    U = HyperRectangle([high, high, high, high], [low, low, low, low])

    # State constraints
    # Hz = np.zeros((1, 6))
    # Hz[0, 4] = 1
    # H = Hz @ model.H
    # b_z = np.array([5])
    # X = Polyhedron(A=H, b=b_z - Hz @ model.z_ref)

    # No constraints for now
    X = None
    ##############################################
    # Problem 2, Circle on side
    ##############################################
    # M = 3
    # T = 5
    # N = 1000
    # r = 10
    # t = np.linspace(0, M*T, M*N)
    # th = np.linspace(0, M*2*np.pi, M*N)
    # x_target = np.zeros(M*N)
    # y_target = r * np.sin(th)
    # z_target = r - r * np.cos(th) + 107.0
    # zf_target = np.zeros((M*N, 6))
    # zf_target[:, 3] = x_target
    # zf_target[:, 4] = y_target
    # zf_target[:, 5] = z_target
    # z = model.zfyf_to_zy(zf=zf_target)

    # # Cost
    # R = .00001 * np.eye(4)
    # Qz = np.zeros((6, 6))
    # Qz[3, 3] = 0.0  # corresponding to x position of end effector
    # Qz[4, 4] = 100.0  # corresponding to y position of end effector
    # Qz[5, 5] = 100.0  # corresponding to z position of end effector

    # # Constraints
    # low = 200.0
    # high = 1500.0
    # U = HyperRectangle([high, high, high, high], [low, low, low, low])
    # X = None

    # Define initial condition to be x_ref for initial solve
    x0 = model.rom.compute_RO_state(xf=model.rom.x_ref)

    # Define GuSTO model
    dt = 0.1
    N = 5
    gusto_model = TPWLGuSTO(model)
    gusto_model.pre_discretize(dt)
    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
                       verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI',
                       max_gusto_iters=5)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'module_test':
        module_test()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    else:
        raise RuntimeError('Not a valid function argument')