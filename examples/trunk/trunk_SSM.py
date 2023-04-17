import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.trunk.model import trunkRobot
from sofacontrol.open_loop_sequences import TrunkRobotSequences

# Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51
N_NODES = 709


def module_test_continuous():
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.SSM import ssm
    from sofacontrol.utils import load_data, qv2x, x2qv
    from sofacontrol.measurement_models import linearModel, OutputModel
    from scipy.io import loadmat

    dt = 0.01

    # 1) Setup model: Grab equilibrium point (x then z)
    rest_file = join(path, 'rest.pkl')
    rest_data = load_data(rest_file)
    q_equilibrium = np.array(rest_data['q'][0])

    x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))
    #outputModel = linearModel([TIP_NODE], 1628)
    outputModel = linearModel([TIP_NODE], N_NODES, vel=False)

    # TODO: This evaluation is a mess - qv option fails terribly if observation is only position (i.e., 3 dim)
    z_eq_point = outputModel.evaluate(x_eq, qv=False)

    # load SSM model as computed using SSMLearn (.mat file)
    pathToModel = path + '/SSMmodels/'
    #SSM_data = loadmat(join(pathToModel, 'SSM_model.mat'))['py_data'][0, 0]
    # SSM_data = loadmat(join(pathToModel, 'SSM_model_delay.mat'))['py_data'][0, 0]
    SSM_data = loadmat(join(pathToModel, 'SSM_model_simulation.mat'))['py_data'][0, 0]
    raw_model = SSM_data['model']
    raw_params = SSM_data['params']

    outputSSMModel = OutputModel(6, 3)
    model = ssm.SSMDynamics(z_eq_point, discrete=False, discr_method='be',
                            model=raw_model, params=raw_params, C=None)
    n = raw_params['state_dim'][0, 0][0, 0]

    # Reuse inputs from TPWL model. Test rollout function and compare figure of rollout figure 8
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

    xbar_traj = np.zeros(np.shape(p_traj))
    zbar_traj = np.zeros(np.shape(z_traj))
    for idx in range(np.shape(xbar_traj)[0]):
        xbar_traj[idx, :] = model.V_map(model.zfyf_to_zy(z_traj)[idx])
        zbar_traj[idx, :] = model.x_to_zfyf(xbar_traj[idx])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(z_traj[:, 3], z_traj[:, 4], z_traj[:, 5], label='Predicted Trajectory')
    ax.plot3D(zq_true[:, 0], zq_true[:, 1], zq_true[:, 2], label='Actual Trajectory')
    ax.plot3D(zbar_traj[:, 3], zbar_traj[:, 4], zbar_traj[:, 5], label='Manifold Projected Trajectory')
    plt.legend()
    plt.title('SSM Open Loop Trajectory')

    z_true_qv = interp1d(t_original, np.hstack((zq_true, zv_true)), axis=0)(t_interp)
    #err_SSM = z_true_qv - z_traj[:-1]
    err_SSM = z_true_qv[:, 0:3] - z_traj[:-1, 3:]
    SSM_RMSE = np.linalg.norm(np.linalg.norm(err_SSM, axis=1))**2 / err_SSM.shape[0]
    print('------ Mean Squared Errors (MSEs)----------')
    print('Ours (SSM): {}'.format(SSM_RMSE))
    plt.show()

    print('Testing rollout functions')

def module_test():
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.SSM import ssm
    from sofacontrol.utils import load_data, qv2x, x2qv
    from sofacontrol.measurement_models import linearModel
    from scipy.io import loadmat

    # 1) Setup model
    rest_file = join(path, 'rest.pkl')
    rest_data = load_data(rest_file)
    q_equilibrium = np.array(rest_data['q'][0])

    x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

    outputModel = linearModel([TIP_NODE], N_NODES)
    z_eq_point = outputModel.evaluate(x_eq, qv=True)

    pathToModel = path + '/SSMmodels/'
    SSM_data = loadmat(join(pathToModel, 'SSM_model.mat'))['py_data'][0, 0]
    raw_model = SSM_data['model']
    raw_params = SSM_data['params']

    n = raw_params['state_dim'][0, 0][0, 0]
    model = ssm.SSMDynamics(z_eq_point, discrete=True, discr_method='be',
                            model=raw_model, params=raw_params)
    dt = 0.01

    # Reuse inputs from TPWL model. Test rollout function and compare figure of rollout figure 8
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
    from sofacontrol.SSM import ssm
    from sofacontrol.SSM.observer import SSMObserver, DiscreteEKFObserver
    from sofacontrol.SSM.controllers import scp
    import pickle

    prob = Problem()
    prob.Robot = trunkRobot()
    prob.ControllerClass = ClosedLoopController

    useTimeDelay = True

    # Load equilibrium point
    rest_file = join(path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    q_equilibrium = np.array(rest_data['q'][0])

    # Setup equilibrium point (no time delay and observed position and velocity of tip)
    x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

    # Set directory for SSM Models
    pathToModel = path + '/SSMmodels/'

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3)
    cov_v = 0.0 * np.eye(3) # * len(DEFAULT_OUTPUT_NODES))
    prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE])

    # load SSM model
    with open(join(pathToModel, 'SSM_model.pkl'), 'rb') as f:
        SSM_data = pickle.load(f)

    raw_model = SSM_data['model']
    raw_params = SSM_data['params']

    if raw_params['delay_embedding']:
        outputModel = linearModel([TIP_NODE], N_NODES, vel=False)
        z_eq_point = outputModel.evaluate(x_eq, qv=False)
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=False, S_q=cov_q)
        outputSSMModel = OutputModel(15, 3) # TODO: modify this
        # outputSSMModel = OutputModel(6, 3) # TODO: modify this
        Cout = outputSSMModel.C
    else:
        outputModel = linearModel([TIP_NODE], N_NODES)
        z_eq_point = outputModel.evaluate(x_eq, qv=True)
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=True, S_q=cov_q, S_v=cov_v)
        Cout = None

    model = ssm.SSMDynamics(z_eq_point, discrete=False, discr_method='be',
                            model=raw_model, params=raw_params, C=Cout)

    # This dt for when to recalculate control
    dt = 0.02

    # Pure SSM Manifold Observer
    observer = SSMObserver(model)

    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[0, 0] = 100.  # corresponding to x position of end effector
    Qz[1, 1] = 100.  # corresponding to y position of end effector
    Qz[2, 2] = 0. # 100.  # corresponding to z position of end effector
    cost.Q = model.H.T @ Qz @ model.H
    cost.R = 0.001 * np.eye(model.input_dim)

    # Define controller (wait 3 seconds of simulation time to start)
    prob.controller = scp(model, cost, dt, N_replan=2, delay=1, feedback=False, EKF=observer)

    # Saving paths
    prob.opt['sim_duration'] = 11.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'ssmr'

    return prob


def run_gusto_solver():
    """
    python3 trunk_SSM.py run_gusto_solver
    """
    from sofacontrol.scp.models.ssm import SSMGuSTO
    from sofacontrol.measurement_models import linearModel, OutputModel
    from sofacontrol.scp.ros import runGuSTOSolverNode
    from sofacontrol.utils import HyperRectangle, load_data, qv2x, Polyhedron
    from sofacontrol.SSM import ssm
    import pickle

    useTimeDelay = True

    # Load equilibrium point
    rest_file = join(path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    q_equilibrium = np.array(rest_data['q'][0])

    # Setup equilibrium point (no time delay and observed position and velocity of tip)
    x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

    # Set directory for SSM Models
    pathToModel = path + '/SSMmodels/'

    # load SSM model
    with open(join(pathToModel, 'SSM_model.pkl'), 'rb') as f:
        SSM_data = pickle.load(f)

    raw_model = SSM_data['model']
    raw_params = SSM_data['params']

    if raw_params['delay_embedding']:
        outputModel = linearModel([TIP_NODE], N_NODES, vel=False)
        z_eq_point = outputModel.evaluate(x_eq, qv=False)
        outputSSMModel = OutputModel(15, 3) # TODO: modify this
        # outputSSMModel = OutputModel(6, 3) # TODO: modify this
        Cout = outputSSMModel.C
    else:
        outputModel = linearModel([TIP_NODE], N_NODES)
        z_eq_point = outputModel.evaluate(x_eq, qv=True)
        Cout = None

    model = ssm.SSMDynamics(z_eq_point, discrete=False, discr_method='be',
                            model=raw_model, params=raw_params, C=Cout)

    # Define initial condition to be x_ref for initial solve
    x0 = np.zeros(model.state_dim)
    
    # Define target trajectory for optimization
    # === figure8 ===
    # M = 1
    # T = 5
    # N = 1000
    # radius = 10.
    # t = np.linspace(0, M * T, M * N + 1)
    # th = np.linspace(0, M * 2 * np.pi, M * N + 1)
    # zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.output_dim - len(z_eq_point)))), (M * N + 1, 1))
    # # zf_target = np.zeros((M*N+1, 6))
    # zf_target[:, 0] += -radius * np.sin(th)
    # zf_target[:, 1] += radius * np.sin(2 * th)

    # === circle with constant z ===
    M = 1
    T = 10
    N = 1000
    radius = 40.
    t = np.linspace(0, M * T, M * N + 1)
    th = np.linspace(0, M * 2 * np.pi, M * N + 1)
    zf_target = np.tile(np.hstack((z_eq_point, np.zeros(model.output_dim - len(z_eq_point)))), (M * N + 1, 1))
    # zf_target = np.zeros((M*N+1, 6))
    zf_target[:, 0] += radius * np.cos(th)
    zf_target[:, 1] += radius * np.sin(th)
    zf_target[:, 2] += -np.ones(len(t))

    z = model.zfyf_to_zy(zf=zf_target)

    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[0, 0] = 100.   # corresponding to x position of end effector
    Qz[1, 1] = 100.   # corresponding to y position of end effector
    Qz[2, 2] = 0. # 100.   # corresponding to z position of end effector
    R = 0.001 * np.eye(model.input_dim)

    dt = 0.02
    N = 3

    # Control constraints
    low = 0.0
    high = 1000.0
    U = HyperRectangle([high] * model.input_dim, [low] * model.input_dim)
    dU = None

    # State constraints
    X = None

    # Define GuSTO model
    gusto_model = SSMGuSTO(model)

    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
                       verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI',
                       max_gusto_iters=0, input_nullspace=None, dU=dU, jit=True)


# def run_scp_OL():
#     """
#      In problem_specification add:

#      from examples.trunk import trunk_SSM
#      problem = trunk_SSM.run_scp

#      then run:

#      python3 launch_sofa.py
#      """
#     from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
#     from sofacontrol.measurement_models import OutputModel, linearModel, MeasurementModel
#     from sofacontrol.scp.models.ssm import SSMGuSTO
#     from sofacontrol.scp.standalone_test import runGuSTOSolverStandAlone
#     from sofacontrol.utils import HyperRectangle, vq2qv, x2qv, load_data, qv2x, SnapshotData, Polyhedron
#     from sofacontrol.SSM import ssm
#     import pickle

#     dt = 0.01
#     prob = Problem()
#     prob.Robot = trunkRobot(dt=0.01)
#     prob.ControllerClass = OpenLoopController
#     Sequences = TrunkRobotSequences(t0=3.0, dt=dt) # t0 is delay before real inputs

#     useTimeDelay = True

#     # Load equilibrium point
#     rest_file = join(path, 'rest.pkl')
#     rest_data = load_data(rest_file)
#     qv_equilibrium = np.array(rest_data['q'][0])

#     # Set directory for SSM Models
#     pathToModel = path + '/SSMmodels/'

#     cov_q = 0.0 * np.eye(3)
#     cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
#     prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE]])

#     # Setup equilibrium point (no time delay and observed position and velocity of tip)
#     x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
#     if useTimeDelay:
#         outputModel = linearModel([TIP_NODE], 1628, vel=False)
#         z_eq_point = outputModel.evaluate(x_eq, qv=False)
#         with open(join(pathToModel, 'SSM_model_delayEmbedding.pkl'), 'rb') as f:
#             SSM_data = pickle.load(f)
#         prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=False, S_q=cov_q)
#         outputSSMModel = OutputModel(15, 3) # TODO: Modify this based on observables
#         Cout = outputSSMModel.C
#     else:
#         outputModel = linearModel([TIP_NODE], N_NODES)
#         z_eq_point = outputModel.evaluate(x_eq, qv=True)
#         with open(join(pathToModel, 'SSM_model.pkl'), 'rb') as f:
#             SSM_data = pickle.load(f)
#         prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=True, S_q=cov_q, S_v=cov_v)
#         Cout = None

#     # Loading SSM model from Matlab
#     raw_model = SSM_data['model']
#     raw_params = SSM_data['params']

#     model = ssm.SSMDynamics(z_eq_point, discrete=False, discr_method='be',
#                             model=raw_model, params=raw_params, C=Cout)

#     # Define cost functions and trajectory
#     Qz = np.zeros((model.output_dim, model.output_dim))
#     Qz[0, 0] = 100  # corresponding to x position of end effector
#     Qz[1, 1] = 100  # corresponding to y position of end effector
#     Qz[2, 2] = 0.0  # corresponding to z position of end effector
#     R = .00001 * np.eye(model.input_dim)

#     # Nullspace penalization (Hardcoded from Matlab) - nullspace of V^T * H
#     # V_ortho = np.array([-0.5106, 0.4126, -0.6370, .4041])

#     #### Define Target Trajectory ####
#     M = 3
#     T = 10
#     N = 1000
#     t = np.linspace(0, M * T, M * N)
#     th = np.linspace(0, M * 2 * np.pi, M * N)
#     zf_target = np.zeros((M * N, model.output_dim))

#     # zf_target[:, 0] = -15. * np.sin(th) - 7.1
#     # zf_target[:, 1] = 15. * np.sin(2 * th)

#     zf_target[:, 0] = -25. * np.sin(th)
#     zf_target[:, 1] = 25. * np.sin(2 * th)

#     # zf_target[:, 0] = -15. * np.sin(8 * th) - 7.1
#     # zf_target[:, 1] = 15. * np.sin(16 * th)

#     # zf_target[:, 0] = -15. * np.sin(4 * th)
#     # zf_target[:, 1] = 15. * np.sin(8 * th)

#     #z = zf_target
#     z = model.zfyf_to_zy(zf=zf_target)

#     # Define controller (wait 3 seconds of simulation time to start)
#     from types import SimpleNamespace
#     target = SimpleNamespace(z=z, Hf=outputModel.C, t=t)

#     # Control constraints
#     low = 200.0
#     high = 4000.0
#     U = HyperRectangle([high, high, high, high], [low, low, low, low])

#     # State constraints (q,v format)
#     # Hz = np.zeros((4, model.output_dim))
#     # Hz[0, 0] = 1
#     # Hz[1, 0] = -1
#     # Hz[2, 1] = 1
#     # Hz[3, 1] = -1
#     #
#     # b_z = np.array([20, 20, 15, 15])
#     # X = Polyhedron(A=Hz, b=b_z - Hz @ model.y_ref)

#     X = None
#     x0 = np.zeros((model.state_dim,))

#     # Define GuSTO model
#     N = 1000
#     gusto_model = SSMGuSTO(model)
#     # xopt, uopt, zopt, topt = runGuSTOSolverStandAlone(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
#     #                    verbose=1, warm_start=False, convg_thresh=0.001, solver='GUROBI')

#     xopt, uopt, zopt, topt = runGuSTOSolverStandAlone(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
#                                                       verbose=1, warm_start=False, convg_thresh=1e-5, solver='GUROBI',
#                                                       input_nullspace=None, jit=False)

#     ###### Plot results. Make sure comment this or robot will not animate ########
#     # zopt = vq2qv(model.zy_to_zfyf(zopt))
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.plot3D(zopt[:, 0], zopt[:, 1], zopt[:, 2], label='Open Loop Optimal Trajectory')
#     # plt.legend()
#     # plt.title('TPWL OCP Open Loop Trajectory')
#     # plt.show()

#     # Open loop
#     u, save, t = Sequences.augment_input_with_base(uopt.T, save_data=True)
#     prob.controller = OpenLoop(u.shape[0], t, u, save, dt=dt, maxNoise=0)

#     # prob.snapshots = SnapshotData(save_dynamics=False)

#     prob.opt['sim_duration'] = 13.
#     prob.simdata_dir = path
#     prob.opt['save_prefix'] = 'scp_OL_SSM'

#     return prob

# def collect_traj_data():
#     """
#     In problem_specification add:

#     from examples.hardware import diamond
#     problem = diamond.collect_POD_data

#     then run:

#     $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

#     or

#     python3 launch_sofa.py

#     This function runs a Sofa simulation with an open loop controller to collect data that will be used to identify the
#     POD basis.
#     """
#     from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
#     from sofacontrol.utils import SnapshotData
#     from sofacontrol.measurement_models import OutputModel, linearModel, MeasurementModel


#     # Adjust dt here as necessary (esp for Koopman)
#     dt = 0.01
#     prob = Problem()
#     prob.Robot = diamondRobot()
#     prob.ControllerClass = OpenLoopController
#     Sequences = DiamondRobotSequences(t0=3.0, dt=dt)  # t0 is delay before real inputs

#     cov_q = 0.0 * np.eye(3)
#     cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
#     prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])
#     outputModel = linearModel([TIP_NODE], 1628)
#     prob.measurement_model = MeasurementModel(nodes=[1354], num_nodes=1628, pos=True, vel=True, S_q=cov_q,
#                                               S_v=cov_v)

#     # Training snapshots
#     u, save, t = Sequences.lhs_sequence(nbr_samples=10, t_step=1.5, add_base=True, interp_pts=1, nbr_zeros=5, seed=4321)
#     prob.controller = OpenLoop(u.shape[0], t, u, save, dt=dt, maxNoise=0)

#     prob.simdata_dir = path
#     prob.opt['save_prefix'] = 'scp_OL_SSM'

#     return prob

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'module_test':
        module_test()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    else:
        raise RuntimeError('Not a valid function argument')