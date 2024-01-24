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
from sofacontrol.utils import load_data, qv2x
from sofacontrol.measurement_models import linearModel


DEFAULT_OUTPUT_NODES = [1354, 726, 139, 1445, 729]
TIP_NODE = 1354
N_NODES = 1628

# Load equilibrium point
rest_file = join(path, 'rest_qv.pkl')
rest_data = load_data(rest_file)
q_equilibrium = np.array(rest_data['q'][0])

# Setup equilibrium point (no time delay and observed position and velocity of tip)
x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))
output_model = linearModel(nodes=[TIP_NODE], num_nodes=N_NODES)
z_eq_point = output_model.evaluate(x_eq, qv=False)

def TPWL_rollout():
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.utils import vq2qv, x2qv
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.measurement_models import MeasurementModel

    dt = 0.01
    prob = Problem()
    prob.Robot = diamondRobot(dt=dt)
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=prob.output_model.C,
                         Cf=prob.measurement_model.C, discr_method='be')

    model.pre_discretize(dt=dt)

    pathToTraj = path + '/checkModel/'
    z_true_file = join(pathToTraj, 'z_big.csv')
    z_true = np.genfromtxt(z_true_file, delimiter=',')
    zq_true, zv_true = x2qv(z_true)
    u_true_file = join(pathToTraj, 'u_big.csv')
    u_true = np.genfromtxt(u_true_file, delimiter=',')

    T = 10.01
    N = int(T / dt)
    t_original = np.linspace(0, T, int(T / 0.01) + 1)
    t_interp = np.linspace(0, T, N + 1)
    u_interp = interp1d(t_original, u_true, axis=0)(t_interp)
    p0 = np.zeros((model.state_dim,))
    p_traj, z_traj = model.rollout(p0, u_interp, dt)

    z_traj = vq2qv(z_traj)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(z_traj[:, 0], z_traj[:, 1], z_traj[:, 2], label='Predicted Trajectory')
    ax.plot3D(zq_true[:, 0], zq_true[:, 1], zq_true[:, 2], label='Actual Trajectory')
    plt.legend()
    plt.title('TPWL Open Loop Trajectory')
    plt.show()

    z_true_qv = interp1d(t_original, np.hstack((zq_true, zv_true)), axis=0)(t_interp)
    err_SSM = z_true_qv - z_traj[:-1]
    SSM_RMSE = np.linalg.norm(np.linalg.norm(err_SSM, axis=1)) ** 2 / err_SSM.shape[0]
    print('------ Mean Squared Errors (MSEs)----------')
    print('Ours (TPWL): {}'.format(SSM_RMSE))

def collect_POD_data():
    """
    In problem_specification add:

    from examples.hardware import diamond
    problem = diamond.collect_POD_data

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py

    This function runs a Sofa simulation with an open loop controller to collect data that will be used to identify the
    POD basis.

    Also use this to collect data for the Koopman model.
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    # Adjust dt here as necessary (esp for Koopman)
    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController
    u_max = 2000. * np.ones(4)

    # Training snapshots
    # u1, save1, t1 = prob.Robot.sequences.lhs_sequence(nbr_samples=40, interp_pts=45, seed=1234,
    #                                                    add_base=True)  # ramp inputs between lhs samples
    # u2, save2, t2 = prob.Robot.sequences.lhs_sequence(nbr_samples=25, t_step=1., seed=4321)  # step inputs of 1.5 seconds
    # # Additional training data to improve models with more features
    # u3, save3, t3 = prob.Robot.sequences.lhs_sequence(nbr_samples=40, interp_pts=45, seed=6969,
    #                                                    add_base=True)  # ramp inputs between lhs samples
    # u, save, t = prob.Robot.sequences.combined_sequence([u1, u2, u3], [save1, save2, save3], [t1, t2, t3])

    # Validation snapshots
    u1, save1, t1 = prob.Robot.sequences.lhs_sequence(nbr_samples=40, interp_pts=45, seed=69,
                                                      add_base=True)  # ramp inputs between lhs samples
    u2, save2, t2 = prob.Robot.sequences.lhs_sequence(nbr_samples=25, t_step=1., seed=420)  # step inputs of 1.5 seconds
    u, save, t = prob.Robot.sequences.combined_sequence([u1, u2], [save1, save2], [t1, t2])

    prob.controller = OpenLoop(u.shape[0], t, u, save)

    prob.snapshots = SnapshotData(save_dynamics=False)

    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'pod'

    return prob


def compute_POD_basis():
    """
    After running the data_collection in a Sofa sim, run this function
    from the command line:

    python3 diamond.py compute_POD_basis

    This function loads the snapshot data and computes a POD ROM.
    """
    from sofacontrol.mor import pod

    snapshots_file = join(path, 'pod_snapshots.pkl')
    POD_file = join(path, 'pod_model.pkl')
    config = pod.pod_config()
    config.pod_tolerance = .0005
    config.pod_type = 'a'
    results = pod.run_POD(snapshots_file, POD_file, config)

    # Plot results
    plt.plot(results['Sigma'])
    plt.yscale('log')
    plt.show()


def collect_TPWL_data():
    """
    In problem_specification add:

    from examples.hardware import diamond
    problem = diamond.collect_TPWL_data

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py

    This function is used to collect snapshots for building the TPWL model.
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.mor import pod
    from sofacontrol.tpwl import tpwl_config
    from sofacontrol.tpwl.tpwl_utils import TPWLSnapshotData

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354], pos=True, vel=True)

    u1, save1, t1 = prob.Robot.sequences.lhs_sequence(nbr_samples=200, interp_pts=45, seed=1234,
                                           add_base=True)  # ramp inputs between lhs samples
    u2, save2, t2 = prob.Robot.sequences.lhs_sequence(nbr_samples=200, t_step=1.0, seed=4321)  # step inputs of 0.5 seconds
    u, save, t = prob.Robot.sequences.combined_sequence([u1, u2], [save1, save2], [t1, t2])
    print('Simulation length: {}'.format(t[-1]))
    prob.controller = OpenLoop(u.shape[0], t, u, save)

    # Specify the model reduction information
    POD_file = join(path, 'pod_model.pkl')
    rom = pod.load_POD(POD_file)

    # Specify config
    config = tpwl_config.tpwl_dynamics_config()
    config.TPWL_threshold = 700
    prob.snapshots = TPWLSnapshotData(rom, config, Hf=prob.output_model.C)

    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'tpwl_model'

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
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES)) # 0.01
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES)) # 60.0
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Load and configure the TPWL model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=prob.output_model.C,
                         Cf=prob.measurement_model.C, discr_method='zoh')

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
    # Qz[3, 3] = 50.0  # corresponding to x position of end effector
    # Qz[4, 4] = 100.0  # corresponding to y position of end effector
    # Qz[5, 5] = 100.0  # corresponding to z position of end effector
    # cost.Q = model.H.T @ Qz @ model.H
    # cost.R = .003 * np.eye(model.input_dim)


    # Define controller (wait 3 seconds of simulation time to start)
    prob.controller = scp(model, cost, dt, N_replan=1, observer=EKF, delay=1)

    # Saving paths
    prob.opt['sim_duration'] = 11.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'tpwl'

    return prob

def run_gusto_solver():
    """
    python3 diamond.py run_gusto_solver
    """
    from sofacontrol.scp.models.tpwl import TPWLGuSTO
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.scp.ros import runGuSTOSolverNode
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import createTargetTrajectory, createControlConstraint, save_data, load_data, CircleObstacle

     ######## User Options ########
    saveControlTask = False
    createNewTask = False
    dt = 0.01
    N = 3

    # Control Task Params
    controlTask = "figure8" # figure8, circle, or custom
    trajAmplitude = 15
    trajFreq = None # rad/s
    flipCoords = True # Use this only when the saved trajectory is from SSM run

    # Trajectory constraint
    # Obstacle constraints
    obstacleDiameter = [10, 8]
    obstacleLoc = [np.array([-12, 12]), np.array([8, 12])]

    # Constrol Constraints
    u_min, u_max = 200.0, 2500.0
    du_max = None

    ######## Generate SSM model and setup control task ########
    output_model = linearModel(nodes=[1354], num_nodes=1628)

    # Load and configure the TPWL model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=output_model.C,
                         discr_method='zoh')
    
    # Define target trajectory for optimization
    trajDir = join(path, "control_tasks")
    taskFile = join(trajDir, controlTask + ".pkl")
    taskParams = {}
    
    if createNewTask:
        ######## Define the trajectory ########
        zf_target, t = createTargetTrajectory(controlTask, 'diamond', model.y_eq, model.output_dim, amplitude=trajAmplitude, freq=trajFreq)
        z = model.zfyf_to_zy(zf=zf_target)

        ######## Define a new state constraint (q, v) format ########
        ## Format [constraint number, variable/state number]
        
        # Obstacle avoidance constraint
        # X = createObstacleConstraint(model.output_dim, model.y_ref, obstacleDiameter, obstacleLoc)
        # No constraint
        X = None

        ######## Define new control constraint ########
        U, dU = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)
        # dU = None

        ######## Save Target Trajectory and Constraints ########
        taskParams = {'z': z, 't': t, 'X': X, 'U': U, 'dU': dU}
        if type(X) is CircleObstacle:
            taskParams = {'z': z, 't': t, 'X': X, 'U': U, 'dU': dU, 'obstacleDiameter': obstacleDiameter, 'obstacleLoc': obstacleLoc}

        if saveControlTask:
            save_data(taskFile, taskParams)
    else:
        taskParams = load_data(taskFile)

        taskParams['U'], taskParams['dU'] = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)

        if flipCoords:
            taskParams['z'] = np.hstack((np.zeros((taskParams['z'].shape[0], 3)), taskParams['z']))
            
            # Need to flip all state/obstacle constraints
            if type(taskParams['X']) is CircleObstacle:
                Hz = np.zeros((2, model.output_dim))
                Hz[0, 3] = 1
                Hz[1, 4] = 1
                H = Hz @ model.H

                taskParams['X'] = CircleObstacle(A=H, center=taskParams['obstacleLoc'] - Hz @ model.z_ref, diameter=taskParams['obstacleDiameter'])


    ######## Cost Function ########
    #############################################
    # Problem 1, X-Y plane cost function
    #############################################
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector
    R = .00001 * np.eye(model.input_dim)

    #############################################
    # Problem 2, X-Y-Z plane cost function
    #############################################
    # R = .00001 * np.eye(model.input_dim)
    # Qz = np.zeros((model.output_dim, model.output_dim))
    # Qz[3, 3] = 100.0  # corresponding to x position of end effector
    # Qz[4, 4] = 100.0  # corresponding to y position of end effector
    # Qz[5, 5] = 100.0  # corresponding to z position of end effector

    # Define initial condition to be x_ref for initial solve
    x0 = model.rom.compute_RO_state(xf=model.rom.x_ref)

    gusto_model = TPWLGuSTO(model)
    gusto_model.pre_discretize(dt)
    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=taskParams['t'], z=taskParams['z'], U=taskParams['U'], X=taskParams['X'],
                       verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI',
                       max_gusto_iters=0, input_nullspace=None, dU=taskParams['dU'], jit=True)

def run_scp_OL():
    """
     In problem_specification add:

     from examples.hardware import diamond
     problem = diamond.run_scp_OL

     then run:

     python3 launch_sofa.py
     """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.scp.models.tpwl import TPWLGuSTO
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.scp.standalone import runGuSTOSolverStandAlone
    from sofacontrol.utils import HyperRectangle, vq2qv, x2qv, SnapshotData

    t0 = 3.0
    dt = 0.05
    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController
    Sequences = DiamondRobotSequences(t0=t0, dt=dt)

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Load and configure the TPWL model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=prob.output_model.C,
                         Cf=prob.measurement_model.C, discr_method='be')

    #############################################
    # Problem 1, Figure 8 with constraints
    #############################################
    M = 3
    T = 10
    N = 500
    t = np.linspace(0, M * T, M * N)
    th = np.linspace(0, M * 2 * np.pi, M * N)
    zf_target = np.zeros((M * N, model.output_dim))
    zf_target[:, 3] = -15. * np.sin(2 * th) - 7.1
    zf_target[:, 4] = 15. * np.sin(4 * th)

    # zf_target[:, 3] = -15. * np.sin(8 * th) - 7.1
    # zf_target[:, 4] = 15. * np.sin(16 * th)
    z = model.zfyf_to_zy(zf=zf_target)

    # Cost
    R = .00001 * np.eye(model.input_dim)
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector

    # Control constraints
    low = 200.0
    high = 4000.0
    U = HyperRectangle([high, high, high, high], [low, low, low, low])

    # State Constraints
    X = None

    x0 = model.rom.compute_RO_state(xf=model.rom.x_ref)

    # Define GuSTO model
    N = 200
    gusto_model = TPWLGuSTO(model)
    gusto_model.pre_discretize(dt)
    xopt, uopt, zopt, topt = runGuSTOSolverStandAlone(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
                       verbose=1, warm_start=False, convg_thresh=0.001, solver='GUROBI')

    ###### Plot results. Make sure comment this or robot will not animate ########
    # zopt = vq2qv(model.zy_to_zfyf(zopt))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot3D(zopt[:, 0], zopt[:, 1], zopt[:, 2], label='Open Loop Optimal Trajectory')
    # plt.legend()
    # plt.title('TPWL OCP Open Loop Trajectory')
    # plt.show()

    # Open loop
    u, save, t = Sequences.augment_input_with_base(uopt.T, save_data=True)
    prob.controller = OpenLoop(u.shape[0], t, u, save, dt=dt)

    #prob.snapshots = SnapshotData(save_dynamics=False)

    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'scp_OL_TPWL'

    return prob

def generate_control_data():
    """
    Prepare OL control data in .mat format for MATLAB code
    """
    from scipy.io import savemat, loadmat
    from sofacontrol.utils import load_data, qv2x, save_data, vq2qv
    from sofacontrol.measurement_models import linearModel

    useVel = True
    control_data_name = 'tpwl_sim'
    names = ['diamond_figure8_full']

    num_nodes = 1628
    ee_node = DEFAULT_OUTPUT_NODES
    TPWL_data = load_data(join(path, '{}.pkl'.format(control_data_name)))

    measurement_models = [linearModel(nodes=ee_node, num_nodes=num_nodes, pos=True, vel=useVel)]


    for i, name in enumerate(names):
        mat_data_file = join(path, '{}.mat'.format(name))
        y = measurement_models[0].evaluate(x=np.asarray(TPWL_data['x']).T)
        matlab_data = dict()
        if useVel:
            matlab_data['y'] = vq2qv(y.T)
        else:
            matlab_data['y'] = y.T
        matlab_data['u'] = np.asarray(TPWL_data['u'])
        matlab_data['t'] = np.atleast_2d(TPWL_data['dt'] * np.asarray(range(len(matlab_data['u']))))

        savemat(mat_data_file, matlab_data)


def run_ilqr():
    """
     In problem_specification add:

     from examples.diamond import diamond
     problem = diamond.run_scp

     then run:

     python3 launch_sofa.py
     """
    from robots import environments
    from examples.hardware.model import diamondRobot
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.tpwl.controllers import ilqr
    from sofacontrol.tpwl.observer import DiscreteEKFObserver
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import QuadraticCost, SnapshotData
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.baselines.rompc.rompc_utils import LinearROM

    prob = Problem()
    #prob.Robot = environments.Diamond()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    output_model = linearModel(nodes=[1354], num_nodes=1628)

    # Load and configure the linear ROM model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=output_model.C)

    dt = 0.1
    model.pre_discretize(dt=dt)

    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector
    cost.Q = Qz
    cost.Qf = np.zeros_like(Qz)
    cost.R = .00001 * np.eye(model.input_dim)

    # Define target trajectory for optimization
    M = 3
    T = 10
    N = 1000
    t = np.linspace(0, M * T, M * N)
    th = np.linspace(0, M * 2 * np.pi, M * N)
    zf_target = np.zeros((M * N, model.output_dim))
    zf_target[:, 3] = -15. * np.sin(th)
    zf_target[:, 4] = 15. * np.sin(2 * th)
    z = model.zfyf_to_zy(zf=zf_target)

    # Define controller (wait 2 seconds of simulation time to start)
    from types import SimpleNamespace
    target = SimpleNamespace(z=z, Hf=output_model.C, t=t)
    N = 20

    #prob.controller = rh_ilqr(model, cost, target, dt, observer=None, delay=3, planning_horizon=N)
    prob.controller = ilqr(model, cost, target, dt, observer=None, delay=3)

    prob.snapshots = SnapshotData(save_dynamics=False)

    # Saving paths
    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'ilqr'

    return prob

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'compute_POD_basis':
        compute_POD_basis()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    elif sys.argv[1] == 'generate_control_data':
        generate_control_data()
    else:
        raise RuntimeError('Not a valid function argument')