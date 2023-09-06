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
N_NODES = 1628

modelType = 'linear' # "delays", "posvel", "singleDelay", "linear"
dt = 0.02 # This dt for when to recalculate control

######## Generate LDO Parameters ########
Mper = 20   # Number of periods to simulate
Tper = 2.  # Period of trajectory
# Nper = int(Tper / dt) + 1    # Number of points per period (this will be trigger for doing LDO)
Nper = None

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
    from sofacontrol.measurement_models import linearModel, OutputModel
    from sofacontrol.utils import QuadraticCost, qv2x, load_data, Polyhedron, generateModel
    from sofacontrol.SSM import ssm
    from sofacontrol.SSM.observer import SSMObserver, DiscreteEKFObserver
    from sofacontrol.SSM.controllers import scp
    from scipy.io import loadmat
    import pickle

    ######## User Options ########
    # Set directory for SSM Models and import
    pathToModel = path + '/SSMmodels/'
    # pathToModel = "/home/jalora/Desktop/diamond_origin/000/SSMmodel_delay-embedding_ROMOrder=3_localV" # join(path, "SSMmodels", "model_004")
    # Simulation settings
    sim_duration = 11.
    save_prefix = 'ssmr_' + modelType

    ######## Setup the Robot Environment and Type of Controller ########
    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    ######## Generate SSM Model ########
    model = generateModel(path, pathToModel, [TIP_NODE], N_NODES, modelType=modelType)

    ######## Specify a measurement of what we observe during simulation ########
    cov_q = 0.001 * np.eye(3)
    cov_v = 60.0 * np.eye(3) # * len(DEFAULT_OUTPUT_NODES))
    prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE])
    if model.params['delay_embedding']:
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=False, S_q=cov_q)
    else:
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=True, S_q=cov_q, S_v=cov_v)
    
    # Pure SSM Manifold Observer
    observer = SSMObserver(model)

    # Set up an EKF observer
    # W = np.diag(np.ones(model.state_dim))
    # V = 0.1 * np.eye(model.output_dim)
    # observer = DiscreteEKFObserver(model, W=W, V=V)

    # Define controller (wait 1 second of simulation time to start)
    prob.controller = scp(model, QuadraticCost(), dt, N_replan=1, delay=1, feedback=False, EKF=observer)

    # Saving paths
    prob.opt['sim_duration'] = sim_duration
    prob.simdata_dir = path
    prob.opt['save_prefix'] = save_prefix

    return prob

def run_gusto_solver():
    """
    python3 diamond.py run_gusto_solver
    """
    from sofacontrol.scp.models.ssm import SSMGuSTO
    from sofacontrol.measurement_models import linearModel, OutputModel
    from sofacontrol.scp.ros import runGuSTOSolverNode
    from sofacontrol.utils import HyperRectangle, load_data, save_data, qv2x, Polyhedron, CircleObstacle, \
        drawContinuousPath, resample_waypoints, generateModel, createTargetTrajectory, createControlConstraint, \
        createObstacleConstraint
    from sofacontrol.SSM import ssm
    from scipy.io import loadmat
    import pickle
    
    ######## User Options ########
    saveControlTask = False
    createNewTask = False
    N = 3

    ###### Circle Parameters ######
    # Control Task Params
    controlTask = "circle" # figure8, circle, or custom
    trajAmplitude = 10
    trajFreq = 20 # rad/s # 15, 20, 25, 30, 35

    # Star trajectory - only used when custom trajectory is selected
    pathToTraceImage = "/home/jalora/Desktop/star.png"
    outdofs = [0, 1, 2]
    z_offset = 107.
    repeat_traj = None

    # ###### Other Traj Parameters ######
    # # Control Task Params
    # controlTask = "figure8" # figure8, circle, or custom
    # trajAmplitude = 15
    # trajFreq = None # rad/s

    # # Star trajectory - only used when custom trajectory is selected
    # pathToTraceImage = "/home/jalora/Desktop/star.png"
    # outdofs = [0, 1, 2] # outdofs = [0, 1, 2]
    # z_offset = 12. + 107. # 107.
    # repeat_traj = 2

    # Trajectory constraint
    # Obstacle constraints
    obstacleDiameter = [10, 8]
    obstacleLoc = [np.array([-12, 12]), np.array([8, 12])]

    # Constrol Constraints
    u_min, u_max = 0.0, 4200.0
    du_max = None

    ######## Generate SSM model and setup control task ########
    # Set directory for SSM Models
    pathToModel = path + '/SSMmodels/'
    # pathToModel = "/home/jalora/Desktop/diamond_origin/000/SSMmodel_delay-embedding_ROMOrder=3_localV" # join(path, "SSMmodels", "model_004")
    model = generateModel(path, pathToModel, [TIP_NODE], N_NODES, modelType=modelType, isLinear=True if modelType == 'linear' else False)
    
    # Define target trajectory for optimization
    trajDir = join(path, "control_tasks")
    taskFile = join(trajDir, controlTask + ".pkl")
    taskParams = {}
    
    if createNewTask:
        ######## Define the trajectory ########
        zf_target, t = createTargetTrajectory(controlTask, 'diamond', model.y_eq, model.output_dim, amplitude=trajAmplitude, 
                                              freq=trajFreq, pathToImage=pathToTraceImage, outdofs=outdofs, z_offset=z_offset,
                                              repeat_traj=repeat_traj)
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
        if saveControlTask:
            save_data(taskFile, taskParams)
    else:
        taskParams = load_data(taskFile)

        taskParams['U'], taskParams['dU'] = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)

        if taskParams['z'].shape[1] < model.output_dim:
            taskParams['z'] = np.hstack((taskParams['z'], np.zeros((taskParams['z'].shape[0], model.output_dim - taskParams['z'].shape[1]))))

    ######## Cost Function ########
    #############################################
    # Problem 1, X-Y plane cost function
    #############################################
    # Qz = np.zeros((model.output_dim, model.output_dim))
    # Qz[0, 0] = 100  # corresponding to x position of end effector
    # Qz[1, 1] = 100  # corresponding to y position of end effector
    # Qz[2, 2] = 0.0  # corresponding to z position of end effector
    # R = .00001 * np.eye(model.input_dim)

    #############################################
    # Problem 2, X-Y-Z plane cost function
    #############################################
    R = .00001 * np.eye(model.input_dim)
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[0, 0] = 100.0  # corresponding to x position of end effector
    Qz[1, 1] = 100.0  # corresponding to y position of end effector
    Qz[2, 2] = 100.0  # corresponding to z position of end effector

    # Define initial condition to be x_ref for initial solve
    x0 = np.zeros(model.state_dim)

    # Define GuSTO model
    gusto_model = SSMGuSTO(model)
    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=taskParams['t'], z=taskParams['z'], U=taskParams['U'], X=taskParams['X'],
                       verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI',
                       max_gusto_iters=0, input_nullspace=None, dU=taskParams['dU'], jit=True)

def run_scp_LDO():
    """
     In problem_specification add:

     from examples.hardware import diamond
     problem = diamond.run_scp

     then run:

     python3 launch_sofa.py
     """
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.measurement_models import linearModel, OutputModel
    from sofacontrol.utils import QuadraticCost, qv2x, load_data, Polyhedron, generateModel
    from sofacontrol.SSM import ssm
    from sofacontrol.SSM.observer import SSMObserver, SSMObserverLDO
    from sofacontrol.SSM.controllers import scp
    from scipy.io import loadmat
    import pickle

    ######## User Options ########
    # Set directory for SSM Models and import
    pathToModel = path + '/SSMmodels/'
    # pathToModel = "/home/jalora/Desktop/diamond_origin/000/SSMmodel_delay-embedding_ROMOrder=3_localV" # join(path, "SSMmodels", "model_004")
    # Simulation settings
    sim_duration = 41.
    if Nper is None:
        save_prefix = 'ssmr_' + modelType
    else:
        save_prefix = 'ssmr_' + modelType + '_LDO'

    ######## Setup the Robot Environment and Type of Controller ########
    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    ######## Generate SSM Model ########
    model = generateModel(path, pathToModel, [TIP_NODE], N_NODES, modelType=modelType, isLinear=True if modelType == 'linear' else False,
                          Nper=Nper, dt=dt, gains_path=path, Tper=Tper)

    ######## Specify a measurement of what we observe during simulation ########
    cov_q = 0.0 * np.eye(3)
    cov_v = 0.0 * np.eye(3) # * len(DEFAULT_OUTPUT_NODES))
    prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE])
    if model.params['delay_embedding']:
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=False, S_q=cov_q)
    else:
        prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=True, S_q=cov_q, S_v=cov_v)
    
    # Pure SSM Manifold Observer
    if model.LDO:
        observer = SSMObserverLDO(model)
    else:
        observer = SSMObserver(model)

    # Set up an EKF observer
    # W = np.diag(np.ones(model.state_dim))
    # V = 0.1 * np.eye(model.output_dim)
    # observer = DiscreteEKFObserver(model, W=W, V=V)

    # Define controller (wait 1 second of simulation time to start)
    prob.controller = scp(model, QuadraticCost(), dt, N_replan=1, delay=1, feedback=False, EKF=observer)

    # Saving paths
    prob.opt['sim_duration'] = sim_duration
    prob.simdata_dir = path
    prob.opt['save_prefix'] = save_prefix

    return prob

def run_gusto_solver_LDO():
    """
    python3 diamond_SSM.py run_gusto_solver_LDO
    """
    from sofacontrol.scp.models.ssm import SSMGuSTO
    from sofacontrol.measurement_models import linearModel, OutputModel
    from sofacontrol.scp.ros import runGuSTOSolverNode
    from sofacontrol.utils import HyperRectangle, load_data, save_data, qv2x, Polyhedron, CircleObstacle, \
        drawContinuousPath, resample_waypoints, generateModel, createTargetTrajectory, createControlConstraint, \
        createObstacleConstraint
    from sofacontrol.SSM import ssm
    from scipy.io import loadmat
    import pickle
    
    ######## User Options ########
    saveControlTask = True
    createNewTask = True
    N = 3

    ###### Circle Parameters ######
    # Control Task Params
    controlTask = "figure8" # figure8, circle, or custom
    trajAmplitude = 30.
    trajFreq = None # rad/s # 15, 20, 25, 30, 35
    z_offset = None
    
    # controlTask = "circle" # figure8, circle, or custom
    # trajAmplitude = 10.
    # trajFreq = None # rad/s # 15, 20, 25, 30, 35
    # z_offset = 107.
    
    outdofs = [0, 1, 2]

    # Constrol Constraints
    u_min, u_max = 0.0, 4200.0
    du_max = None


    ######## Generate SSM model and setup control task ########
    # Set directory for SSM Models
    pathToModel = path + '/SSMmodels/'
    # pathToModel = "/home/jalora/Desktop/diamond_origin/000/SSMmodel_delay-embedding_ROMOrder=3_localV" # join(path, "SSMmodels", "model_004")
    model = generateModel(path, pathToModel, [TIP_NODE], N_NODES, modelType=modelType, isLinear=True if modelType == 'linear' else False, 
                          Nper=Nper, dt=dt, gains_path=path, Tper=Tper)
    
    # Define target trajectory for optimization
    trajDir = join(path, "control_tasks")
    taskFile = join(trajDir, controlTask + ".pkl")
    taskParams = {}
    
    if createNewTask:
        ######## Define the trajectory ########
        zf_target, t = createTargetTrajectory(controlTask, 'diamond', model.y_eq, model.output_dim, amplitude=trajAmplitude, 
                                              freq=trajFreq, Mper=Mper, Tper=Tper, outdofs=outdofs, z_offset=z_offset)
        z = model.zfyf_to_zy(zf=zf_target)

        ######## Define a new state constraint (q, v) format ########
        ## Format [constraint number, variable/state number]
        
        # Obstacle avoidance constraint
        # X = createObstacleConstraint(model.output_dim, model.y_ref, obstacleDiameter, obstacleLoc)
        # No constraint
        X = None

        ######## Define new control constraint ########
        U, dU = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)
        # U, dU = None, None

        ######## Save Target Trajectory and Constraints ########
        taskParams = {'z': z, 't': t, 'X': X, 'U': U, 'dU': dU}
        if saveControlTask:
            save_data(taskFile, taskParams)
    else:
        taskParams = load_data(taskFile)

        taskParams['U'], taskParams['dU'] = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)

        if taskParams['z'].shape[1] < model.output_dim:
            taskParams['z'] = np.hstack((taskParams['z'], np.zeros((taskParams['z'].shape[0], model.output_dim - taskParams['z'].shape[1]))))

    ######## Cost Function ########
    #############################################
    # Problem 1, X-Y plane cost function
    #############################################
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[0, 0] = 100  # corresponding to x position of end effector
    Qz[1, 1] = 100  # corresponding to y position of end effector
    Qz[2, 2] = 0.0  # corresponding to z position of end effector
    R = 0.001 * np.eye(model.input_dim) # 0.00001

    #############################################
    # Problem 2, X-Y-Z plane cost function
    #############################################
    # R = 0.001 * np.eye(model.input_dim) # 0.001
    # Qz = np.zeros((model.output_dim, model.output_dim))
    # Qz[0, 0] = 100.0  # corresponding to x position of end effector
    # Qz[1, 1] = 100.0  # corresponding to y position of end effector
    # Qz[2, 2] = 100.0  # corresponding to z position of end effector

    # Define initial condition to be x_ref for initial solve
    x0 = np.zeros(model.state_dim)

    # Define GuSTO model
    gusto_model = SSMGuSTO(model)
    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=taskParams['t'], z=taskParams['z'], U=taskParams['U'], X=taskParams['X'],
                       verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI',
                       max_gusto_iters=0, input_nullspace=None, dU=taskParams['dU'], jit=True)

def run_scp_OL():
    """
     In problem_specification add:

     from examples.hardware import diamond
     problem = diamond.run_scp

     then run:

     python3 launch_sofa.py
     """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.measurement_models import OutputModel, linearModel, MeasurementModel
    from sofacontrol.scp.models.ssm import SSMGuSTO
    from sofacontrol.scp.standalone_test import runGuSTOSolverStandAlone
    from sofacontrol.utils import HyperRectangle, vq2qv, x2qv, load_data, qv2x, SnapshotData, Polyhedron
    from sofacontrol.SSM import ssm
    from scipy.io import loadmat
    import pickle

    dt = 0.01
    prob = Problem()
    prob.Robot = diamondRobot(dt=0.01)
    prob.ControllerClass = OpenLoopController
    Sequences = DiamondRobotSequences(t0=3.0, dt=dt) # t0 is delay before real inputs

    useTimeDelay = False

    # Load equilibrium point
    rest_file = join(path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    qv_equilibrium = np.array(rest_data['rest'])

    # Set directory for SSM Models
    pathToModel = path + '/SSMmodels/'

    cov_q = 0.0 * np.eye(3)
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Setup equilibrium point (no time delay and observed position and velocity of tip)
    x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
    if useTimeDelay:
        outputModel = linearModel([TIP_NODE], 1628, vel=False)
        z_eq_point = outputModel.evaluate(x_eq, qv=False)
        with open(join(pathToModel, 'SSM_model_delayEmbedding.pkl'), 'rb') as f:
            SSM_data = pickle.load(f)
        prob.measurement_model = MeasurementModel(nodes=[1354], num_nodes=1628, pos=True, vel=False, S_q=cov_q)
        outputSSMModel = OutputModel(15, 3) # TODO: Modify this based on observables
        Cout = outputSSMModel.C
    else:
        outputModel = linearModel([TIP_NODE], 1628)
        z_eq_point = outputModel.evaluate(x_eq, qv=True)
        with open(join(pathToModel, 'SSM_model.pkl'), 'rb') as f:
            SSM_data = pickle.load(f)
        prob.measurement_model = MeasurementModel(nodes=[1354], num_nodes=1628, pos=True, vel=True, S_q=cov_q, S_v=cov_v)
        Cout = None

    # Loading SSM model from Matlab
    raw_model = SSM_data['model']
    raw_params = SSM_data['params']

    model = ssm.SSMDynamics(z_eq_point, discrete=False, discr_method='be',
                            model=raw_model, params=raw_params, C=Cout)

    # Define cost functions and trajectory
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[0, 0] = 100  # corresponding to x position of end effector
    Qz[1, 1] = 100  # corresponding to y position of end effector
    Qz[2, 2] = 0.0  # corresponding to z position of end effector
    R = .00001 * np.eye(model.input_dim)

    # Nullspace penalization (Hardcoded from Matlab) - nullspace of V^T * H
    # V_ortho = np.array([-0.5106, 0.4126, -0.6370, .4041])

    #### Define Target Trajectory ####
    M = 3
    T = 10
    N = 1000
    t = np.linspace(0, M * T, M * N)
    th = np.linspace(0, M * 2 * np.pi, M * N)
    zf_target = np.zeros((M * N, model.output_dim))

    # zf_target[:, 0] = -15. * np.sin(th) - 7.1
    # zf_target[:, 1] = 15. * np.sin(2 * th)

    zf_target[:, 0] = -25. * np.sin(th)
    zf_target[:, 1] = 25. * np.sin(2 * th)

    # zf_target[:, 0] = -15. * np.sin(8 * th) - 7.1
    # zf_target[:, 1] = 15. * np.sin(16 * th)

    # zf_target[:, 0] = -15. * np.sin(4 * th)
    # zf_target[:, 1] = 15. * np.sin(8 * th)

    #z = zf_target
    z = model.zfyf_to_zy(zf=zf_target)

    # Define controller (wait 3 seconds of simulation time to start)
    from types import SimpleNamespace
    target = SimpleNamespace(z=z, Hf=outputModel.C, t=t)

    # Control constraints
    low = 200.0
    high = 4000.0
    U = HyperRectangle([high, high, high, high], [low, low, low, low])

    # State constraints (q,v format)
    # Hz = np.zeros((4, model.output_dim))
    # Hz[0, 0] = 1
    # Hz[1, 0] = -1
    # Hz[2, 1] = 1
    # Hz[3, 1] = -1
    #
    # b_z = np.array([20, 20, 15, 15])
    # X = Polyhedron(A=Hz, b=b_z - Hz @ model.y_ref)

    X = None
    x0 = np.zeros((model.state_dim,))

    # Define GuSTO model
    N = 1000
    gusto_model = SSMGuSTO(model)
    # xopt, uopt, zopt, topt = runGuSTOSolverStandAlone(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
    #                    verbose=1, warm_start=False, convg_thresh=0.001, solver='GUROBI')

    xopt, uopt, zopt, topt = runGuSTOSolverStandAlone(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
                                                      verbose=1, warm_start=False, convg_thresh=1e-5, solver='GUROBI',
                                                      input_nullspace=None, jit=False)

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
    prob.controller = OpenLoop(u.shape[0], t, u, save, dt=dt, maxNoise=0)

    # prob.snapshots = SnapshotData(save_dynamics=False)

    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'scp_OL_SSM'

    return prob

def collect_traj_data():
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
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData
    from sofacontrol.measurement_models import OutputModel, linearModel, MeasurementModel


    # Adjust dt here as necessary (esp for Koopman)
    dt = 0.01
    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController
    Sequences = DiamondRobotSequences(t0=3.0, dt=dt)  # t0 is delay before real inputs

    cov_q = 0.0 * np.eye(3)
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])
    outputModel = linearModel([TIP_NODE], 1628)
    prob.measurement_model = MeasurementModel(nodes=[1354], num_nodes=1628, pos=True, vel=True, S_q=cov_q,
                                              S_v=cov_v)

    # Training snapshots
    u, save, t = Sequences.lhs_sequence(nbr_samples=10, t_step=1.5, add_base=True, interp_pts=1, nbr_zeros=5, seed=4321)
    prob.controller = OpenLoop(u.shape[0], t, u, save, dt=dt, maxNoise=0)

    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'scp_OL_SSM'

    return prob

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'module_test':
        module_test()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    elif sys.argv[1] == 'run_gusto_solver_LDO':
        run_gusto_solver_LDO()
    else:
        raise RuntimeError('Not a valid function argument')