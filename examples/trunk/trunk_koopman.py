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
from sofacontrol.utils import load_data, qv2x, remove_decimal
from sofacontrol.measurement_models import linearModel

# Default nodes are the "end effector (1354)" and the "elbows (726, 139, 1445, 729)"
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51
N_NODES = 709

modelType = "linear" # "linear" or "nonlinear"
dt = 0.05

# Load equilibrium point
rest_file = join(path, 'rest_qv.pkl')
rest_data = load_data(rest_file)
q_equilibrium = np.array(rest_data['q'][0])

# Setup equilibrium point (no time delay and observed position and velocity of tip)
x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))
output_model = linearModel(nodes=[TIP_NODE], num_nodes=N_NODES)
z_eq_point = output_model.evaluate(x_eq, qv=False)

def generate_koopman_data():
    """
    Prepare Koopman data in .mat format for MATLAB Koopman code (generate_koopman_model.m)
    """
    from scipy.io import savemat
    from sofacontrol.utils import load_data, qv2x
    from sofacontrol.measurement_models import linearModel

    koopman_data_name = 'pod_snapshots'
    num_nodes = 709
    ee_node = [51]
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


def run_koopman(T=11.):
    """
    In problem_specification add:

    from examples.diamond import diamond_koopman
    problem = diamond_koopman.run_koopman

    then run:

    python3 launch_sofa.py
    """
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.baselines.koopman import koopman_utils, koopman
    from robots import environments
    from scipy.io import loadmat
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.utils import Polyhedron
    
    if modelType == 'linear':
        koopman_data = loadmat(join(path, 'DMD_' + remove_decimal(dt) + '.mat'))['py_data'][0, 0]
        raw_model = koopman_data['model']
        raw_params = koopman_data['params']
    else:
        koopman_data = loadmat(join(path, 'koopman_model_' + remove_decimal(dt) + '.mat'))['py_data'][0, 0]
        raw_model = koopman_data['model']
        raw_params = koopman_data['params']
    
    model = koopman_utils.KoopmanModel(raw_model, raw_params, DMD=False)

    prob = Problem()
    prob.Robot = trunkRobot()
    prob.ControllerClass = ClosedLoopController

    cov_q = 0.001 * np.eye(3)
    prob.measurement_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=prob.Robot.nb_nodes, pos=True, vel=False, S_q=cov_q)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE])

    # Building A matrix
    # Hz = np.zeros((1, 3))
    # Hz[0, 1] = 1
    # b_z = np.array([5])
    # Y = Polyhedron(A=Hz, b=b_z, with_reproject=True)
    Y = None

    prob.controller = koopman.KoopmanMPC(dyn_sys=model, dt=model.Ts, delay=1, rollout_horizon=1, Y=Y)

    prob.opt['sim_duration'] = T  # Simulation time, optional
    prob.simdata_dir = path
    if modelType == 'linear':
        prob.opt['save_prefix'] = 'DMD'
    else:
        prob.opt['save_prefix'] = 'koopman'

    return prob

def run_koopman_solver():
    """
    python3 diamond_koopman.py run_koopman_solver
    """
    from sofacontrol.baselines.koopman import koopman_utils
    from sofacontrol.baselines.ros import runMPCSolverNode
    from sofacontrol.tpwl.tpwl_utils import Target
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.utils import HyperRectangle, load_data, save_data, qv2x, Polyhedron, CircleObstacle, \
        drawContinuousPath, resample_waypoints, generateModel, createTargetTrajectory, createControlConstraint, \
        createObstacleConstraint
    from scipy.io import loadmat
    
    ######## User Options ########
    saveControlTask = False
    createNewTask = False
    N = 3

    # Control Task Params
    controlTask = "ASL" # figure8, circle, or custom
    trajAmplitude = 15
    trajFreq = 17 # rad/s

    # Trajectory constraint
    # Obstacle constraints
    obstacleDiameter = [10, 8]
    obstacleLoc = [np.array([-12, 12]), np.array([8, 12])]

    # Constrol Constraints
    u_min, u_max = 0.0, 800.0
    du_max = None

    ######## Generate Koopman model and setup control task ########
    if modelType == 'linear':
        koopman_data = loadmat(join(path, 'DMD_' + remove_decimal(dt) + '.mat'))['py_data'][0, 0]
        raw_model = koopman_data['model']
        raw_params = koopman_data['params']
    else:
        koopman_data = loadmat(join(path, 'koopman_model_' + remove_decimal(dt) + '.mat'))['py_data'][0, 0]
        raw_model = koopman_data['model']
        raw_params = koopman_data['params']
        
    model = koopman_utils.KoopmanModel(raw_model, raw_params, DMD=False)
    scaling = koopman_utils.KoopmanScaling(scale=model.scale)
    
    # Define target trajectory for optimization
    trajDir = join(path, "control_tasks")
    taskFile = join(trajDir, controlTask + ".pkl")
    taskParams = {}
    
    if createNewTask:
        ######## Define the trajectory ########
        zf_target, t = createTargetTrajectory(controlTask, 'trunk', model.y_eq, model.output_dim, amplitude=trajAmplitude, freq=trajFreq)
        z = model.zfyf_to_zy(zf=zf_target)

        ######## Define a new state constraint (q, v) format ########
        ## Format [constraint number, variable/state number]
        
        # Obstacle avoidance constraint
        # X = createObstacleConstraint(model.output_dim, model.y_ref, obstacleDiameter, obstacleLoc)
        # No constraint
        X = None

        ######## Define new control constraint ########
        U, dU = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)

        ######## Save Target Trajectory and Constraints ########
        taskParams = {'z': z, 't': t, 'X': X, 'U': U, 'dU': dU}
        
        if saveControlTask:
            save_data(taskFile, taskParams)
    else:
        taskParams = load_data(taskFile)
        taskParams['z'] += z_eq_point[3:]

        # Control constraints
        u_ub = u_max * np.ones(model.m)
        u_lb = u_min * np.ones(model.m)
        u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
        u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
        U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)

        if du_max is not None:
            du_max_scaled = scaling.scale_down(u=du_max).reshape(-1)
            dU = HyperRectangle(ub=du_max_scaled, lb=-du_max_scaled)
        else:
            dU = None

        # State constraints
        if type(taskParams['X']) is CircleObstacle:
            Hz = np.zeros((2, model.n))
            Hz[0, 0] = 1
            Hz[1, 1] = 1

            obstacleLoc = np.asarray([scaling.scale_down(y=np.append(vector, 0)).reshape(-1)[:2] for vector in taskParams['obstacleLoc']])
            obstacleDiameter = [scaling.scale_down(y=diameter).reshape(-1)[0] for diameter in taskParams['obstacleDiameter']]

            taskParams['X'] = CircleObstacle(A=Hz, center=obstacleLoc, diameter=obstacleDiameter)


    # Define target trajectory for optimization
    target = Target()
    target.t = taskParams['t']
    target.z = scaling.scale_down(y=taskParams['z'][:, :model.n])
    target.u = scaling.scale_down(u=np.zeros(model.m)).reshape(-1)

    ######## Cost Function ########
    cost = QuadraticCost()
    #############################################
    # Problem 1, X-Y plane cost function
    #############################################
    cost.R = .00001 * np.eye(model.m)
    cost.Q = np.zeros((model.n, model.n))
    cost.Q[0, 0] = 100  # corresponding to x position of end effector
    cost.Q[1, 1] = 100  # corresponding to y position of end effector
    cost.Q[2, 2] = 0.0  # corresponding to z position of end effector

    # #############################################
    # # Problem 2, X-Y-Z plane cost function
    # #############################################
    # cost.R = .00001 * np.eye(model.m)
    # cost.Q = np.zeros((3, 3))
    # cost.Q[0, 0] = 50.0  # corresponding to x position of end effector
    # cost.Q[1, 1] = 100.0  # corresponding to y position of end effector
    # cost.Q[2, 2] = 100.0  # corresponding to z position of end effector

    # Consider same "scaled" cost parameters as other models
    cost.R *= np.diag(scaling.u_factor[0])
    cost.Q *= np.diag(scaling.y_factor[0])


    runMPCSolverNode(model=model, N=N, cost_params=cost, target=target, dt=model.Ts, verbose=1,
                     warm_start=True, U=U, X=taskParams['X'], dU=dU, solver='GUROBI')

def run_koopman_solver_call(taskParams):
    """
    Call in launch_sofa_closedLoopAnalysis.py
        :taskParams: dictionary of task parameters (Always in SSM coordinates)
    """
    from sofacontrol.baselines.koopman import koopman_utils
    from sofacontrol.baselines.ros import runMPCSolverNode
    from sofacontrol.tpwl.tpwl_utils import Target
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.utils import HyperRectangle, CircleObstacle
    from scipy.io import loadmat

    # Constrol Constraints
    u_min, u_max = 200.0, 2500.0
    du_max = None
    N = 3

    ######## Generate Koopman model and setup control task ########
    koopman_data = loadmat(join(path, 'koopman_model.mat'))['py_data'][0, 0]
    raw_model = koopman_data['model']
    raw_params = koopman_data['params']
    model = koopman_utils.KoopmanModel(raw_model, raw_params)
    scaling = koopman_utils.KoopmanScaling(scale=model.scale)
    
    
    # Control constraints
    u_ub = u_max * np.ones(model.m)
    u_lb = u_min * np.ones(model.m)
    u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
    u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
    U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)

    # State constraints. Assume CircleObstacle
    Hz = np.zeros((2, model.n))
    Hz[0, 0] = 1
    Hz[1, 1] = 1

    obstacleLoc = np.asarray([scaling.scale_down(y=np.append(vector, 0)).reshape(-1)[:2] for vector in taskParams['obstacleLoc']])
    obstacleDiameter = [scaling.scale_down(y=diameter).reshape(-1)[0] for diameter in taskParams['obstacleDiameter']]

    taskParams['X'] = CircleObstacle(A=Hz, center=obstacleLoc, diameter=obstacleDiameter)


    # Define target trajectory for optimization
    target = Target()
    target.t = taskParams['t']
    target.z = scaling.scale_down(y=taskParams['z'][:, :model.n])
    target.u = scaling.scale_down(u=np.zeros(model.m)).reshape(-1)

    ######## Cost Function ########
    cost = QuadraticCost()
    #############################################
    # Problem 1, X-Y plane cost function
    #############################################
    cost.R = .00001 * np.eye(model.m)
    cost.Q = np.zeros((model.n, model.n))
    cost.Q[0, 0] = 100  # corresponding to x position of end effector
    cost.Q[1, 1] = 100  # corresponding to y position of end effector
    cost.Q[2, 2] = 0.0  # corresponding to z position of end effector

    #############################################
    # Problem 2, X-Y-Z plane cost function
    #############################################
    # cost.R = .00001 * np.eye(model.m)
    # cost.Q = np.zeros((3, 3))
    # cost.Q[0, 0] = 100.0  # corresponding to x position of end effector
    # cost.Q[1, 1] = 100.0  # corresponding to y position of end effector
    # cost.Q[2, 2] = 100.0  # corresponding to z position of end effector

    # Consider same "scaled" cost parameters as other models
    cost.R *= np.diag(scaling.u_factor[0])
    cost.Q *= np.diag(scaling.y_factor[0])


    runMPCSolverNode(model=model, N=N, cost_params=cost, target=target, dt=model.Ts, verbose=1,
                     warm_start=True, U=U, X=taskParams['X'], solver='GUROBI')

# def run_koopman_solver():
#     """
#     python3 diamond_koopman.py run_koopman_solver
#     """
#     from scipy.io import loadmat
#     from sofacontrol.baselines.koopman import koopman_utils
#     from sofacontrol.baselines.ros import runMPCSolverNode
#     from sofacontrol.tpwl.tpwl_utils import Target
#     from sofacontrol.utils import QuadraticCost
#     from sofacontrol.utils import HyperRectangle, Polyhedron

#     koopman_data = loadmat(join(path, 'koopman_model.mat'))['py_data'][0, 0]
#     raw_model = koopman_data['model']
#     raw_params = koopman_data['params']
#     model = koopman_utils.KoopmanModel(raw_model, raw_params)
#     scaling = koopman_utils.KoopmanScaling(scale=model.scale)

#     cost = QuadraticCost()
#     target = Target()
#     #############################################
#     # Problem 1, Figure 8 with constraints
#     #############################################
#     M = 3
#     T = 10
#     N = 500
#     t = np.linspace(0, M*T, M*N)
#     th = np.linspace(0, M * 2 * np.pi, M*N)
#     zf_target = np.zeros((M*N, model.n))
#     zf_target[:, 0] = -20. * np.sin(th)
#     zf_target[:, 1] = 20. * np.sin(2 * th)

#     # Cost
#     cost.R = .00001 * np.eye(model.m)
#     cost.Q = np.zeros((model.n, model.n))
#     cost.Q[0, 0] = 100  # corresponding to x position of end effector
#     cost.Q[1, 1] = 100  # corresponding to y position of end effector
#     cost.Q[2, 2] = 0.0  # corresponding to z position of end effector

#     # Control constraints
#     u_ub = 800. * np.ones(model.m)
#     u_lb = 0. * np.ones(model.m)
#     u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
#     u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
#     U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)

#     # State constraints
#     # Hz = np.zeros((1, 3))
#     # Hz[0, 1] = 1
#     # H = Hz @ model.H
#     # b_z = np.array([5])
#     # b_z_ub_norm = scaling.scale_down(y=b_z).reshape(-1)[1]
#     # X = Polyhedron(A=H, b=b_z_ub_norm)

#     ##############################################
#     # Problem 2, Circle on side
#     ##############################################
#     # M = 3
#     # T = 5
#     # N = 1000
#     # r = 10
#     # t = np.linspace(0, M*T, M*N)
#     # th = np.linspace(0, M*2*np.pi, M*N)
#     # x_target = np.zeros(M*N)X
#     # zf_target = np.zeros((M*N, 3))
#     # zf_target[:, 0] = x_target
#     # zf_target[:, 1] = y_target
#     # zf_target[:, 2] = z_target
#     #
#     # # Cost
#     # cost.R = .00001 * np.eye(model.m)
#     # cost.Q = np.zeros((3, 3))
#     # cost.Q[0, 0] = 0.0  # corresponding to x position of end effector
#     # cost.Q[1, 1] = 100.0  # corresponding to y position of end effector
#     # cost.Q[2, 2] = 100.0  # corresponding to z position of end effector
#     #
#     # # Constraints
#     # u_ub = 1500. * np.ones(model.m)
#     # u_lb = 200. * np.ones(model.m)
#     # u_ub_norm = scaling.scale_down(u=u_ub).reshape(-1)
#     # u_lb_norm = scaling.scale_down(u=u_lb).reshape(-1)
#     # U = HyperRectangle(ub=u_ub_norm, lb=u_lb_norm)
#     # X = None

    
#     # Define target trajectory for optimization
#     target.t = t
#     target.z = scaling.scale_down(y=zf_target)
#     target.u = scaling.scale_down(u=np.zeros(model.m)).reshape(-1)

#     # Consider same "scaled" cost parameters as other models
#     cost.R *= np.diag(scaling.u_factor[0])
#     cost.Q *= np.diag(scaling.y_factor[0])

#     N = 3

#     runMPCSolverNode(model=model, N=N, cost_params=cost, target=target, dt=model.Ts, verbose=1,
#                      warm_start=True, U=U, solver='GUROBI')



if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'generate_koopman_data':
        generate_koopman_data()
    elif sys.argv[1] == 'run_koopman_solver':
        run_koopman_solver()
    else:
        raise RuntimeError('Not a valid function argument')
