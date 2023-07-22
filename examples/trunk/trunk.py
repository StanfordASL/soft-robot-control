import sys
from os.path import dirname, abspath, join, split, isfile

import numpy as np
from matplotlib import pyplot as plt

import pickle

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from sofacontrol.open_loop_sequences import TrunkRobotSequences
from sofacontrol.utils import load_data, qv2x
from sofacontrol.measurement_models import linearModel


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
output_model = linearModel(nodes=[TIP_NODE], num_nodes=N_NODES)
z_eq_point = output_model.evaluate(x_eq, qv=False)

def sim_OL():
    """
     In problem_specification add:

     from examples.hardware import diamond
     problem = diamond.sim_OL

     then run:

     python3 launch_sofa.py
     """
    from examples.trunk.model import trunkRobot
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.utils import SnapshotData

    path = "/media/jonas/Backup Plus/jonas_soft_robot_data/autonomous_ASSM_tests"
    with open(join(path, 'u_perturbed.pkl'), 'rb') as f:
        u = pickle.load(f)
    
    # t0 = 3.0
    dt = 0.01
    prob = Problem()
    prob.Robot = trunkRobot()
    prob.ControllerClass = OpenLoopController
    Sequences = TrunkRobotSequences(dt=dt)

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[TIP_NODE])

    # Open loop
    u, save, t = Sequences.augment_input_with_base(u, save_data=True)
    prob.controller = OpenLoop(u.shape[0], t, u, save, dt=dt)
    prob.snapshots = SnapshotData(save_dynamics=False)
    prob.opt['sim_duration'] = 11.
    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'OL_sim'

    return prob



def apply_constant_input(input, pre_tensioning, q0=None, t0=0.0, save_data=True, filepath=f"{path}/undef_traj"):
    """
    In problem_specification add:

    import examples.trunk as trunk
    problem = diamond.apply_constant_input():

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa_collectDecayTrajectories.py

    This function runs a Sofa simulation with an open loop controller to collect decaying
    trajectory data
    """
    from examples.trunk.model import trunkRobot
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = trunkRobot(q0=q0)
    prob.ControllerClass = OpenLoopController
    
    # t0 is when force is actually applied and when data is saved
    Sequences = TrunkRobotSequences(t0=t0, dt=prob.Robot.dt)

    # 1) Wind up the robot
    t_duration1 = 1.0
    # print("input", input)
    # print("pre_tensioning", pre_tensioning)
    u_const = input + pre_tensioning
    u1, save1, t1 = Sequences.constant_input(u_const, t_duration1, save_data=False)
    u1 *= np.concatenate([np.linspace(0.5, 1, int(0.8*len(t1))), np.ones(len(t1) - int(0.8*len(t1)))])
    # 2) Remove force (how long to settle down before stopping simulation)
    t_duration2 = 4.0
    u_const = pre_tensioning
    u2, save2, t2 = Sequences.constant_input(u_const, t_duration2, save_data=save_data)
    # combine the two sequences
    u, save, t = Sequences.combined_sequence([u1, u2], [save1, save2], [t1, t2])

    prob.controller = OpenLoop(u.shape[0], t, u, save)
    prob.snapshots = SnapshotData(save_dynamics=False)
    prob.snapshots_dir, prob.opt['save_prefix'] = split(filepath)[0], split(filepath)[1]
    prob.opt['sim_duration'] = t_duration1 + t_duration2

    return prob


def collect_open_loop_data(u_max=None, pre_tensioning=None, q0=None, t0=0.0, save_data=True, filepath=f"{path}/undef_traj"):

    from examples.trunk.model import trunkRobot
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData
    from scipy.interpolate import CubicSpline

    prob = Problem()
    prob.Robot = trunkRobot(q0=q0)
    prob.ControllerClass = OpenLoopController

    dt = prob.Robot.dt
    # Sequences = TrunkRobotSequences(t0=t0, dt=dt, umax=u_max)
    # u, save, t = Sequences.lhs_sequence(nbr_samples=200, interp_pts=20, seed=1234, add_base=True)  # ramp inputs between lhs samples

    n_steps = 200
    n_interp = 100
    u_dim = 8
    t_sparse = np.linspace(0, dt * n_steps * n_interp, n_steps)
    u_sparse = np.random.uniform(0, u_max, size=u_dim*n_steps).reshape((u_dim, n_steps))
    # u = np.zeros((u_dim, n_steps * n_interp + 1))
    u_eq = np.atleast_2d(pre_tensioning).astype(float).T
    u = np.tile(u_eq, (1, n_steps * n_interp + 1))
    t = np.linspace(0, dt * n_steps * n_interp, n_steps * n_interp + 1)
    for i in range(u_dim):
        u_interpolator = CubicSpline(t_sparse, u_sparse[i, :])
        u[i, :] += u_interpolator(t)
    # u = np.clip(u, 0, u_max)
    save = np.tile(True, len(t))
    assert len(t) == len(save) == u.shape[1]

    # # use smooth circle inputs to track a circle using the open-loop controller
    # with open(join("/home/jonas/Projects/stanford/soft-robot-control/examples/trunk/dataCollection/open-loop_circle", f'u.pkl'), 'rb') as f:
    #     u = pickle.load(f).T * 2
    # t = np.arange(0, u.shape[1]) * dt
    # print(t)
    # save = [True] * len(t)
    
    prob.controller = OpenLoop(u.shape[0], t, u, save)
    prob.snapshots = SnapshotData(save_dynamics=False)
    prob.snapshots_dir, prob.opt['save_prefix'] = split(filepath)[0], split(filepath)[1]
    
    return prob


def run_ilqr():
    """
     In problem_specification add:

     from examples.diamond import diamond
     problem = diamond.run_scp

     then run:

     python3 launch_sofa.py
     """
    from robots import environments
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.tpwl.controllers import ilqr, TrajTracking
    from sofacontrol.tpwl.observer import DiscreteEKFObserver
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.baselines.rompc.rompc_utils import LinearROM

    prob = Problem()
    prob.Robot = environments.Trunk()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[51])

    output_model = linearModel(nodes=[51], num_nodes=709)

    # Load and configure the linear ROM model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=output_model.C)

    dt = 0.05
    model.pre_discretize(dt=dt)

    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0  # corresponding to z position of end effector
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
    zf_target[:, 3] = -10. * np.sin(th)
    zf_target[:, 4] = 10. * np.sin(2 * th)
    z = model.zfyf_to_zy(zf=zf_target)

    # Define controller (wait 2 seconds of simulation time to start)
    from types import SimpleNamespace
    target = SimpleNamespace(z=z, Hf=output_model.C, t=t)
    N = 20

    #prob.controller = rh_ilqr(model, cost, target, dt, observer=None, delay=3, planning_horizon=N)
    prob.controller = ilqr(model, cost, target, dt, observer=None, delay=3)

    # Saving paths
    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'ilqr'

    return prob

def run_scp():
    """
     In problem_specification add:

     from examples.diamond import diamond
     problem = diamond.run_scp

     then run:

     python3 launch_sofa.py
     """
    from robots import environments
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.tpwl.controllers import scp
    from sofacontrol.tpwl.observer import DiscreteEKFObserver
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import QuadraticCost

    prob = Problem()
    prob.Robot = environments.Trunk()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.001 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 60.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[51])

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

    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0  # corresponding to z position of end effector
    cost.Q = model.H.T @ Qz @ model.H
    cost.R = .00001 * np.eye(model.input_dim)

    # Define controller (wait 2 seconds of simulation time to start)
    prob.controller = scp(model, cost, dt, N_replan=10, observer=EKF, delay=1)

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
    dt = 0.1
    N = 3

    # Control Task Params
    controlTask = "custom" # figure8, circle, pacman, or custom
    trajAmplitude = 15
    trajFreq = 17 # rad/s
    flipCoords = True # Use this only when the saved trajectory is from SSM run

    # Trajectory constraint
    # Obstacle constraints
    obstacleDiameter = [10, 8]
    obstacleLoc = [np.array([-12, 12]), np.array([8, 12])]

    # Constrol Constraints
    u_min, u_max = 200.0, 800.0
    du_max = 0.

    ######## Generate SSM model and setup control task ########
    output_model = linearModel(nodes=[51], num_nodes=709)

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
        # dU = None

        ######## Save Target Trajectory and Constraints ########
        taskParams = {'z': z, 't': t, 'X': X, 'U': U, 'dU': dU}
        if type(X) is CircleObstacle:
            taskParams = {'z': z, 't': t, 'X': X, 'U': U, 'dU': dU, 'obstacleDiameter': obstacleDiameter, 'obstacleLoc': obstacleLoc}

        if saveControlTask:
            save_data(taskFile, taskParams)
    else:
        taskParams = load_data(taskFile)
        # U, dU = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)
        # taskParams['U'], taskParams['dU'] = U, dU

        if flipCoords:
            if taskParams['z'].shape[1] == model.output_dim:
                taskParams['z'][:, :3], taskParams['z'][:, 3:] = taskParams['z'][:, 3:].copy(), taskParams['z'][:, :3].copy()
            else:
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

# def run_gusto_solver():
#     """
#     python3 trunk.py run_gusto_solver
#     """
#     from sofacontrol.scp.models.tpwl import TPWLGuSTO
#     from sofacontrol.measurement_models import linearModel
#     from sofacontrol.scp.ros import runGuSTOSolverNode
#     from sofacontrol.tpwl import tpwl_config, tpwl
#     from sofacontrol.utils import HyperRectangle, Polyhedron, save_data

#     output_model = linearModel(nodes=[51], num_nodes=709)

#     # Load and configure the TPWL model from data saved
#     tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
#     config = tpwl_config.tpwl_dynamics_config()
#     model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=output_model.C)

#     # Define target trajectory for optimization
#     M = 3
#     T = 10
#     N = 500
#     t = np.linspace(0, M*T, M*N)
#     th = np.linspace(0, M * 2 * np.pi, M*N)
#     zf_target = np.zeros((M*N, model.output_dim))

#     zf_target[:, 3] = -30. * np.sin(th)
#     zf_target[:, 4] = 30. * np.sin(2 * th)

#     z = model.zfyf_to_zy(zf=zf_target)

#     R = .00001 * np.eye(model.input_dim)
#     Qz = np.zeros((model.output_dim, model.output_dim))
#     Qz[3, 3] = 100  # corresponding to x position of end effector
#     Qz[4, 4] = 100  # corresponding to y position of end effector
#     Qz[5, 5] = 0  # corresponding to z position of end effector
#     dt = 0.1
#     N = 3

#     # Control constraints
#     U = HyperRectangle([800., 800., 800., 800., 800., 800., 800., 800.], [0., 0., 0., 0., 0., 0., 0., 0.])
#     dU = None

#     # State constraints
#     X = None

#     controlTask = "figure8"
#     trajDir = join(path, "control_tasks")
#     taskFile = join(trajDir, controlTask + ".pkl")

#     z_save = np.zeros((z.shape[0], z.shape[1]))
#     z_save[:, :3], z_save[:, 3:] = z[:, 3:].copy(), z[:, :3].copy()
#     taskParams = {'z': z_save, 't': t, 'X': X, 'U': U, 'dU': dU}

#     save_data(taskFile, taskParams)

#     # Define initial condition to be x_ref for initial solve
#     x0 = model.rom.compute_RO_state(xf=model.rom.x_ref)

#     # Define GuSTO model
#     gusto_model = TPWLGuSTO(model)
#     gusto_model.pre_discretize(dt)
#     runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
#                        verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI')

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'compute_POD_basis':
        compute_POD_basis()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    else:
        raise RuntimeError('Not a valid function argument')
