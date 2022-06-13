import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.hardware.model import diamondRobot
from sofacontrol.open_loop_sequences import DiamondRobotSequences


DEFAULT_OUTPUT_NODES = [1354, 726, 139, 1445, 729]

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
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController

    u1, save1, t1 = prob.Robot.sequences.lhs_sequence(nbr_samples=40, interp_pts=45, seed=1234,
                                                       add_base=True)  # ramp inputs between lhs samples
    u2, save2, t2 = prob.Robot.sequences.lhs_sequence(nbr_samples=25, t_step=1, seed=4321)  # step inputs of 1.5 seconds
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
    t = np.linspace(0, M*T, M*N)
    th = np.linspace(0, M * 2 * np.pi, M*N)
    zf_target = np.zeros((M*N, model.output_dim))
    zf_target[:, 3] = -15. * np.sin(th) - 7.1
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
    from sofacontrol.tpwl.controllers import ilqr, rh_ilqr, TrajTracking
    from sofacontrol.tpwl.observer import DiscreteEKFObserver
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import QuadraticCost
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
    else:
        raise RuntimeError('Not a valid function argument')