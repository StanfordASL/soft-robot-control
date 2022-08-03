import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from sofacontrol.open_loop_sequences import DiamondRobotSequences

# Default nodes are the "end effector (1354)" and the "elbows (726, 139, 1445, 729)"
DEFAULT_OUTPUT_NODES = [1354, 726, 139, 1445, 729]

def apply_constant_input(input=np.zeros(4), q0=None, save_data=False, t0=0.0, filename=None, scale_mode=1000):
    """
    In problem_specification add:

    import examples.diamond as diamond
    problem = diamond.apply_constant_input():

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py

    This function runs a Sofa simulation with an open loop controller to collect decaying
    trajectory data
    """
    from robots import environments
    from examples.hardware.model import diamondRobot
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    #prob.Robot = environments.Diamond(q0=q0)
    prob.Robot = diamondRobot(q0=q0, scale_mode=scale_mode)
    prob.ControllerClass = OpenLoopController

    # t0 is when force is actually applied and when data is saved
    Sequences = DiamondRobotSequences(t0=t0, dt=0.001)

    # 1) Wind up the robot
    t_duration1 = 1.0
    u_const = input
    u1, save1, t1 = Sequences.constant_input(u_const, t_duration1, save_data=save_data)

    # 2) Remove force (how long to settle down before stopping simulation)
    t_duration2 = 2.0
    u_const = np.array([0, 0, 0, 0])
    u2, save2, t2 = Sequences.constant_input(u_const, t_duration2, save_data=save_data)

    u, save, t = Sequences.combined_sequence([u1, u2], [save1, save2], [t1, t2])
    prob.controller = OpenLoop(u.shape[0], t, u, save)

    prob.snapshots = SnapshotData(save_dynamics=False)

    prob.snapshots_dir = path + "/dataCollection/"

    if filename is None:
        prob.opt['save_prefix'] = 'decay'
    else:
        prob.opt['save_prefix'] = filename
    prob.opt['sim_duration'] = t_duration1 + t_duration2

    return prob

def collect_POD_data():
    """
    In problem_specification add:

    import examples.diamond as diamond
    problem = diamond.collect_POD_data

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py

    This function runs a Sofa simulation with an open loop controller to collect data that will be used to identify the
    POD basis.
    """
    from examples.hardware.model import diamondRobot
    from robots import environments
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController

    Sequences = DiamondRobotSequences(t0=0.5)
    u1, save1, t1 = Sequences.lhs_sequence(nbr_samples=50, interp_pts=10, seed=1234,
                                           add_base=True)  # ramp inputs between lhs samples
    u2, save2, t2 = Sequences.lhs_sequence(nbr_samples=50, t_step=0.5, seed=4321)  # step inputs of 0.5 seconds
    u, save, t = Sequences.combined_sequence([u1, u2], [save1, save2], [t1, t2])

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
    config.pod_tolerance = .00005
    config.pod_type = 'a'
    results = pod.run_POD(snapshots_file, POD_file, config)

    # Plot results
    plt.plot(results['Sigma'])
    plt.yscale('log')
    plt.show()


def collect_TPWL_data():
    """
    In problem_specification add:

    from examples.diamond import diamond
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
    from examples.hardware.model import diamondRobot

    prob = Problem()
    prob.Robot = diamondRobot()

    prob.ControllerClass = OpenLoopController
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354], pos=True, vel=True)

    # Let robot settle down to equilibrium
    t0 = 2.0
    Sequences = DiamondRobotSequences(t0=t0)
    u1, save1, t1 = Sequences.lhs_sequence(nbr_samples=500, interp_pts=10, seed=1234,
                                           add_base=True)  # ramp inputs between lhs samples
    u2, save2, t2 = Sequences.lhs_sequence(nbr_samples=500, t_step=0.5, seed=4321)  # step inputs of 0.5 seconds
    u, save, t = Sequences.combined_sequence([u1, u2], [save1, save2], [t1, t2])
    print('Simulation length: {}'.format(t[-1]))
    prob.controller = OpenLoop(u.shape[0], t, u, save)

    # Specify the model reduction information
    POD_file = join(path, 'pod_model.pkl')
    rom = pod.load_POD(POD_file)

    # Specify config
    config = tpwl_config.tpwl_dynamics_config()
    config.TPWL_threshold = 200
    prob.snapshots = TPWLSnapshotData(rom, config, Hf=prob.output_model.C)

    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'tpwl_model'

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
    from examples.hardware.model import diamondRobot
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
    cov_q = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
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
    V = 0.1 * np.eye(model.get_meas_dim())
    EKF = DiscreteEKFObserver(model, W=W, V=V)

    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector
    cost.Q = model.H.T @ Qz @ model.H
    cost.Qf = model.H.T @ np.zeros((model.output_dim, model.output_dim)) @ model.H
    cost.R = .00001 * np.eye(model.input_dim)

    # Define controller (wait 2 seconds of simulation time to start)
    prob.controller = scp(model, cost, dt, N_replan=10, observer=EKF, delay=2)

    # Saving paths
    prob.opt['sim_duration'] = 12.
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

    # Define target trajectory for optimization
    T = 10
    t = np.linspace(0, T, 1000)
    th = np.linspace(0, 2 * np.pi, 1000)
    zf_target = np.zeros((1000, model.output_dim))
    zf_target[:, 3] = -20. * np.sin(th) - 5.5
    zf_target[:, 4] = 10. * np.sin(2 * th) + 1.5
    z = model.zfyf_to_zy(zf=zf_target)

    R = .00001 * np.eye(model.input_dim)
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector

    # Control constraints
    U = HyperRectangle([1500., 1500., 1500., 1500.], [0., 0., 0., 0.])

    # State constraints
    Hz = np.zeros((2, 6))
    Hz[0, 3] = 1
    Hz[1, 4] = 1
    H = Hz @ model.H
    H_full = np.vstack([-H, H])
    b_z_lb = np.array([-17.5 - 5.5, -20 + 1.5])
    b_z_ub = np.array([17.5 - 5.5, 20 + 1.5])
    offset = Hz @ model.z_ref
    b_z = np.hstack([-(b_z_lb - offset), b_z_ub - offset])
    X = Polyhedron(A=H_full, b=b_z)

    # Define initial condition to be x_ref for initial solve
    x0 = model.rom.compute_RO_state(xf=model.rom.x_ref)

    dt = 0.05
    N = 5

    # Define GuSTO model
    gusto_model = TPWLGuSTO(model)
    gusto_model.pre_discretize(dt)
    runGuSTOSolverNode(gusto_model, N, dt, Qz, R, x0, t=t, z=z, U=U, X=X,
                       verbose=1, warm_start=True, convg_thresh=0.001, solver='GUROBI')

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
    from sofacontrol.utils import QuadraticCost

    prob = Problem()
    #prob.Robot = environments.Diamond()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
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
    V = 0.1 * np.eye(model.get_meas_dim())
    EKF = DiscreteEKFObserver(model, W=W, V=V)

    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector
    cost.Q = model.H.T @ Qz @ model.H
    cost.R = .00001 * np.eye(model.input_dim)

    # Define target trajectory for optimization
    T = 10
    t = np.linspace(0, T, 1000)
    th = np.linspace(0, 2 * np.pi, 1000)
    zf_target = np.zeros((1000, model.output_dim))
    zf_target[:, 3] = -20. * np.sin(th) - 5.5
    zf_target[:, 4] = 10. * np.sin(2 * th) + 1.5
    z = model.zfyf_to_zy(zf=zf_target)

    # Define controller (wait 2 seconds of simulation time to start)
    prob.controller = ilqr(model, cost, z, dt, observer=EKF, delay=2)

    # Saving paths
    prob.opt['sim_duration'] = 12.
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
