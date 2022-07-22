import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from sofacontrol.open_loop_sequences import TrunkRobotSequences

# TODO: Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]


def apply_constant_input(input=np.zeros(8), q0=None, save_data=False, t0=0.0, filename=None, scale_mode=1000):
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
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = environments.Trunk()
    prob.ControllerClass = OpenLoopController

    # TODO: Setting this to zero for now (t0 is when force is actually applied and when data is saved)
    Sequences = TrunkRobotSequences(t0=t0, dt=0.01)

    # 1) Wind up the robot
    t_duration1 = 1.0
    u_const = input
    u1, save1, t1 = Sequences.constant_input(u_const, t_duration1, save_data=save_data)

    # 2) Remove force (how long to settle down before stopping simulation)
    t_duration2 = 4.0
    u_const = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    u2, save2, t2 = Sequences.constant_input(u_const, t_duration2, save_data=save_data)

    # Smooth infinity symbol
    #u2, save2, t2 = Sequences.traj_tracking(amplitude=200, period=2.5, repetitions=4)

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
    from robots import environments
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = environments.Trunk()
    prob.ControllerClass = OpenLoopController

    Sequences = TrunkRobotSequences(t0=0.5, max_amplitude=100)
    u1, save1, t1 = Sequences.traj_tracking(amplitude=50, period=1.)
    u2, save2, t2 = Sequences.traj_tracking(amplitude=100, period=1.5)
    u3, save3, t3 = Sequences.traj_tracking(amplitude=150, period=2.)
    u4, save4, t4 = Sequences.traj_tracking(amplitude=200, period=2.5)
    u5, save5, t5 = Sequences.traj_tracking(amplitude=250, period=3.)
    u6, save6, t6 = Sequences.traj_tracking(amplitude=300, period=3.5)
    u7, save7, t7 = Sequences.traj_tracking(amplitude=350, period=1.0)
    u8, save8, t8 = Sequences.traj_tracking(amplitude=400, period=1.5)
    u9, save9, t9 = Sequences.traj_tracking(amplitude=450, period=2.0)
    u10, save10, t10 = Sequences.lhs_sequence(nbr_samples=100, interp_pts=10, seed=1234,
                                           add_base=True)
    u11, save11, t11 = Sequences.lhs_sequence(nbr_samples=100, t_step=0.5, seed=4321)

    u, save, t = Sequences.combined_sequence([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11],
                                             [save1, save2, save3, save4, save5, save6, save7, save8, save9, save10, save11],
                                             [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11])

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
    from robots import environments
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.mor import pod
    from sofacontrol.tpwl import tpwl_config
    from sofacontrol.tpwl.tpwl_utils import TPWLSnapshotData

    prob = Problem()
    prob.Robot = environments.Trunk()

    prob.ControllerClass = OpenLoopController
    prob.output_model = prob.Robot.get_measurement_model(nodes=[51], pos=True, vel=True)

    # Let robot settle down to equilibrium
    t0 = 2.0
    Sequences = TrunkRobotSequences(t0=t0, max_amplitude=200)
    u1, save1, t1 = Sequences.lhs_sequence(nbr_samples=350, interp_pts=10, seed=1234,
                                           add_base=True)  # ramp inputs between lhs samples
    u2, save2, t2 = Sequences.lhs_sequence(nbr_samples=350, t_step=0.5, seed=4321)  # step inputs of 0.5 seconds
    u3, save3, t3 = Sequences.traj_tracking(amplitude=150, period=2.)
    u4, save4, t4 = Sequences.traj_tracking(amplitude=200, period=2.5)
    u5, save5, t5 = Sequences.traj_tracking(amplitude=250, period=3.)

    u, save, t = Sequences.combined_sequence([u1, u2, u3, u4, u5],
                                             [save1, save2, save3, save4, save5],
                                             [t1, t2, t3, t4, t5])
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
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
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
    prob.controller = scp(model, cost, dt, N_replan=30, observer=EKF, delay=3)

    # Saving paths
    prob.opt['sim_duration'] = 13.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'scp'

    return prob

def run_gusto_solver():
    """
    python3 trunk.py run_gusto_solver
    """
    from sofacontrol.scp.models.tpwl import TPWLGuSTO
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.scp.ros import runGuSTOSolverNode
    from sofacontrol.tpwl import tpwl_config, tpwl
    from sofacontrol.utils import HyperRectangle, Polyhedron

    output_model = linearModel(nodes=[51], num_nodes=709)

    # Load and configure the TPWL model from data saved
    tpwl_model_file = join(path, 'tpwl_model_snapshots.pkl')
    config = tpwl_config.tpwl_dynamics_config()
    model = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=output_model.C)

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

    R = .00001 * np.eye(model.input_dim)
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0  # corresponding to z position of end effector
    dt = 0.1
    N = 5

    # Control constraints
    U = HyperRectangle([800., 800., 800., 800., 800., 800., 800., 800.], [0., 0., 0., 0., 0., 0., 0., 0.])

    # State constraints
    X = None

    # Define initial condition to be x_ref for initial solve
    x0 = model.rom.compute_RO_state(xf=model.rom.x_ref)

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

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'compute_POD_basis':
        compute_POD_basis()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    else:
        raise RuntimeError('Not a valid function argument')
