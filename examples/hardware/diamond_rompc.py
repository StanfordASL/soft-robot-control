import sys
from os.path import dirname, abspath, join

import numpy as np

path = dirname(abspath(__file__))
root = dirname(dirname(path))
sys.path.append(root)

from examples import Problem
from examples.hardware.model import diamondRobot

# Default nodes are the "end effector (1354)" and the "elbows (726, 139, 1445, 729)"
DEFAULT_OUTPUT_NODES = [1354, 726, 139, 1445, 729]


def generate_linearized_ROM():
    """
    Generate a linear ROM by converting a TPWL model

    python3 diamond_rompc.py generate_linearized_rom
    """
    from sofacontrol.baselines.rompc.rompc_utils import TPWL2LinearROM

    tpwl_file = 'tpwl_model_snapshots'
    linrom_file = 'rompc_model'

    tpwl_loc = join(path, '{}.pkl'.format(tpwl_file))
    save_loc = join(path, '{}.pkl'.format(linrom_file))
    TPWL2LinearROM(tpwl_loc, save_loc)


def run_rompc():
    """
     In problem_specification add:

     from examples.hardware import diamond_rompc
     problem = diamond_rompc.run_rompc

     then run:

     python3 launch_sofa.py

     """
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.baselines.rompc.rompc_utils import LinearROM
    from sofacontrol.baselines.rompc import rompc
    from sofacontrol.utils import QuadraticCost

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.0 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, prob.Robot.nb_nodes, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Load and configure the linear ROM model from data saved
    linrom_model_file = join(path, 'rompc_model.pkl')
    dt = 0.01
    model = LinearROM(linrom_model_file, dt, Hf=prob.output_model.C, Cf=prob.measurement_model.C)


    ##############################################
    # Problem 1, Figure 8 with constraints
    ##############################################
    # cost = QuadraticCost()
    # Qz = np.zeros((model.output_dim, model.output_dim))
    # Qz[3, 3] = 100  # corresponding to x position of end effector
    # Qz[4, 4] = 100  # corresponding to y position of end effector
    # Qz[5, 5] = 0.0  # corresponding to z position of end effector
    # cost.Q = model.H.T @ Qz @ model.H
    # cost.R = 0.00001 * np.eye(4)
    #
    # costL = QuadraticCost()
    # costL.Q = cost.Q
    # costL.R = .000001 * np.eye(30)

    ##############################################
    # Problem 2, Circle on side
    ##############################################
    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 0.0  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 100  # corresponding to z position of end effector
    cost.Q = model.H.T @ Qz @ model.H
    cost.R = 0.00001 * np.eye(4)

    costL = QuadraticCost()
    costL.Q = cost.Q
    costL.R = .000001 * np.eye(30)


    # Define controller
    prob.controller = rompc.ROMPC(model, cost, costL, dt, N_replan=1, delay=3)

    # Saving paths
    prob.opt['sim_duration'] = 8.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'rompc'

    return prob


def run_rompc_solver():
    """
    python3 diamond_rompc.py run_rompc_solver
    """
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.baselines.rompc.rompc_utils import LinearROM
    from sofacontrol.baselines.ros import runMPCSolverNode
    from sofacontrol.tpwl.tpwl_utils import Target
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.utils import HyperRectangle, Polyhedron

    output_model = linearModel(nodes=[1354], num_nodes=1628)

    # Load and configure the linear ROM model from data saved
    linrom_model_file = join(path, 'rompc_model.pkl')
    dt = 0.1
    model = LinearROM(linrom_model_file, dt, Hf=output_model.C)


    #############################################
    # Problem 1, Figure 8 with constraints
    #############################################
    # target = Target()
    # M = 3
    # T = 10
    # N = 500
    # target.t = np.linspace(0, M*T, M*N)
    # th = np.linspace(0, M * 2 * np.pi, M*N)
    # zf_target = np.zeros((M*N, model.output_dim))
    # zf_target[:, 3] = -15. * np.sin(th) - 7.1
    # zf_target[:, 4] = 15. * np.sin(2 * th)
    # target.z = model.zfyf_to_zy(zf=zf_target)
    #
    # # Cost
    # cost = QuadraticCost()
    # cost.R = .00001 * np.eye(model.input_dim)
    # Qz = np.zeros((model.output_dim, model.output_dim))
    # Qz[3, 3] = 100  # corresponding to x position of end effector
    # Qz[4, 4] = 100  # corresponding to y position of end effector
    # Qz[5, 5] = 0.0  # corresponding to z position of end effector
    # cost.Q = Qz
    #
    # # Control constraints
    # low = 200.0
    # high = 1500.0
    # U = HyperRectangle([high, high, high, high], [low, low, low, low])
    #
    # # State constraints
    # Hz = np.zeros((1, 6))
    # Hz[0, 4] = 1
    # H = Hz @ model.H
    # b_z = np.array([5])
    # X = Polyhedron(A=H, b=b_z - Hz @ model.z_ref)

    ##############################################
    # Problem 2, Circle on side
    ##############################################
    target = Target()
    M = 3
    T = 5
    N = 1000
    r = 20
    target.t = np.linspace(0, M*T, M*N)
    th = np.linspace(0, M*2*np.pi, M*N)
    x_target = np.zeros(M*N)
    y_target = r * np.sin(th)
    z_target = r - r * np.cos(th) + 107.0
    zf_target = np.zeros((M*N, 6))
    zf_target[:, 3] = x_target
    zf_target[:, 4] = y_target
    zf_target[:, 5] = z_target
    target.z = model.zfyf_to_zy(zf=zf_target)

    # Cost
    cost = QuadraticCost()
    cost.R = .00001 * np.eye(4)
    Qz = np.zeros((6, 6))
    Qz[3, 3] = 0.0  # corresponding to x position of end effector
    Qz[4, 4] = 100.0  # corresponding to y position of end effector
    Qz[5, 5] = 100.0  # corresponding to z position of end effector
    cost.Q = Qz

    # Constraints
    low = 200.0
    high = 2500.0
    U = HyperRectangle([high, high, high, high], [low, low, low, low])
    X = None

    # Define GuSTO model
    N = 10
    runMPCSolverNode(model=model, N=N, cost_params=cost, X=X, target=target, dt=dt, verbose=1,
                     warm_start=True, U=U, solver='GUROBI')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'generate_linearized_rom':
        generate_linearized_ROM()
    elif sys.argv[1] == 'run_rompc_solver':
        run_rompc_solver()
    else:
        raise RuntimeError('Not a valid command line argument')
