import sys
from os.path import dirname, abspath, join

import numpy as np

path = dirname(abspath(__file__))
root = dirname(dirname(path))
sys.path.insert(0, root) # insert to beginning of array to reduce chance of file name collision

from examples import Problem

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

     from examples.diamond import diamond_rompc
     problem = diamond_rompc.run_rompc

     then run:

     python3 launch_sofa.py

     """
    from robots import environments
    from sofacontrol.closed_loop_controller import ClosedLoopController
    from sofacontrol.measurement_models import MeasurementModel
    from sofacontrol.baselines.rompc.rompc_utils import LinearROM
    from sofacontrol.baselines.rompc import rompc
    from sofacontrol.utils import QuadraticCost

    prob = Problem()
    prob.Robot = environments.Diamond()
    prob.ControllerClass = ClosedLoopController

    # Specify a measurement and output model
    # covariance between states of v (and q) but not together
    cov_q = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    cov_v = 0.1 * np.eye(3 * len(DEFAULT_OUTPUT_NODES))
    prob.measurement_model = MeasurementModel(DEFAULT_OUTPUT_NODES, 1628, S_q=cov_q, S_v=cov_v)
    prob.output_model = prob.Robot.get_measurement_model(nodes=[1354])

    # Load and configure the linear ROM model from data saved
    linrom_model_file = join(path, 'rompc_model.pkl')
    dt = 0.01
    model = LinearROM(linrom_model_file, dt, Hf=prob.output_model.C, Cf=prob.measurement_model.C)

    cost = QuadraticCost()
    Qz = np.zeros((model.output_dim, model.output_dim))
    Qz[3, 3] = 100  # corresponding to x position of end effector
    Qz[4, 4] = 100  # corresponding to y position of end effector
    Qz[5, 5] = 0.0  # corresponding to z position of end effector
    cost.Q = model.H.T @ Qz @ model.H
    cost.R = .0001 * np.eye(model.input_dim)

    costL = QuadraticCost()
    costL.Q = cost.Q
    costL.R = .001 * np.eye(model.meas_dim)

    # Define controller
    prob.controller = rompc.ROMPC(model, cost, costL, dt, N_replan=10, delay=2)

    # Saving paths
    prob.opt['sim_duration'] = 12.
    prob.simdata_dir = path
    prob.opt['save_prefix'] = 'rompc'

    return prob


def run_rompc_solver():
    """
    python3 diamond_rompc.py run_rompc_solver
    """
    from sofacontrol.measurement_models import linearModel
    from sofacontrol.baselines.ros import runMPCSolverNode
    from sofacontrol.baselines.rompc.rompc_utils import LinearROM
    from sofacontrol.utils import QuadraticCost
    from sofacontrol.tpwl.tpwl_utils import Target
    from sofacontrol.utils import HyperRectangle, Polyhedron

    output_model = linearModel(nodes=[1354], num_nodes=1628)

    # Load and configure the linear ROM model from data saved
    linrom_model_file = join(path, 'rompc_model.pkl')
    dt = 0.05
    model = LinearROM(linrom_model_file, dt, Hf=output_model.C)

    # Define target trajectory for optimization
    target = Target()
    T = 10
    target.t = np.linspace(0, T, 1000)
    th = np.linspace(0, 2 * np.pi, 1000)
    zf_target = np.zeros((1000, model.output_dim))
    zf_target[:, 3] = -20. * np.sin(th) - 5.5
    zf_target[:, 4] = 10. * np.sin(2 * th) + 1.5
    target.z = model.zfyf_to_zy(zf=zf_target)

    cost_params = QuadraticCost()
    cost_params.R = .00001 * np.eye(model.input_dim)
    cost_params.Q = np.zeros((model.output_dim, model.output_dim))
    cost_params.Q[3, 3] = 100  # corresponding to x position of end effector
    cost_params.Q[4, 4] = 100  # corresponding to y position of end effector
    cost_params.Q[5, 5] = 0.0

    planning_horizon = 5

    # Control constraints
    U = HyperRectangle([6000., 6000., 6000., 6000.], [0., 0., 0., 0.])

    # State constraints
    # Building A matrix
    Hz = np.zeros((2, 6))
    Hz[0, 3] = 1
    Hz[1, 4] = 1
    H = Hz @ model.H
    H_full = np.vstack([-H, H])
    b_z_lb = np.array([-17.5 - 5.5, -20 + 1.5])
    b_z_ub = np.array([17.5 - 5.5, 20 + 1.5])
    offset = Hz @ model.z_ref
    b_z = np.hstack([-(b_z_lb - offset), b_z_ub - offset])
    X = Polyhedron(A=H_full, b=b_z, with_reproject=True)

    # Define GuSTO model
    runMPCSolverNode(model=model, N=planning_horizon, dt=dt, cost_params=cost_params, target=target, U=U, X=X,
                     verbose=1, warm_start=True, solver='GUROBI')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'generate_linearized_rom':
        generate_linearized_ROM()
    elif sys.argv[1] == 'run_rompc_solver':
        run_rompc_solver()
    else:
        raise RuntimeError('Not a valid command line argument')
