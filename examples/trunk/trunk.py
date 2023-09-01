import sys
from os.path import dirname, abspath, join, split

import numpy as np
from matplotlib import pyplot as plt

import pickle

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from sofacontrol.open_loop_sequences import TrunkRobotSequences

# Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51


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

    ALPHA = 0.1

    path = "/media/jonas/Backup Plus/jonas_soft_robot_data/autonomous_ASSM_tests"
    with open(join(path, "OL_sim_perturbed", f"alpha={ALPHA:.1f}", 'u_perturbed.pkl'), 'rb') as f:
    # with open(join(path, "OL_sim_perturbed" 'u_perturbed.pkl'), 'rb') as f:
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
    prob.opt['sim_duration'] = len(t) * dt
    prob.snapshots_dir = join(path, "OL_sim_perturbed", f"alpha={ALPHA:.1f}")
    prob.opt['save_prefix'] = 'OL_sim'

    return prob



def apply_constant_input(input, pre_tensioning, q0=None, t0=0.0, save_data=True, filepath=f"{path}/undef_traj"):
    """
    In problem_specification add:

    import examples.trunk as trunk
    problem = trunk.apply_constant_input():

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


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Include command line arguments')
    elif sys.argv[1] == 'compute_POD_basis':
        compute_POD_basis()
    elif sys.argv[1] == 'run_gusto_solver':
        run_gusto_solver()
    else:
        raise RuntimeError('Not a valid function argument')
