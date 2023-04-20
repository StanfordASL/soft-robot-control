import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.trunk.model import trunkRobot


def rest_calibration():
    """
    In problem_specification add:

    from examples.trunk import calibration
    problem = calibration.rest_calibration

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py

    This function saves a snapshot at the steady state with no inputs.
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = trunkRobot()
    prob.ControllerClass = OpenLoopController

    # Define an open loop control sequence
    u, save, t = prob.Robot.sequences.constant_input(u_constant=np.zeros(8), t=2.0, add_base=False)
    save[-1] = True # save the last step only

    prob.controller = OpenLoop(u.shape[0], t, u, save)

    prob.snapshots = SnapshotData(save_dynamics=False)

    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'rest'

    return prob