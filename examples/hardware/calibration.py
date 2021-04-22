import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

from examples import Problem
from examples.hardware.model import diamondRobot


def output_node_calibration():
    """
    In problem_specification add:

    from examples.hardware import calibration
    problem = calibration.output_node_calibration

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py

    This function runs a Sofa simulation and saves data that can be used to determine 
    which nodes should be used in the output models.
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController

    # Define an open loop control sequence
    u, save, t = prob.Robot.sequences.constant_input(u_constant=np.zeros(4), t=1.0, add_base=False)
    save[0] = True # save the first step only

    prob.controller = OpenLoop(u.shape[0], t, u, save)

    prob.snapshots = SnapshotData(save_dynamics=False)

    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'output_node_calibration'

    return prob

def rest_calibration():
    """
    In problem_specification add:

    from examples.hardware import calibration
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
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController

    # Define an open loop control sequence
    u, save, t = prob.Robot.sequences.constant_input(u_constant=np.zeros(4), t=2.0, add_base=False)
    save[-1] = True # save the last step only

    prob.controller = OpenLoop(u.shape[0], t, u, save)

    prob.snapshots = SnapshotData(save_dynamics=False)

    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'rest_calibration'

    return prob

def model_calibration():
    """
    In problem_specification add:

    from examples.hardware import calibration
    problem = calibration.model_calibration

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py

    This function runs a Sofa simulation and saves data that can be used to compare
    against the real hardware to model accuracy
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController

    # Define an open loop control sequence
    val = 1500
    u0 = np.array([0, 0, 0, 0])
    u0, save0, t0 = prob.Robot.sequences.constant_input(u_constant=u0, t=1.0, add_base=False)
    save0 += True
    u1 = np.array([val, 0, 0, 0])
    u1, save1, t1 = prob.Robot.sequences.constant_input(u_constant=u1, t=2.0, add_base=False)
    save1 += True
    u2 = np.array([0, 0, 0, 0])
    u2, save2, t2 = prob.Robot.sequences.constant_input(u_constant=u2, t=1.0, add_base=False)
    save2 += True
    u3 = np.array([0, val, 0, 0])
    u3, save3, t3 = prob.Robot.sequences.constant_input(u_constant=u3, t=2.0, add_base=False)
    save3 += True
    u4 = np.array([0, 0, 0, 0])
    u4, save4, t4 = prob.Robot.sequences.constant_input(u_constant=u4, t=1.0, add_base=False)
    save4 += True
    u5 = np.array([0, 0, val, 0])
    u5, save5, t5 = prob.Robot.sequences.constant_input(u_constant=u5, t=2.0, add_base=False)
    save5 += True
    u6 = np.array([0, 0, 0, 0])
    u6, save6, t6 = prob.Robot.sequences.constant_input(u_constant=u6, t=1.0, add_base=False)
    save6 += True
    u7 = np.array([0, 0, 0, val])
    u7, save7, t7 = prob.Robot.sequences.constant_input(u_constant=u7, t=2.0, add_base=False)
    save7 += True
    u8 = np.array([0, 0, 0, 0])
    u8, save8, t8 = prob.Robot.sequences.constant_input(u_constant=u8, t=1.0, add_base=False)
    save8 += True

    u, save, t = prob.Robot.sequences.combined_sequence([u0, u1, u2, u3, u4, u5, u6, u7, u8], 
        [save0, save1, save2, save3, save4, save5, save6, save7, save8], 
        [t0, t1, t2, t3, t4, t5, t6, t7, t8])

    # Add some sine wave components
    u1 = np.array([val, 0, 0, 0])
    u1, save1, t1 = prob.Robot.sequences.sine_input(u_max=u1, t=2.0, add_base=False)
    save1 += True
    u2 = np.array([0, val, 0, 0])
    u2, save2, t2 = prob.Robot.sequences.sine_input(u_max=u2, t=2.0, add_base=False)
    save2 += True
    u3 = np.array([0, 0, val, 0])
    u3, save3, t3 = prob.Robot.sequences.sine_input(u_max=u3, t=2.0, add_base=False)
    save3 += True
    u4 = np.array([0, 0, 0, val])
    u4, save4, t4 = prob.Robot.sequences.sine_input(u_max=u4, t=2.0, add_base=False)
    save4 += True

    u, save, t = prob.Robot.sequences.combined_sequence([u, u1, u2, u3, u4], 
                                        [save, save1, save2, save3, save4], 
                                        [t, t1, t2, t3, t4])

    prob.controller = OpenLoop(u.shape[0], t, u, save)

    prob.snapshots = SnapshotData(save_dynamics=False)

    # Save simulation data
    prob.opt['sim_duration'] = 13
    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'model_calibration'

    return prob


def actuator_calibration():
    """
    In problem_specification add:

    from examples.hardware import calibration
    problem = calibration.actuator_calibration

    then run:

    $SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py

    or

    python3 launch_sofa.py
    """
    from sofacontrol.open_loop_controller import OpenLoopController, OpenLoop
    from sofacontrol.utils import SnapshotData

    prob = Problem()
    prob.Robot = diamondRobot()
    prob.ControllerClass = OpenLoopController

    # Define an open loop control sequence
    max_val = 1500
    u0 = np.array([0, 0, 0, 0])
    u0, save0, t0 = prob.Robot.sequences.constant_input(u_constant=u0, t=1.0, add_base=False)
    save0 += True
    u1 = np.array([0.25*max_val, 0, 0, 0])
    u1, save1, t1 = prob.Robot.sequences.constant_input(u_constant=u1, t=2.0, add_base=False)
    save1 += True
    u2 = np.array([0, 0, 0, 0])
    u2, save2, t2 = prob.Robot.sequences.constant_input(u_constant=u2, t=1.0, add_base=False)
    save2 += True
    u3 = np.array([0.5*max_val, 0, 0, 0])
    u3, save3, t3 = prob.Robot.sequences.constant_input(u_constant=u3, t=2.0, add_base=False)
    save3 += True
    u4 = np.array([0, 0, 0, 0])
    u4, save4, t4 = prob.Robot.sequences.constant_input(u_constant=u4, t=1.0, add_base=False)
    save4 += True
    u5 = np.array([0.75*max_val, 0, 0, 0])
    u5, save5, t5 = prob.Robot.sequences.constant_input(u_constant=u5, t=2.0, add_base=False)
    save5 += True
    u6 = np.array([0, 0, 0, 0])
    u6, save6, t6 = prob.Robot.sequences.constant_input(u_constant=u6, t=1.0, add_base=False)
    save6 += True
    u7 = np.array([max_val, 0, 0, 0])
    u7, save7, t7 = prob.Robot.sequences.constant_input(u_constant=u7, t=2.0, add_base=False)
    save7 += True
    u8 = np.array([0, 0, 0, 0])
    u8, save8, t8 = prob.Robot.sequences.constant_input(u_constant=u8, t=1.0, add_base=False)
    save8 += True

    u, save, t = prob.Robot.sequences.combined_sequence([u0, u1, u2, u3, u4, u5, u6, u7, u8], 
        [save0, save1, save2, save3, save4, save5, save6, save7, save8], 
        [t0, t1, t2, t3, t4, t5, t6, t7, t8])

    prob.controller = OpenLoop(u.shape[0], t, u, save)

    prob.snapshots = SnapshotData(save_dynamics=False)

    # Save simulation data
    prob.opt['sim_duration'] = 13
    prob.snapshots_dir = path
    prob.opt['save_prefix'] = 'actuator_calibration'

    return prob