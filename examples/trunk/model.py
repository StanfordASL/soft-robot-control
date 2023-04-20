"""
This file defines the parameters for the Trunk robot FEM
"""
import pickle
from os.path import dirname, abspath, join
import numpy as np


PATH = dirname(abspath(__file__))

U_MAX = 800 # mN
DT = 0.01

TIP_NODE = 51
N_NODES = 709

# equilibrium position after gravity
with open(join(PATH, 'rest_qv.pkl'), 'rb') as file:
    QV_EQUILIBRIUM =  [pickle.load(file)['q'][0], np.zeros(3*N_NODES)]


def trunkRobot(q0=None, scale_mode=1000, dt=DT):
    from robots import environments
    robot = environments.Trunk()
    # Add open loop input sequences
    from sofacontrol.open_loop_sequences import TrunkRobotSequences
    import numpy as np
    robot.sequences = TrunkRobotSequences(dt=dt, t0=1.)
    robot.sequences.umax = np.repeat(U_MAX, 8)

    return robot