"""
This file defines the parameters for the Diamond robot FEM
model that corresond to the physical hardware
"""

M = 0.45 # total mass, kg # measured as .424
E = 175 # Young's modulus, MPa # at 175
NU = .45 # Poisson's ratio
ALPHA = 2.5 # Rayleigh mass
BETA = 0.01 # Rayleigh stiffness
U_MAX = 1500 # mN
DT = 0.01

def diamondRobot():
    from robots import environments
    robot = environments.Diamond(totalMass=M, 
                                 poissonRatio=NU, 
                                 youngModulus=E, 
                                 rayleighMass=ALPHA, 
                                 rayleighStiffness=BETA,
                                 dt=DT)

    # Add open loop input sequences
    from sofacontrol.open_loop_sequences import DiamondRobotSequences
    import numpy as np
    robot.sequences = DiamondRobotSequences(dt=DT, t0=1.)
    robot.sequences.umax = np.array([U_MAX, U_MAX, U_MAX, U_MAX])

    return robot