from os.path import dirname, abspath, join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from sofacontrol.utils import load_data

path = dirname(abspath(__file__))

# Load SCP data
decay_simdata_file = join(path, 'decay_snapshots.pkl')
rest_file = join(path, 'rest.pkl')

decay_data = load_data(decay_simdata_file)
rest_data = load_data(rest_file)
t_decay = decay_data['t']
q_decay = decay_data['q']
q_equilibrium = rest_data['rest']


q_shift_decay = np.array(q_decay - q_equilibrium)
v_decay = np.array(decay_data['v'])

np.savetxt("q_small.csv", q_shift_decay)
np.savetxt("v_small.csv", v_decay)

# Getting initial state to infer mode
dt = 0.001
t0 = 2.0
idx_start = int(t0 / dt)
q0 = q_shift_decay[idx_start-1, :]

print('Done')
