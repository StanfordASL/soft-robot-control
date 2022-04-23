from os.path import dirname, abspath, join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from sofacontrol.utils import load_data

path = dirname(abspath(__file__))

# Load SCP data
decay_simdata_file = join(path, 'decay_snapshots.pkl')
decay_data = load_data(decay_simdata_file)
t_decay = decay_data['t']
z_scp = decay_data['z']
solve_times_scp = decay_data['info']['solve_times']
real_time_limit_scp = decay_data['info']['rollout_time']
