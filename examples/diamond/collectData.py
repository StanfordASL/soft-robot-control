import os
from os import listdir
from os.path import dirname, abspath, join, isfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from sofacontrol.utils import load_data

current_dir = dirname(abspath(__file__))
data_dir = "/dataCollection/"
path = current_dir + data_dir
allDataFiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for decay_file in tqdm(allDataFiles):
    rest_file = join(current_dir, 'rest.pkl')

    decay_data = load_data(decay_file)
    rest_data = load_data(rest_file)
    t_decay = decay_data['t']
    q_decay = decay_data['q']
    q_equilibrium = rest_data['rest']

    q_shift_decay = np.array(q_decay - q_equilibrium)
    v_decay = np.array(decay_data['v'])

    q_decay_filename = path + "csv/" + Path(decay_file).stem + "_q" + ".csv"
    v_decay_filename = path + "csv/" + Path(decay_file).stem + "_v" + ".csv"
    # Saves as (N, n) where N is length of simulation and n is state dim
    np.savetxt(q_decay_filename, q_shift_decay)
    np.savetxt(v_decay_filename, v_decay)

print('Done')