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
csvPath = path + "csv/"

allDataFiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
allCSVFiles = [os.path.join(csvPath, f) for f in os.listdir(csvPath) if os.path.isfile(os.path.join(csvPath, f))]

outputNode = 1354
option = int(input("Select '1' to convert pkl to csv or '2' to consolidate full system trajectory data \n"))

sample_rate = 1

if option == 1:
    for decay_file in tqdm(allDataFiles):
        rest_file = join(current_dir, 'rest.pkl')

        decay_data = load_data(decay_file)
        rest_data = load_data(rest_file)
        t_decay = decay_data['t']
        q_decay = decay_data['q'][::sample_rate]
        #q_decay = decay_data['q']
        q_equilibrium = rest_data['rest']

        q_shift_decay = np.array(q_decay - q_equilibrium)
        v_decay = np.array(decay_data['v'][::sample_rate])
        #v_decay = np.array(decay_data['v'])

        # Retrieve output node position (zq) and velocity (zv) trajectories
        zq_shift_decay = q_shift_decay[:, 3*outputNode:3*outputNode+3]
        zv_decay = v_decay[:, 3*outputNode:3*outputNode+3]

        #output_decay = np.hstack((zq_shift_decay, zv_decay))
        output_decay = np.hstack((q_shift_decay, v_decay))
        #output_decay = q_shift_decay

        #q_decay_filename = path + "csv/" + Path(decay_file).stem + "_q" + ".csv"
        #v_decay_filename = path + "csv/" + Path(decay_file).stem + "_v" + ".csv"

        # Saves as (N, n) where N is length of simulation and n is state dim
        # Currently saving output node only (x, y, z)
        #np.savetxt(q_decay_filename, zq_shift_decay)
        #np.savetxt(v_decay_filename, zv_decay)

        decay_filename = path + "csv/" + Path(decay_file).stem + "_output" + ".csv"
        np.savetxt(decay_filename, output_decay)
else:
    traj_data = []
    for traj_file in tqdm(allCSVFiles):
        traj_data.append(np.loadtxt(traj_file))

    traj_data = np.array(traj_data)
    traj_shape = np.shape(traj_data)
    traj_data = traj_data.reshape(traj_shape[0]*traj_shape[1], traj_shape[2])


print('Done')