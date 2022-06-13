import os
from os import listdir
from os.path import dirname, abspath, join, isfile
from pathlib import Path
from sofacontrol.utils import qv2x
from sofacontrol.measurement_models import linearModel

import numpy as np
from tqdm.auto import tqdm

from sofacontrol.utils import load_data

current_dir = dirname(abspath(__file__))
data_dir = "/dataCollection/"
path = current_dir + data_dir
csvPath = path + "csv/"

outputNode = 1354

rest_file = join(current_dir, 'rest_qv.pkl')
scp_simdata_file = join(current_dir, 'scp_sim.pkl')
scp_data = load_data(scp_simdata_file)

rest_data = load_data(rest_file)
#q_decay = decay_data['q']
qv_equilibrium = np.array(rest_data['rest'])

dt = 0.01
Tstart = 3.
skip = int(Tstart / dt)
zq = np.array(scp_data['z'])[skip:]
u = np.array(scp_data['u'][skip:])

x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
outputModel = linearModel([outputNode], 1628)

z_equilibrium = outputModel.evaluate(x_eq, qv=True)


print('Done')