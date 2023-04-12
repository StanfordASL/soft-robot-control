import sys
from os.path import dirname, abspath, join

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import casadi as ca
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from sofacontrol.utils import load_data
from sofacontrol.SSM import ssm
from sofacontrol.utils import qv2x, vq2qv
from sofacontrol.measurement_models import linearModel

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

pathToModel = path + '/../SSMmodels/'
pathToSimData = path + '/../'

# Load SSM model
TIP_NODE = 1354
rest_file = join(pathToSimData, 'rest_qv.pkl') # Load equilibrium point
rest_data = load_data(rest_file)
qv_equilibrium = np.array(rest_data['rest'])
x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
outputModel = linearModel([TIP_NODE], 1628, vel=True)
z_eq_point = outputModel.evaluate(x_eq, qv=True)

SSM_data = loadmat(join(pathToModel, 'SSM_model.mat'))['py_data'][0, 0]
raw_model = SSM_data['model']
raw_params = SSM_data['params']

model = ssm.SSMDynamics(z_eq_point, discrete=False, discr_method='be',
                            model=raw_model, params=raw_params, C=None)

num_nodes = 1628
ee_node = [1354]

# Load simulation data
ssm_simdata_file = join(pathToSimData, 'pod_snapshots.pkl')
ssm_data = load_data(ssm_simdata_file)
koop_measurement = linearModel(nodes=ee_node, num_nodes=num_nodes, pos=True, vel=True)
state = qv2x(q=ssm_data['q'], v=ssm_data['v'])

idx = 0
z = koop_measurement.evaluate(x=state.T)
z_ssm = vq2qv(z.T)

xr = np.array([model.compute_RO_state(z_ssm[i]) for i in range(z_ssm.shape[0])])
wz = model.W_map

# Shuffle xr
np.random.shuffle(xr)

LwzVals = []
while xr.shape[0] > 0:
    if xr.shape[0] == 1:
        xr = np.array([])
    else:
        xr_1, xr_2 = xr[0], xr[1]
        xr = xr[2:]
        LwzVals.append(np.linalg.norm(wz(xr_1) - wz(xr_2)) / np.linalg.norm(xr_1 - xr_2))

Lwz = max(LwzVals)
print(Lwz)