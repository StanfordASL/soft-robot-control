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

def tubeDynamics(x, u, L_n, L_r, B_ub):
    """
    :x: (n_x,) array of state variables. (Decision variable)
    x[0] represents the error orthogonal dynamics
    x[1] represents the on-manifold error dynamics
    :u: float scalar of input.
    :lam_n: float scalar. Leading eigenvalue of orthogonal dynamics
    :lam_r: float scalar. Leading eigenvalue of manifold dynamics
    :L_n: float scalar. Lipschitz constant of fnl (Decision variable)
    :L_r: float scalar. Lipschitz constant of rnl (Decision variable)
    :B_ub: float scalar. Upperbound on input influence term (Decision variable)
    """
    lam_n = 14.982
    lam_r = 3.516
    xdot = np.array([-(lam_n - L_n) * x[0] + B_ub * u,
                      -(lam_r - L_r) * x[1] + L_n * x[0]])
    return xdot

def tubeDynamicsCasadi(x, u, L_n, L_r, B_ub):
    """
    :x: (n_x,) array of state variables. (Decision variable)
    x[0] represents the error orthogonal dynamics
    x[1] represents the on-manifold error dynamics
    :u: float scalar of input.
    :lam_n: float scalar. Leading eigenvalue of orthogonal dynamics
    :lam_r: float scalar. Leading eigenvalue of manifold dynamics
    :L_n: float scalar. Lipschitz constant of fnl (Decision variable)
    :L_r: float scalar. Lipschitz constant of rnl (Decision variable)
    :B_ub: float scalar. Upperbound on input influence term (Decision variable)
    """
    lam_n = 14.982
    lam_r = 3.516
    xdot = ca.vertcat(-(lam_n - L_n) * x[0] + B_ub * u,
                      -(lam_r - L_r) * x[1] + L_n * x[0])
    return xdot

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

# Load simulation data
ssm_simdata_file = join(pathToSimData, 'scp_OL_SSM_sim_step.pkl')
ssm_data = load_data(ssm_simdata_file)
idx = np.argwhere(ssm_data['t'] >= 3)[0][0]
u_ssm = ssm_data['u'][idx:, :]
t_ssm = ssm_data['t'][idx:] - ssm_data['t'][idx]
z_ssm = ssm_data['z'][idx:, :]
# Flip to qv
z_ssm = vq2qv(z_ssm)

# Let's check the model
T = t_ssm[-1]
dt = t_ssm[1] - t_ssm[0]    # Sampling time
N = int(T / dt)             # Prediction Horizon

t_interp = np.linspace(0, T, N+1)
u_interp = interp1d(t_ssm, u_ssm, axis=0)(t_interp)
x0 = np.zeros((model.state_dim,))   # Initial state
x_traj, z_traj = model.rollout(x0, u_interp, dt)

fig = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax = fig.add_subplot(211)
ax.plot(t_ssm, z_ssm[:, 0], 'tab:blue', label='True System', linewidth=3)
ax.plot(t_ssm, z_traj[:-1, 0], 'tab:orange', linestyle='--', label='SSM Model', linewidth=3)

ax2 = fig.add_subplot(212)
ax2.plot(t_ssm, z_ssm[:, 1], 'tab:blue', label='True System', linewidth=3)
ax2.plot(t_ssm, z_traj[:-1, 1], 'tab:orange', linestyle='--', label='SSM Model', linewidth=3)

plt.legend()
plt.show()

# Define the data
xr = x_traj[:-1, :]
zr = np.array([model.compute_RO_state(z_ssm[i]) for i in range(N+1)])
# delta_r = np.linalg.norm(xr - zr, axis=1)

# Try smoothing delta_r
# TODO: This is sort of a hack... figure out if spiking error makes sense
from scipy.signal import savgol_filter
delta_r = savgol_filter(np.linalg.norm(xr - zr, axis=1), 101, 3)
delta_r = np.concatenate((np.zeros(300), delta_r[300:]))

# TODO: Use 100 for figure 8 trajectory
# Define tube dynamics parameters
u_abs = np.linalg.norm(u_ssm, axis=1)
u_abs_interp = interp1d(t_ssm, u_abs, axis=0)
s0 = np.array([0.0, np.linalg.norm(xr[0, :] - zr[0, :])])     # Initial state of tube dynamics

# Integrate tube dynamics
# L_n = 5.269
# L_r = 3.471
# B_err = 0.106
L_n = 4.
L_r = 2.
B_err = 7.
tubeDyn = lambda t, x: tubeDynamics(x, u_abs_interp(t), L_n, L_r, B_err)
sol = solve_ivp(tubeDyn, [0., t_ssm[-1]], s0, t_eval=t_interp)
error_states = sol.y

# Plot errors
fig2 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax = fig2.add_subplot(111)
ax.plot(t_ssm, error_states[0, :], 'tab:blue', label='s', linewidth=3)
ax.plot(t_ssm, error_states[1, :], 'tab:orange', label=r'$\delta$', linewidth=3)
ax.plot(t_ssm, delta_r, 'tab:green', label=r'True $\delta$', linewidth=3)
plt.legend()
plt.show()

# print('Done')

# Define Casadi symbolic variables
x_sym = ca.MX.sym('x_sym', 2)
u_sym = ca.MX.sym('u_sym', 4)

# Define ODE
f = ca.Function('f', [x_sym, u_sym], [tubeDynamicsCasadi(x_sym, u_sym[0], u_sym[1], u_sym[2], u_sym[3])])


# Define casadi time integration
intg_options = {}
intg_options['tf'] = t_ssm[1] - t_ssm[0]
intg_options['simplify'] = True
intg_options['number_of_finite_elements'] = 4

dae = {}
dae['x'] = x_sym
dae['p'] = u_sym
dae['ode'] = f(x_sym, u_sym)

# Define symbolic integrator
intg = ca.integrator('intg', 'collocation', dae, intg_options)

intg_result = intg(x0=x_sym, p=u_sym)
x_next_sym = intg_result['xf']
F = ca.Function('F', [x_sym, u_sym], [x_next_sym])

opti = ca.Opti()

# Define decision variables
X = opti.variable(2, N + 1)
L_n = opti.variable(1)
L_r = opti.variable(1)
B_ub = opti.variable(1)

# Define parameters
u_param = opti.parameter(1, N)
delta_r_param = opti.parameter(1, N + 1)
x0_param = opti.parameter(2)

J = 0
# Define constraints: x, u, lam_n, lam_r, L_n, L_r, B_ub
for k in range(N):
    opti.subject_to(X[:,k+1] == F(X[:, k], ca.vertcat(u_param[k], L_n, L_r, B_ub)))
    # Define objective function (need to scale)
    J += ca.mtimes(X[:, k].T, X[:, k])/np.max(delta_r)

# Additional constraints
opti.subject_to(delta_r_param <= X[1, :])
opti.subject_to(0 <= X[1, :])
opti.subject_to(X[:, 0] == x0_param)

opti.subject_to(0 <= L_n)
# opti.subject_to(L_n <= 14.)

opti.subject_to(0 <= L_r)
# opti.subject_to(L_r <= 3.5)

opti.subject_to(0 <= B_ub)

opti.minimize(J)

# Set parameter values
opti.set_value(u_param, u_abs[1:].T)
opti.set_value(delta_r_param, delta_r.T)
opti.set_value(x0_param, s0)

# Initial guess
Ln0 = 4.
Lr0 = 2.
Bnorm0 = 7.

opti.set_initial(L_n, Ln0)
opti.set_initial(L_r, Lr0)
opti.set_initial(B_ub, Bnorm0)

# Define the solver options
options = {
    "ipopt": {
        # "max_iter": 100,
        "print_level": 5,
        "acceptable_tol": 1e-3,
        "acceptable_obj_change_tol": 1e-4,
    }
}
opti.solver("ipopt", options)
sol = opti.solve()

# Plot optimal coefficients
Ln = sol.value(L_n)
Lr = sol.value(L_r)
Berr = sol.value(B_ub)

print('L_n: ', Ln)
print('L_r: ', Lr)
print('B_norm: ', Berr)

tubeDyn = lambda t, x: tubeDynamics(x, u_abs_interp(t), Ln, Lr, Berr)
sol = solve_ivp(tubeDyn, [0., t_ssm[-1]], s0, t_eval=t_interp)
error_states = sol.y

# Plot errors
fig2 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax = fig2.add_subplot(111)
ax.plot(t_ssm, error_states[0, :], 'tab:blue', label='s', linewidth=3)
ax.plot(t_ssm, error_states[1, :], 'tab:orange', label=r'$\delta$', linewidth=3)
ax.plot(t_ssm, delta_r, 'tab:green', label=r'True $\delta$', linewidth=3)
plt.legend()
plt.show()


print('Done')