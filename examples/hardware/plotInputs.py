from os.path import dirname, abspath, join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pdb

from sofacontrol.utils import load_data

path = dirname(abspath(__file__))

#############################################
# Problem 1, Figure 8 with constraints
#############################################
# M = 3
# T = 10
# N = 500
# t_target = np.linspace(0, M*T, M*N)
# th = np.linspace(0, M * 2 * np.pi, M*N)
# zf_target = np.zeros((M*N, 6))
# zf_target[:, 3] = -40. * np.sin(th) - 7.1
# zf_target[:, 4] = 40. * np.sin(2 * th)
# y_ub = 5
# name = 'figure8_inputs'

##############################################
# Problem 2, Circle on side
##############################################
M = 3
T = 5
N = 1000
r = 10
t_target = np.linspace(0, M*T, M*N)
th = np.linspace(0, M*2*np.pi, M*N)
x_target = np.zeros(M*N)
y_target = r * np.sin(th)
z_target = r - r * np.cos(th) + 107
zf_target = np.zeros((M*N, 6))
zf_target[:, 3] = x_target
zf_target[:, 4] = y_target
zf_target[:, 5] = z_target
name = 'circle_inputs'


# Load SCP data
scp_simdata_file = join(path, 'scp_CL_sim.pkl')
scp_data = load_data(scp_simdata_file)
idx = np.argwhere(scp_data['t'] >= 3)[0][0]
u_scp = scp_data['u'][idx:, :]
#x_scp = scp_data['x']
#q_scp = scp_data['q']
t_scp = scp_data['t'][idx:] - scp_data['t'][idx]
z_scp = scp_data['z'][idx:, :]
#zhat = scp_data['z_hat'][idx:, :]
solve_times_ssm = scp_data['info']['solve_times']
real_time_limit_ssm = scp_data['info']['rollout_time']

z_opt_rollout = scp_data['info']['z_rollout']
t_opt_rollout = scp_data['info']['t_rollout']

# Load iLQR data
ilqr_simdata_file = join(path, 'scp_sim.pkl')
ilqr_data = load_data(ilqr_simdata_file)
idx = np.argwhere(ilqr_data['t'] >= 3)[0][0]
t_ilqr = ilqr_data['t'][idx:] - ilqr_data['t'][idx]
z_ilqr = ilqr_data['z'][idx:, :]
u_ilqr = ilqr_data['u'][idx:, :]
solve_times_tpwl = ilqr_data['info']['solve_times']
real_time_limit_tpwl = ilqr_data['info']['rollout_time']

# Load Koopman data
koop_simdata_file = join(path, 'koopman_sim.pkl')
koop_data = load_data(koop_simdata_file)
idx = np.argwhere(koop_data['t'] >= 3)[0][0]
t_koop = koop_data['t'][idx:] - koop_data['t'][idx]
z_koop = koop_data['z'][idx:, :]
u_koop = koop_data['u'][idx:, :]
solve_times_koop = koop_data['info']['solve_times']
real_time_limit_koop = koop_data['info']['rollout_time']
t_opt = koop_data['info']['t_opt']
# z_opt_rollout = koop_data['info']['z_rollout']
# t_opt_rollout = koop_data['info']['t_rollout']

plot_rollouts = False
m_w = 30
##################################################
# Plot trajectory as function of time
##################################################
fig2 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax2 = fig2.add_subplot(411)

ax2.plot(t_ilqr, u_ilqr[:, 0], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
ax2.plot(t_koop, u_koop[:, 0], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
ax2.plot(t_scp, u_scp[:, 0], 'tab:blue', label='SSM CL', linewidth=3)
plt.ylabel(r'$u_1$', fontsize=14)

ax2.set_xlim([0, 10])
plt.legend(loc='best', prop={'size': 14})

ax3 = fig2.add_subplot(412)
ax3.plot(t_ilqr, u_ilqr[:, 1], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
ax3.plot(t_koop, u_koop[:, 1], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
ax3.plot(t_scp, u_scp[:, 1], 'tab:blue', label='SSM CL', linewidth=3)
plt.ylabel(r'$u_2$', fontsize=14)
ax3.set_xlim([0, 10])

ax4 = fig2.add_subplot(413)
ax4.plot(t_ilqr, u_ilqr[:, 2], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
ax4.plot(t_koop, u_koop[:, 2], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
ax4.plot(t_scp, u_scp[:, 2], 'tab:blue', label='SSM CL', linewidth=3)
plt.ylabel(r'$u_3$', fontsize=14)
ax4.set_xlim([0, 10])


ax5 = fig2.add_subplot(414)
ax5.plot(t_ilqr, u_ilqr[:, 3], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
ax5.plot(t_koop, u_koop[:, 3], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
ax5.plot(t_scp, u_scp[:, 3], 'tab:blue', label='SSM CL', linewidth=3)
plt.ylabel(r'$u_4$', fontsize=14)

ax5.set_xlim([0, 10])

plt.xlabel(r'$t$ [s]', fontsize=14)
figure_file = join(path, name + '.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')


# MSE calculations
# Calculation of desired trajectory
if name == 'figure8':
    zf_desired = zf_target.copy()
    # zf_desired[:, 4] = np.minimum(y_ub, zf_target[:,4])
else:
    zf_desired = zf_target.copy()

f = interp1d(t_target, zf_desired, axis=0)
zd_koop = f(t_koop)
zd_scp = f(t_scp)
zd_rompc = f(t_ilqr)

if name == 'figure8':
    err_koop = (z_koop - zd_koop)[:,3:5]
    err_scp = (z_scp - zd_scp)[:,3:5]
    err_ilqr = (z_ilqr - zd_rompc)[:,3:5]
else:
    err_koop = (z_koop - zd_koop)[:,4:6]
    err_scp = (z_scp - zd_scp)[:,4:6]
    err_ilqr = (z_ilqr - zd_rompc)[:,4:6]

# inner norm gives euclidean distance, outer norm squared / nbr_samples gives MSE
mse_koop = np.linalg.norm(np.linalg.norm(err_koop, axis=1))**2 / err_koop.shape[0]
mse_rompc = np.linalg.norm(np.linalg.norm(err_ilqr, axis=1))**2 / err_ilqr.shape[0]
mse_scp = np.linalg.norm(np.linalg.norm(err_scp, axis=1))**2 / err_scp.shape[0]

print('------ Mean Squared Errors (MSEs)----------')
print('Ours (SSM CL): {}'.format(mse_scp))
print('Koopman CL: {}'.format(mse_koop))
print('TPWL CL: {}'.format(mse_rompc))

print('-------------Solve times ---------------')
print('TPWL: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_tpwl), np.mean(solve_times_tpwl),
                                                     np.max(solve_times_tpwl)))

print('Koopman: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_koop), np.mean(solve_times_koop),
                                                        np.max(solve_times_koop)))

print('Ours (SSM): Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_ssm), np.mean(solve_times_ssm),
                                                     np.max(solve_times_ssm)))

plt.show()