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
M = 3
T = 10
N = 500
t_target = np.linspace(0, M*T, M*N)
th = np.linspace(0, M * 2 * np.pi, M*N)
zf_target = np.zeros((M*N, 6))
zf_target[:, 3] = -20. * np.sin(th)
zf_target[:, 4] = 20. * np.sin(2 * th)
name = 'figure8'

##############################################
# Problem 2, Circle on side
##############################################
# M = 3
# T = 5
# N = 1000
# r = 10
# t_target = np.linspace(0, M*T, M*N)
# th = np.linspace(0, M*2*np.pi, M*N)
# x_target = np.zeros(M*N)
# y_target = r * np.sin(th)
# z_target = r - r * np.cos(th) + 107
# zf_target = np.zeros((M*N, 6))
# zf_target[:, 3] = x_target
# zf_target[:, 4] = y_target
# zf_target[:, 5] = z_target
# name = 'circle'


# Load SCP data
scp_simdata_file = join(path, 'scp_sim.pkl')
scp_data = load_data(scp_simdata_file)
idx = np.argwhere(scp_data['t'] >= 3)[0][0]
u_scp = scp_data['u'][idx:, :]
x_scp = scp_data['x']
#q_scp = scp_data['q']
t_scp = scp_data['t'][idx:] - scp_data['t'][idx]
z_scp = scp_data['z'][idx:, :]
zhat = scp_data['z_hat'][idx:, :]
solve_times_scp = scp_data['info']['solve_times']
real_time_limit_scp = scp_data['info']['rollout_time']

# Load iLQR data
# ilqr_simdata_file = join(path, 'ilqr_sim.pkl')
# ilqr_data = load_data(ilqr_simdata_file)
# idx = np.argwhere(ilqr_data['t'] >= 3)[0][0]
# t_ilqr = ilqr_data['t'][idx:] - ilqr_data['t'][idx]
# z_ilqr = ilqr_data['z'][idx:, :]
# u_ilqr = ilqr_data['u'][idx:, :]

# Load Koopman data
koop_simdata_file = join(path, 'koopman_sim.pkl')
koop_data = load_data(koop_simdata_file)
idx = np.argwhere(koop_data['t'] >= 3)[0][0]
t_koop = koop_data['t'][idx:] - koop_data['t'][idx]
z_koop = koop_data['z'][idx:, :]
solve_times_koop = koop_data['info']['solve_times']
real_time_limit_koop = koop_data['info']['rollout_time']

m_w = 30
##################################################
# Plot trajectory as function of time
##################################################
fig2 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax2 = fig2.add_subplot(211)

if name == 'figure8':
    #ax2.plot(t_ilqr, z_ilqr[:, 3], 'tab:green', marker='x', markevery=m_w, label='iLQR (Open Loop)', linewidth=1)
    ax2.plot(t_koop, z_koop[:, 3], 'tab:orange', marker='^', markevery=m_w, label='Koopman MPC', linewidth=1)
    ax2.plot(t_scp, z_scp[:, 3], 'tab:blue', label='Nonlinear ROMPC', linewidth=3)
    ax2.plot(t_target, zf_target[:, 3], '--k', alpha=1, linewidth=1, label='Target')
    plt.ylabel(r'$x_{ee}$ [mm]', fontsize=14)
else:
    # ax2.plot(t_rompc, z_rompc[:, 4], 'tab:green', marker='x', markevery=m_w, label='Linear ROMPC', linewidth=1)
    # ax2.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=m_w, label='Koopman MPC', linewidth=1)
    # ax2.plot(t_scp, z_scp[:, 4], 'tab:blue', label='Nonlinear ROMPC', linewidth=3)
    ax2.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
ax2.set_xlim([0, 10])
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.legend(loc='best', prop={'size': 14})

ax3 = fig2.add_subplot(212)
if name == 'figure8':
    #ax3.plot(t_ilqr, z_ilqr[:, 4], 'tab:green', marker='x', markevery=m_w, label='iLQR (Open Loop)', linewidth=1)
    ax3.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=m_w, label='Koopman MPC', linewidth=1)
    ax3.plot(t_scp, z_scp[:, 4], 'tab:blue', label='Nonlinear ROMPC', linewidth=3)
    ax3.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
else:
    # ax3.plot(t_rompc, z_rompc[:, 5], 'tab:green', marker='x', markevery=m_w, label='Linear ROMPC', linewidth=1)
    # ax3.plot(t_koop, z_koop[:, 5], 'tab:orange', marker='^', markevery=m_w, label='Koopman MPC', linewidth=1)
    # ax3.plot(t_scp, z_scp[:, 5], 'tab:blue', label='Nonlinear ROMPC', linewidth=3)
    ax3.plot(t_target, zf_target[:, 5], '--k', alpha=1, linewidth=1, label='Target')
    plt.ylabel(r'$z_{ee}$ [mm]', fontsize=14)
ax3.set_xlim([0, 10])
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.legend(loc='best', prop={'size': 14})

figure_file = join(path, name + '.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')


# MSE calculations
# Calculation of desired trajectory
if name == 'figure8':
    zf_desired = zf_target.copy()
else:
    zf_desired = zf_target.copy()

f = interp1d(t_target, zf_desired, axis=0)
zd_koop = f(t_koop)
zd_scp = f(t_scp)

if name == 'figure8':
    err_koop = (z_koop - zd_koop)[:,3:5]
    err_scp = (z_scp - zd_scp)[:,3:5]
else:
    err_koop = (z_koop - zd_koop)[:,4:6]
    err_scp = (z_scp - zd_scp)[:,4:6]

# inner norm gives euclidean distance, outer norm squared / nbr_samples gives MSE
mse_koop = np.linalg.norm(np.linalg.norm(err_koop, axis=1))**2 / err_koop.shape[0]
mse_scp = np.linalg.norm(np.linalg.norm(err_scp, axis=1))**2 / err_scp.shape[0]

print('------ Mean Squared Errors (MSEs)----------')
print('Ours (SCP): {}'.format(mse_scp))
print('Koopman: {}'.format(mse_koop))

print('-------------Solve times ---------------')
print('Ours: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_scp), np.mean(solve_times_scp),
                                                     np.max(solve_times_scp)))
print('Koopman: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_koop), np.mean(solve_times_koop),
                                                        np.max(solve_times_koop)))

plt.show()
