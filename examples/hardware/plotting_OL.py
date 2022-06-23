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
zf_target[:, 3] = -15. * np.sin(8 * th)
zf_target[:, 4] = 15. * np.sin(16 * th)
y_ub = 5
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


# Load TPWL data
#scp_simdata_file = join(path, 'scp_sim.pkl')
tpwl_simdata_file = join(path, 'scp_OL_TPWL_sim.pkl')
tpwl_data = load_data(tpwl_simdata_file)
idx = np.argwhere(tpwl_data['t'] >= 3)[0][0]
u_tpwl = tpwl_data['u'][idx:, :]
t_tpwl = tpwl_data['t'][idx:] - tpwl_data['t'][idx]
z_tpwl = tpwl_data['z'][idx:, :]

# Load SSM data
ssm_simdata_file = join(path, 'scp_OL_SSM_sim.pkl')
ssm_data = load_data(ssm_simdata_file)
idx = np.argwhere(ssm_data['t'] >= 3)[0][0]
u_ssm = ssm_data['u'][idx:, :]
t_ssm = ssm_data['t'][idx:] - ssm_data['t'][idx]
z_ssm = ssm_data['z'][idx:, :]

m_w = 30
##################################################
# Plot trajectory as function of time
##################################################
fig2 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax2 = fig2.add_subplot(211)

if name == 'figure8':
    ax2.plot(t_ssm, z_ssm[:, 3], 'tab:blue', label='SSM Open Loop', linewidth=3)
    ax2.plot(t_tpwl, z_tpwl[:, 3], 'tab:orange', marker='^', markevery=m_w, label='TPWL Open Loop', linewidth=1)
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
    ax3.plot(t_ssm, z_ssm[:, 4], 'tab:blue', label='SSM Open Loop', linewidth=3)
    ax3.plot(t_tpwl, z_tpwl[:, 4], 'tab:orange', marker='^', markevery=m_w, label='TPWL Open Loop', linewidth=1)
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
    # zf_desired[:, 4] = np.minimum(y_ub, zf_target[:,4])
else:
    zf_desired = zf_target.copy()

f = interp1d(t_target, zf_desired, axis=0)
zd_tpwl = f(t_tpwl)
zd_ssm = f(t_ssm)

if name == 'figure8':
    err_tpwl = (z_tpwl - zd_tpwl)[:, 3:5]
    err_ssm = (z_ssm - zd_ssm)[:, 3:5]
else:
    err_tpwl = (z_tpwl - zd_tpwl)[:, 4:6]
    err_ssm = (z_ssm - zd_ssm)[:, 4:6]

# inner norm gives euclidean distance, outer norm squared / nbr_samples gives MSE
mse_tpwl = np.linalg.norm(np.linalg.norm(err_tpwl, axis=1))**2 / err_tpwl.shape[0]
mse_ssm = np.linalg.norm(np.linalg.norm(err_ssm, axis=1))**2 / err_ssm.shape[0]

print('------ Mean Squared Errors (MSEs)----------')
print('Ours (SSM): {}'.format(mse_ssm))
print('TPWL: {}'.format(mse_tpwl))

# print('-------------Solve times ---------------')
# print('Ours: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_scp), np.mean(solve_times_scp),
#                                                      np.max(solve_times_scp)))

# print('Koopman: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_koop), np.mean(solve_times_koop),
#                                                         np.max(solve_times_koop)))

plt.show()
