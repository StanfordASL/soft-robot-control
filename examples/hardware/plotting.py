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
zf_target[:, 3] = -15. * np.sin(th) - 7.1
zf_target[:, 4] = 15. * np.sin(2 * th)

# zf_target[:, 3] = -25. * np.sin(th) + 13
# zf_target[:, 4] = 25. * np.sin(2 * th) + 20

# zf_target[:, 3] = -40. * np.sin(th) - 7.1
# zf_target[:, 4] = 40. * np.sin(2 * th)

# zf_target[:, 3] = -5. * np.sin(th) - 7.1
# zf_target[:, 4] = 5. * np.sin(2 * th)

# zf_target[:, 3] = -15. * np.sin(th)
# zf_target[:, 4] = 15. * np.sin(2 * th)

# zf_target[:, 3] = -15. * np.sin(8 * th) - 7.1
# zf_target[:, 4] = 15. * np.sin(16 * th)

# y_ub = 5
name = 'figure8'

##############################################
# Problem 2, Circle on side
##############################################
# M = 3
# T = 5
# N = 1000
# t_target = np.linspace(0, M*T, M*N)
# th = np.linspace(0, M*2*np.pi, M*N)
#
# r = 15
# x_target = np.zeros(M*N)
# y_target = r * np.sin(2 * th)
# z_target = r - r * np.cos(2 * th) + 107

# r = 20
# x_target = np.zeros(M*N)
# y_target = r * np.sin(17 * th)
# z_target = r - r * np.cos(17 * th) + 107

# zf_target = np.zeros((M*N, 6))
# zf_target[:, 3] = x_target
# zf_target[:, 4] = y_target
# zf_target[:, 5] = z_target
# name = 'circle'


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
solve_times_koop = koop_data['info']['solve_times']
real_time_limit_koop = koop_data['info']['rollout_time']
t_opt = koop_data['info']['t_opt']
# z_opt_rollout = koop_data['info']['z_rollout']
# t_opt_rollout = koop_data['info']['t_rollout']

plot_rollouts = False
m_w = 30

fig1 = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
##################################################
# Plot infinity sign via x vs. y
##################################################
if name == 'figure8':
    ax1 = fig1.add_subplot(111)
    # z_lb = np.array([-20 - 7.1, -25 + 0])
    # z_ub = np.array([20 - 7.1, 5 + 0])

    z_lb = np.array([-20 - 7.1, -20 + 0])
    z_ub = np.array([20 - 7.1, 20 + 0])

    ax1.add_patch(
        patches.Rectangle(
            xy=(z_lb[0], z_lb[1]),  # point of origin.
            width=z_ub[0] - z_lb[0],
            height=z_ub[1] - z_lb[1],
            linewidth=2,
            color='tab:red',
            fill=False,
        )
    )

    ax1.plot(z_ilqr[:, 3], z_ilqr[:, 4], 'tab:green', marker='x', markevery=20, label='TPWL CL', linewidth=1)
    ax1.plot(z_koop[:, 3], z_koop[:, 4], 'tab:orange', marker='^', markevery=20, label='Koopman CL', linewidth=1)
    ax1.plot(z_scp[:, 3], z_scp[:, 4], 'tab:blue', label='SSM CL', linewidth=3)
    ax1.plot(zf_target[:, 3], zf_target[:, 4], '--k', alpha=1, linewidth=1)
else:
    ax1 = fig1.add_subplot(111, projection='3d')
    # z_lb = np.array([-30 - 0, -30 + 127.])
    # z_ub = np.array([30 - 0, 30 + 127.])

    # ax1.add_patch(
    #     patches.Rectangle(
    #         xy=(z_lb[0], z_lb[1]),  # point of origin.
    #         width=z_ub[0] - z_lb[0],
    #         height=z_ub[1] - z_lb[1],
    #         linewidth=2,
    #         color='tab:red',
    #         fill=False,
    #     )
    # )

    ax1.plot3D(z_ilqr[:, 3], z_ilqr[:, 4], z_ilqr[:, 5], 'tab:green', marker='x', markevery=20, label='TPWL CL', linewidth=1)
    ax1.plot3D(z_koop[:, 3], z_koop[:, 4], z_koop[:, 5], 'tab:orange', marker='^', markevery=20, label='Koopman CL', linewidth=1)
    ax1.plot3D(z_scp[:, 3], z_scp[:, 4], z_scp[:, 5], 'tab:blue', label='SSM CL', linewidth=3)
    ax1.plot3D(zf_target[:, 3], zf_target[:, 4], zf_target[:, 5], '--k', alpha=1, linewidth=1)

# plt.axis('off')
plt.legend(loc='upper center', prop={'size': 14})

figure_file = join(path, 'diamond_x_vs_y.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')

##################################################
# Plot trajectory as function of time
##################################################
fig2 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax2 = fig2.add_subplot(211)

if name ==  'figure8':
    ax2.plot(t_ilqr, z_ilqr[:, 3], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
    ax2.plot(t_koop, z_koop[:, 3], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
    ax2.plot(t_scp, z_scp[:, 3], 'tab:blue', label='SSM CL', linewidth=3)
    ax2.plot(t_target, zf_target[:, 3], '--k', alpha=1, linewidth=1, label='Target')
    idx = 0
    if plot_rollouts:
        for idx in range(np.shape(z_opt_rollout)[0]):
            if idx % 5 == 0:
                z_horizon = z_opt_rollout[idx]
                t_horizon = t_opt_rollout[idx]
                ax2.plot(t_horizon, z_horizon[:, 0], 'tab:red', marker='o', markevery=2)
    plt.ylabel(r'$x_{ee}$ [mm]', fontsize=14)
else:
    ax2.plot(t_ilqr, z_ilqr[:, 4], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
    ax2.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
    ax2.plot(t_scp, z_scp[:, 4], 'tab:blue', label='SSM CL', linewidth=3)
    ax2.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
ax2.set_xlim([0, 10])
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.legend(loc='best', prop={'size': 14})

ax3 = fig2.add_subplot(212)
if name == 'figure8':
    ax3.plot(t_ilqr, z_ilqr[:, 4], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
    ax3.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
    ax3.plot(t_scp, z_scp[:, 4], 'tab:blue', label='SSM CL', linewidth=3)
    ax3.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    # ax3.plot(t_target, y_ub * np.ones_like(t_target), 'r', label='Constraint')
    if plot_rollouts:
        for idx in range(np.shape(z_opt_rollout)[0]):
            if idx % 5 == 0:
                z_horizon = z_opt_rollout[idx]
                t_horizon = t_opt_rollout[idx]
                ax3.plot(t_horizon, z_horizon[:, 1], 'tab:red', marker='o', markevery=2)
    plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
else:
    ax3.plot(t_ilqr, z_ilqr[:, 5], 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
    ax3.plot(t_koop, z_koop[:, 5], 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
    ax3.plot(t_scp, z_scp[:, 5], 'tab:blue', label='SSM CL', linewidth=3)
    ax3.plot(t_target, zf_target[:, 5], '--k', alpha=1, linewidth=1, label='Target')
    plt.ylabel(r'$z_{ee}$ [mm]', fontsize=14)
ax3.set_xlim([0, 10])
plt.xlabel(r'$t$ [s]', fontsize=14)
# plt.legend(loc='best', prop={'size': 14})

figure_file = join(path, name + '.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')
plt.show()

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
    err_koop = (z_koop - zd_koop)[:,3:6]
    err_scp = (z_scp - zd_scp)[:,3:6]
    err_ilqr = (z_ilqr - zd_rompc)[:,3:6]

# inner norm gives euclidean distance, outer norm squared / nbr_samples gives MSE
koop_norm = np.linalg.norm(np.linalg.norm(zd_koop, axis=1))**2
rompc_norm = np.linalg.norm(np.linalg.norm(zd_rompc, axis=1))**2
scp_norm = np.linalg.norm(np.linalg.norm(zd_scp, axis=1))**2

mse_koop = np.linalg.norm(np.linalg.norm(err_koop, axis=1))**2 / err_koop.shape[0]
mse_rompc = np.linalg.norm(np.linalg.norm(err_ilqr, axis=1))**2 / err_ilqr.shape[0]
mse_scp = np.linalg.norm(np.linalg.norm(err_scp, axis=1))**2 / err_scp.shape[0]

### Plotting Errors ###
fig3 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')

ax_err1 = fig3.add_subplot(111)
if name == 'figure8':
    ax_err1.plot(t_ilqr, np.linalg.norm(err_ilqr, axis=1, ord=2), 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
    ax_err1.plot(t_koop, np.linalg.norm(err_koop, axis=1, ord=2), 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
    ax_err1.plot(t_scp, np.linalg.norm(err_scp, axis=1, ord=2), 'tab:blue', marker='*', markevery=m_w, label='SSM CL', linewidth=3)
    plt.ylabel(r'$\log ||z - z_{des}||_2$', fontsize=14)
else:
    ax_err1.plot(t_ilqr, np.linalg.norm(err_ilqr, axis=1), 'tab:green', marker='x', markevery=m_w, label='TPWL CL', linewidth=1)
    ax_err1.plot(t_koop, np.linalg.norm(err_koop, axis=1), 'tab:orange', marker='^', markevery=m_w, label='Koopman CL', linewidth=1)
    ax_err1.plot(t_scp, np.linalg.norm(err_scp, axis=1), 'tab:blue', marker='*', markevery=m_w, label='SSM CL', linewidth=3)
    plt.ylabel(r'$\log ||z - z_{des}||_2$', fontsize=14)
ax_err1.set_xlim([0, 10])
ax_err1.set_yscale('log')
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.legend(loc='best', prop={'size': 14})
plt.grid()


name_err = name + '_error'
figure_file = join(path, name_err + '.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')
plt.show()

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