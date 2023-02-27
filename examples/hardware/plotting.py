from os.path import dirname, abspath, join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pdb

from sofacontrol.utils import load_data, set_axes_equal

path = dirname(abspath(__file__))

constrained = True
plot_rompc = False
#############################################
# Problem 1, Figure 8 with constraints
#############################################
M = 3
T = 10
N = 500
t_target = np.linspace(0, M*T, M*N)
th = np.linspace(0, M * 2 * np.pi, M*N)
zf_target = np.zeros((M*N, 6))

zf_target[:, 3] = -25. * np.sin(th)
zf_target[:, 4] = 25. * np.sin(2 * th)

# zf_target[:, 3] = -30. * np.sin(th) - 7.1
# zf_target[:, 4] = 30. * np.sin(2 * th)

# zf_target[:, 3] = -30. * np.sin(2 * th)
# zf_target[:, 4] = 30. * np.sin(4 * th)

# zf_target[:, 3] = -35. * np.sin(th) - 7.1
# zf_target[:, 4] = 35. * np.sin(2 * th)

# zf_target[:, 3] = -5. * np.sin(th) - 7.1
# zf_target[:, 4] = 5. * np.sin(2 * th)

# zf_target[:, 3] = -15. * np.sin(th)
# zf_target[:, 4] = 15. * np.sin(2 * th)

# zf_target[:, 3] = -15. * np.sin(8 * th) - 7.1
# zf_target[:, 4] = 15. * np.sin(16 * th)
if constrained:
    y_ub = 15

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
# phi = 17
# x_target = np.zeros(M*N)
# y_target = r * np.sin(phi * T / (2 * np.pi) * th)
# z_target = r - r * np.cos(phi * T / (2 * np.pi) * th) + 107.0

# r = 15
# x_target = np.zeros(M*N)
# y_target = r * np.sin(th)
# z_target = r - r * np.cos(th) + 107

# zf_target = np.zeros((M*N, 6))
# zf_target[:, 3] = x_target
# zf_target[:, 4] = y_target
# zf_target[:, 5] = z_target
# name = 'circle'


# Load SSM data
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

# Load TPWL data
tpwl_simdata_file = join(path, 'scp_sim.pkl')
tpwl_data = load_data(tpwl_simdata_file)
idx = np.argwhere(tpwl_data['t'] >= 3)[0][0]
t_tpwl = tpwl_data['t'][idx:] - tpwl_data['t'][idx]
z_tpwl = tpwl_data['z'][idx:, :]
u_tpwl = tpwl_data['u'][idx:, :]
solve_times_tpwl = tpwl_data['info']['solve_times']
real_time_limit_tpwl = tpwl_data['info']['rollout_time']

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

# Load rompc data
rompc_simdata_file = join(path, 'rompc_sim.pkl')
rompc_data = load_data(rompc_simdata_file)
idx = np.argwhere(rompc_data['t'] >= 3)[0][0]
t_rompc = rompc_data['t'][idx:] - rompc_data['t'][idx]
z_rompc = rompc_data['z'][idx:, :]
solve_times_rompc = rompc_data['info']['solve_times']
real_time_limit_rompc = rompc_data['info']['rollout_time']

plot_rollouts = True
m_w = 30

fig1 = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
##################################################
# Plot infinity sign via x vs. y
##################################################
if name == 'figure8':
    ax1 = fig1.add_subplot(111)
    # z_lb = np.array([-20 - 7.1, -25 + 0])
    # z_ub = np.array([20 - 7.1, 5 + 0])

    z_lb = np.array([-16 - 7.1, -16 + 0])
    z_ub = np.array([16 - 7.1, 5 + 0])

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

    ax1.plot(z_tpwl[:, 3], z_tpwl[:, 4], 'tab:green', marker='x', markevery=20, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    ax1.plot(z_koop[:, 3], z_koop[:, 4], 'tab:orange', marker='^', markevery=20, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    ax1.plot(z_scp[:, 3], z_scp[:, 4], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    ax1.plot(zf_target[:, 3], zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    if plot_rompc:
        ax1.plot(z_rompc[:, 3], z_rompc[:, 4], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    ax1.set_xlabel(r'$x_{ee}$ [mm]', fontsize=14)
    ax1.set_ylabel(r'$y_{ee}$ [mm]', fontsize=14)

    # Remove top and right border
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
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

    ax1.plot3D(z_tpwl[:, 3], z_tpwl[:, 4], z_tpwl[:, 5], 'tab:green', marker='x', markevery=20, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    ax1.plot3D(z_koop[:, 3], z_koop[:, 4], z_koop[:, 5], 'tab:orange', marker='^', markevery=20, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    ax1.plot3D(z_scp[:, 3], z_scp[:, 4], z_scp[:, 5], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    ax1.plot3D(zf_target[:, 3], zf_target[:, 4], zf_target[:, 5], '--k', alpha=1, linewidth=1, label='Target')
    if plot_rompc:
        ax1.plot3D(z_rompc[:, 3], z_rompc[:, 4], z_rompc[:, 5], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    ax1.set_xlabel(r'$x_{ee}$ [mm]', fontsize=14)
    ax1.set_ylabel(r'$y_{ee}$ [mm]', fontsize=14)
    ax1.set_zlabel(r'$z_{ee}$ [mm]', fontsize=14)
    set_axes_equal(ax1)

    # startx, endx = ax1.get_xlim()
    # starty, endy = ax1.get_ylim()
    # startz, endz = ax1.get_zlim()
    # ax1.xaxis.set_ticks(np.arange(int(startx),int(endx),10))
    # ax1.yaxis.set_ticks(np.arange(int(starty),int(endy),10))
    # ax1.zaxis.set_ticks(np.arange(int(startz),int(endz),10))

#plt.axis('off')
#plt.legend(loc='upper left', prop={'size': 14}, borderaxespad=0, bbox_to_anchor=(0.25, 0.12))
ax1.tick_params(axis='both', labelsize=18)

figure_file = join(path, 'diamond_x_vs_y.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')

##################################################
# Plot trajectory as function of time
##################################################
#fig2 = plt.figure(figsize=(20, 6), facecolor='w', edgecolor='k')
fig2 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')

ax2 = fig2.add_subplot(211)

if name ==  'figure8':
    #ax2.plot(t_tpwl, z_tpwl[:, 3], 'tab:green', marker='x', markevery=m_w, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    #ax2.plot(t_koop, z_koop[:, 3], 'tab:orange', marker='^', markevery=m_w, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    ax2.plot(t_scp, z_scp[:, 3], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    ax2.plot(t_target, zf_target[:, 3], '--k', alpha=1, linewidth=1, label='Target')
    if plot_rompc:
        ax2.plot(t_rompc, z_rompc[:, 3], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    idx = 0
    if plot_rollouts:
        for idx in range(np.shape(z_opt_rollout)[0]):
            if idx % 2 == 0:
                z_horizon = z_opt_rollout[idx]
                t_horizon = t_opt_rollout[idx]
                ax2.plot(t_horizon, z_horizon[:, 0], 'tab:red', marker='o', markevery=2)
    plt.ylabel(r'$x_{ee}$ [mm]', fontsize=14)
else:
    ax2.plot(t_tpwl, z_tpwl[:, 4], 'tab:green', marker='x', markevery=m_w, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    ax2.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=m_w, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    ax2.plot(t_scp, z_scp[:, 4], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    ax2.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    if plot_rompc:
        ax2.plot(t_rompc, z_rompc[:, 4], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
ax2.set_xlim([0, 10])
ax2.tick_params(axis='both', labelsize=18)
plt.xlabel(r'$t$ [s]', fontsize=14)
#plt.legend(loc='upper left', prop={'size': 12})

ax3 = fig2.add_subplot(212)
if name == 'figure8':
    #ax3.plot(t_tpwl, z_tpwl[:, 4], 'tab:green', marker='x', markevery=m_w, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    #ax3.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=m_w, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    ax3.plot(t_scp, z_scp[:, 4], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    ax3.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    if plot_rompc:
        ax3.plot(t_rompc, z_rompc[:, 4], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    if constrained:
        ax3.plot(t_target, y_ub * np.ones_like(t_target), 'r', label='Constraint')
    if plot_rollouts:
        for idx in range(np.shape(z_opt_rollout)[0]):
            if idx % 2 == 0:
                z_horizon = z_opt_rollout[idx]
                t_horizon = t_opt_rollout[idx]
                ax3.plot(t_horizon, z_horizon[:, 1], 'tab:red', marker='o', markevery=2)
    plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
else:
    ax3.plot(t_tpwl, z_tpwl[:, 5], 'tab:green', marker='x', markevery=m_w, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    ax3.plot(t_koop, z_koop[:, 5], 'tab:orange', marker='^', markevery=m_w, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    ax3.plot(t_scp, z_scp[:, 5], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    ax3.plot(t_target, zf_target[:, 5], '--k', alpha=1, linewidth=1, label='Target')
    if plot_rompc:
        ax3.plot(t_rompc, z_rompc[:, 5], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    plt.ylabel(r'$z_{ee}$ [mm]', fontsize=14)
ax3.set_xlim([0, 10])
ax3.tick_params(axis='both', labelsize=18)
plt.xlabel(r'$t$ [s]', fontsize=14)
#plt.legend(loc='upper left', prop={'size': 14}, borderaxespad=0, bbox_to_anchor=(0, 1.2))

figure_file = join(path, name + '.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')
plt.show()

# MSE calculations
# Calculation of desired trajectory
if name == 'figure8':
    zf_desired = zf_target.copy()
    if constrained:
        zf_desired[:, 4] = np.minimum(y_ub, zf_target[:,4])
else:
    zf_desired = zf_target.copy()

f = interp1d(t_target, zf_desired, axis=0)
zd_koop = f(t_koop)
zd_scp = f(t_scp)
zd_tpwl = f(t_tpwl)
zd_rompc = f(t_rompc)

# Remove constraints from error and compute num violations / num points
# idx represents where the constraints are active. idx_viol_x represent where
# the constraints are violated
constraint_idx = 4
if constrained:
    idx_koop = np.argwhere(zd_koop[:, constraint_idx] >= y_ub)
    viol_koop = np.count_nonzero(z_koop[idx_koop.flatten(), constraint_idx] > y_ub + 0.2) / idx_koop.size
    idx_viol_koop = np.argwhere(z_koop[idx_koop.flatten(), constraint_idx] > y_ub + 0.2)
    zd_koop = np.delete(zd_koop, idx_viol_koop, axis=0)
    z_koop = np.delete(z_koop, idx_viol_koop, axis=0)

    idx_scp = np.argwhere(zd_scp[:, constraint_idx] >= y_ub)
    viol_scp = np.count_nonzero(z_scp[idx_scp.flatten(), constraint_idx] > y_ub + 0.2) / idx_scp.size
    idx_viol_scp = np.argwhere(z_scp[idx_scp.flatten(), constraint_idx] > y_ub + 0.2)
    zd_scp = np.delete(zd_scp, idx_viol_scp, axis=0)
    z_scp = np.delete(z_scp, idx_viol_scp, axis=0)

    idx_tpwl = np.argwhere(zd_tpwl[:, constraint_idx] >= y_ub)
    viol_tpwl = np.count_nonzero(z_tpwl[idx_tpwl.flatten(), constraint_idx] > y_ub + 0.2) / idx_tpwl.size
    idx_viol_tpwl = np.argwhere(z_tpwl[idx_tpwl.flatten(), constraint_idx] > y_ub + 0.2)
    zd_tpwl = np.delete(zd_tpwl, idx_viol_tpwl, axis=0)
    z_tpwl = np.delete(z_tpwl, idx_viol_tpwl, axis=0)

    idx_rompc = np.argwhere(zd_rompc[:, constraint_idx] >= y_ub)
    viol_rompc = np.count_nonzero(z_rompc[idx_rompc.flatten(), constraint_idx] > y_ub + 0.2) / idx_rompc.size
    idx_viol_rompc = np.argwhere(z_rompc[idx_rompc.flatten(), constraint_idx] > y_ub + 0.2)
    zd_rompc = np.delete(zd_rompc, idx_viol_rompc, axis=0)
    z_rompc = np.delete(z_rompc, idx_viol_rompc, axis=0)

if name == 'figure8':
    err_koop = (z_koop - zd_koop)[:,3:5]
    err_scp = (z_scp - zd_scp)[:,3:5]
    err_tpwl = (z_tpwl - zd_tpwl)[:,3:5]
    err_rompc = (z_rompc - zd_rompc)[:, 3:5]
else:
    err_koop = (z_koop - zd_koop)[:,3:6]
    err_scp = (z_scp - zd_scp)[:,3:6]
    err_tpwl = (z_tpwl - zd_tpwl)[:,3:6]
    err_rompc = (z_rompc - zd_rompc)[:, 3:6]

# inner norm gives euclidean distance, outer norm squared / nbr_samples gives MSE
koop_norm = np.linalg.norm(np.linalg.norm(zd_koop, axis=1))**2
tpwl_norm = np.linalg.norm(np.linalg.norm(zd_tpwl, axis=1))**2
scp_norm = np.linalg.norm(np.linalg.norm(zd_scp, axis=1))**2
rompc_norm = np.linalg.norm(np.linalg.norm(zd_rompc, axis=1))**2

# TODO: Wait for transients to dies down
if name == 'circle':
    mse_koop = np.linalg.norm(np.linalg.norm(err_koop[20:], axis=1))**2 / err_koop.shape[0]
    mse_tpwl = np.linalg.norm(np.linalg.norm(err_tpwl[20:], axis=1))**2 / err_tpwl.shape[0]
    mse_scp = np.linalg.norm(np.linalg.norm(err_scp[20:], axis=1))**2 / err_scp.shape[0]
    mse_rompc = np.linalg.norm(np.linalg.norm(err_rompc[20:], axis=1)) ** 2 / err_rompc.shape[0]
else:
    mse_koop = np.linalg.norm(np.linalg.norm(err_koop, axis=1)) ** 2 / err_koop.shape[0]
    mse_tpwl = np.linalg.norm(np.linalg.norm(err_tpwl, axis=1)) ** 2 / err_tpwl.shape[0]
    mse_scp = np.linalg.norm(np.linalg.norm(err_scp, axis=1)) ** 2 / err_scp.shape[0]
    mse_rompc = np.linalg.norm(np.linalg.norm(err_rompc, axis=1)) ** 2 / err_rompc.shape[0]

if name == 'circle':
    fig3 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')

    ax_circ_x = fig3.add_subplot(111)
    ax_circ_x.plot(t_tpwl, z_tpwl[:, 3], 'tab:green', marker='x', markevery=m_w, label='TPWL', linewidth=1)
    ax_circ_x.plot(t_koop, z_koop[:, 3], 'tab:orange', marker='^', markevery=m_w, label='Koopman', linewidth=1)
    ax_circ_x.plot(t_scp, z_scp[:, 3], 'tab:blue', label='SSMR (Ours)', linewidth=3)
    ax_circ_x.plot(t_target, zf_target[:, 3], '--k', alpha=1, linewidth=1, label='Target')
    ax_circ_x.set_xlim([0, 10])

### Plotting Errors ###
if not constrained:
    fig4 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')

    ax_err1 = fig4.add_subplot(111)
    if name == 'figure8':
        ax_err1.plot(t_tpwl, np.linalg.norm(err_tpwl, axis=1, ord=2), 'tab:green', marker='x', markevery=m_w, label='TPWL', linewidth=1)
        ax_err1.plot(t_koop, np.linalg.norm(err_koop, axis=1, ord=2), 'tab:orange', marker='^', markevery=m_w, label='Koopman', linewidth=1)
        ax_err1.plot(t_scp, np.linalg.norm(err_scp, axis=1, ord=2), 'tab:blue', marker='*', markevery=m_w, label='SSMR', linewidth=3)
        plt.ylabel(r'$\log ||z - z_{des}||_2$', fontsize=14)
    else:
        ax_err1.plot(t_tpwl, np.linalg.norm(err_tpwl, axis=1), 'tab:green', marker='x', markevery=m_w, label='TPWL', linewidth=1)
        ax_err1.plot(t_koop, np.linalg.norm(err_koop, axis=1), 'tab:orange', marker='^', markevery=m_w, label='Koopman', linewidth=1)
        ax_err1.plot(t_scp, np.linalg.norm(err_scp, axis=1), 'tab:blue', marker='*', markevery=m_w, label='SSMR', linewidth=3)
        plt.ylabel(r'$\log ||z - z_{des}||_2$', fontsize=14)
    ax_err1.set_xlim([0, 10])
    ax_err1.set_yscale('log')
    plt.xlabel(r'$t$ [s]', fontsize=14)
    plt.legend(loc='best', prop={'size': 14})
    plt.grid()
    # Save error
    name_err = name + '_error'
    figure_file = join(path, name_err + '.png')
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    plt.show()

print('------ Mean Squared Errors (MSEs)----------')
print('Ours (SSMR): {}'.format(mse_scp))
print('Koopman: {}'.format(mse_koop))
print('TPWL: {}'.format(mse_tpwl))
print('Linear: {}'.format(mse_rompc))


print('-------------Solve times ---------------')
print('TPWL: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_tpwl), np.mean(solve_times_tpwl),
                                                     np.max(solve_times_tpwl)))

print('Koopman: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_koop), np.mean(solve_times_koop),
                                                        np.max(solve_times_koop)))

print('Ours (SSMR): Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_ssm), np.mean(solve_times_ssm),
                                                     np.max(solve_times_ssm)))

print('Linear: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_rompc), np.mean(solve_times_rompc),
                                                     np.max(solve_times_rompc)))

if constrained:
    print('-------------Fraction of Constraint Violations ---------------')
    print('Ours (SSMR): {}'.format(viol_scp))
    print('Koopman: {}'.format(viol_koop))
    print('TPWL: {}'.format(viol_tpwl))
    print('Linear: {}'.format(viol_rompc))
