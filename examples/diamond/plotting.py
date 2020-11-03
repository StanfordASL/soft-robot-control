from os.path import dirname, abspath, join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from sofacontrol.utils import load_data

path = dirname(abspath(__file__))

# Load SCP data
scp_simdata_file = join(path, 'scp_sim.pkl')
scp_data = load_data(scp_simdata_file)
idx = np.argwhere(scp_data['t'] >= 2)[0][0]
t_scp = scp_data['t'][idx:] - scp_data['t'][idx]
z_scp = scp_data['z'][idx:, :]
solve_times_scp = scp_data['info']['solve_times']
real_time_limit_scp = scp_data['info']['rollout_time']

# Load ROMPC data
rompc_simdata_file = join(path, 'rompc_sim.pkl')
rompc_data = load_data(rompc_simdata_file)
idx = np.argwhere(rompc_data['t'] >= 2)[0][0]
t_rompc = rompc_data['t'][idx:] - rompc_data['t'][idx]
z_rompc = rompc_data['z'][idx:, :]
solve_times_rompc = rompc_data['info']['solve_times']
real_time_limit_rompc = rompc_data['info']['rollout_time']

# Load Koopman data
koop_simdata_file = join(path, 'koopman_sim.pkl')
koop_data = load_data(koop_simdata_file)
idx = np.argwhere(koop_data['t'] >= 2)[0][0]
t_koop = koop_data['t'][idx:] - koop_data['t'][idx]
z_koop = koop_data['z'][idx:, :]
solve_times_koop = koop_data['info']['solve_times']
real_time_limit_koop = koop_data['info']['rollout_time']

# State constraints
z_lb = np.array([-17.5 - 5.5, -20 + 1.5])
z_ub = np.array([17.5 - 5.5, 20 + 1.5])

# Target trajectory
T = 10
t_target = np.linspace(0, T, 1000)
th = np.linspace(0, 2 * np.pi, 1000)
zf_target = np.zeros((1000, 6))
zf_target[:, 3] = -20. * np.sin(th) - 5.5
zf_target[:, 4] = 10. * np.sin(2 * th) + 1.5

##################################################
# Plot infinity sign via x vs. y
##################################################
fig1 = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
ax1 = fig1.add_subplot(111)
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
ax1.plot(z_rompc[:, 3], z_rompc[:, 4], 'tab:green', marker='x', markevery=20, label='ROMPC', linewidth=1)
ax1.plot(z_koop[:, 3], z_koop[:, 4], 'tab:orange', marker='^', markevery=20, label='Koopman', linewidth=1)
ax1.plot(z_scp[:, 3], z_scp[:, 4], 'tab:blue', label='Ours', linewidth=3)
ax1.plot(zf_target[:, 3], zf_target[:, 4], '--k', alpha=1)

plt.axis('off')
plt.legend(loc='upper center', prop={'size': 14})

figure_file = join(path, 'diamond_x_vs_y.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')

##################################################
# Plot trajectory as function of time
##################################################
fig2 = plt.figure(figsize=(18, 4), facecolor='w', edgecolor='k')
ax2 = fig2.add_subplot(121)
ax2.plot(t_target, z_lb[0] * np.ones_like(t_target), 'r')
ax2.plot(t_target, z_ub[0] * np.ones_like(t_target), 'r')

ax2.plot(t_rompc, z_rompc[:, 3], 'tab:green', marker='x', markevery=40, label='ROMPC', linewidth=1)
ax2.plot(t_koop, z_koop[:, 3], 'tab:orange', marker='^', markevery=40, label='Koopman', linewidth=1)
ax2.plot(t_scp, z_scp[:, 3], 'tab:blue', label='Ours', linewidth=3)
ax2.plot(t_target, zf_target[:, 3], '--k', alpha=1, linewidth=1)
ax2.set_xlim([0, 10])
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.ylabel(r'$x_{ee}$ [mm]', fontsize=14)
plt.legend(loc='upper left', prop={'size': 14})

ax3 = fig2.add_subplot(122)
# ax3.plot(t_target, z_lb[1]*np.ones_like(t_target), 'r', label='y constraint')
# ax3.plot(t_target, z_ub[1]*np.ones_like(t_target), 'r')
ax3.plot(t_rompc, z_rompc[:, 4], 'tab:green', marker='x', markevery=40, label='ROMPC', linewidth=1)
ax3.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=40, label='Koopman', linewidth=1)
ax3.plot(t_scp, z_scp[:, 4], 'tab:blue', label='Ours', linewidth=3)
ax3.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1)
ax3.set_xlim([0, 10])
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
plt.legend(loc='upper right', prop={'size': 14})

figure_file = join(path, 'diamond_xy_vs_t.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')

##################################################
# Solve times
##################################################
# fig4, ax4 = plt.subplots()
# ax4.boxplot(solve_times_scp)
# ax4.plot([0.85, 1.15], np.ones(2) * real_time_limit_scp, '--r', label='Real time limit')
# plt.ylabel('Computation time [s]')
# plt.legend()
# ax4.set_xlim([0.85, 1.15])
# ax4.set_ylim(bottom=0.)

print('Ours: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_scp), np.mean(solve_times_scp),
                                                     np.max(solve_times_scp)))
print('ROMPC: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_rompc), np.mean(solve_times_rompc),
                                                      np.max(solve_times_rompc)))
print('Koopman: Min: {}, Mean: {} ms, Max: {} s'.format(np.min(solve_times_koop), np.mean(solve_times_koop),
                                                        np.max(solve_times_koop)))

plt.show()
