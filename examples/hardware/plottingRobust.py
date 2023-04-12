from os.path import dirname, abspath, join

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pdb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from sofacontrol.utils import load_data, set_axes_equal

path = dirname(abspath(__file__))

constrained = False
plot_rompc = False
plot_tubes = True
#############################################
# Problem 1, Figure 8 with constraints
#############################################
M = 5
T = 0.05
N = 1000
t_target = np.linspace(0, M*T, M*N)
th = np.linspace(0, M * 2 * np.pi, M*N)
zf_target = np.zeros((M*N, 6))

# Define the coordinates of the corners of the square
center = np.array([-7.1, 0.])
top_mid = np.array([-7.1, 1.])
top_left = np.array([-9, 1.])
top_right = np.array([25., 1.])
bottom_left = np.array([-9., -24.])
bottom_right = np.array([25, -24])

# Define the number of points along each edge of the square
num_points = M * N

# Create a set of points that trace out the perimeter of the square
# Transient points
points_center_topmid = np.linspace(center, top_mid, int(num_points / 2), endpoint=False)
points_top_mid_right = np.linspace(top_mid, top_right, int(num_points / 2), endpoint=False)

# Square points
points_right = np.linspace(top_right, bottom_right, num_points, endpoint=False)
points_bottom = np.linspace(bottom_right, bottom_left, num_points, endpoint=False)
points_left = np.linspace(bottom_left, top_left, num_points, endpoint=False)
points_top = np.linspace(top_left, top_right, num_points, endpoint=False)

# Setpoint to top left corner
setptLength = 10
setpoint_left = np.linspace(np.array([-10., 3.]), np.array([-10., 3.]), setptLength * num_points, endpoint=False)

# Combine the points from each edge into a single array
numRepeat = 2
pointsTransient = np.concatenate((points_center_topmid, points_top_mid_right))
points = np.concatenate((points_right, points_bottom, points_left))
pointsConnected = np.concatenate((points, points_top))
squarePeriodic = np.tile(pointsConnected, (numRepeat-1, 1))
squareTraj = np.concatenate((pointsTransient, squarePeriodic, points, setpoint_left))

numSegments = 4 * (numRepeat) + setptLength
t_target = np.linspace(0, numSegments * M * T, numSegments * M * N)

zf_target = np.zeros((squareTraj.shape[0], 6))
zf_target[:, 3] = squareTraj[:, 0]
zf_target[:, 4] = squareTraj[:, 1]

# zf_target[:, 3] = -25. * np.sin(th)
# zf_target[:, 4] = 25. * np.sin(2 * th)

# zf_target[:, 3] = -30. * np.sin(th)
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
x_opt_ssm = scp_data['info']['x_opt']
s_delta_scp = x_opt_ssm

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

plot_rollouts = False
m_w = 30

x_ub = 25
x_lb = -10
y_ub = 3.1
y_lb = -25.1

fig1 = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
##################################################
# Plot infinity sign via x vs. y
##################################################
if name == 'figure8':
    ax1 = fig1.add_subplot(111)
    # z_lb = np.array([-20 - 7.1, -25 + 0])
    # z_ub = np.array([20 - 7.1, 5 + 0])

    z_ub = np.array([25, 3 + 0])
    z_lb = np.array([-10.05, -25 + 0])

    # small plot
    # z_ub = np.array([5, 3 + 0])
    # z_lb = np.array([-10., -5 + 0])

    ax1.add_patch(
        patches.Rectangle(
            xy=(z_lb[0], z_lb[1]),  # point of origin.
            width=z_ub[0] - z_lb[0],
            height=z_ub[1] - z_lb[1],
            linewidth=5,
            color='tab:red',
            fill=False,
        )
    )

    # Plot initial condition
    plt.scatter(z_scp[0, 3], z_scp[0, 4], color='green', marker='^', s=200)

    # Plot trajectory
    ax1.plot(z_scp[:, 3], z_scp[:, 4], 'tab:blue', label='SSMR (Ours)', linewidth=3)
    ax1.plot(pointsConnected[:, 0], pointsConnected[:, 1], '--k', alpha=1, linewidth=1, label='Target')

    # Plot steady state
    plt.scatter(zf_target[-1, 3], zf_target[-1, 4], color="blue", marker="*", s=200)

    # Plot steady state point
    # plt.scatter(z_scp[-1, 3], z_scp[-1, 4], color="red", marker="^", s=100)

    ax1.set_xlabel(r'$x$ [mm]', fontsize=20)
    ax1.set_ylabel(r'$y$ [mm]', fontsize=20)
    ax1.tick_params(axis='both', labelsize=24)

    # Remove top and right border
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    # Plot tubes
    # plot_idxs = [60, 20, 32, 40, 50, 115]
    plot_idxs = [62, 20, 34, 52, 130]

    # Faster
    # plot_idxs = [54, 68, 30, 42, 130]

    # if plot_tubes:
    #     for idx in plot_idxs:
    #         s_horizon = s_delta_scp[idx]
    #         z_horizon = z_opt_rollout[idx]
    #         ax1.plot(z_horizon[:, 0], z_horizon[:, 1], 'tab:red', marker='o', markevery=1)
    #         r = s_horizon[:, -2] + 0.09 * s_horizon[:, -1]
    #         for idxPoint in range(z_horizon.shape[0]):
    #             circle = patches.Circle((z_horizon[idxPoint, 0], z_horizon[idxPoint, 1]), r[idxPoint],
    #                                     facecolor='orange', edgecolor='black', alpha=0.5)
    #             ax1.add_patch(circle)

    # Plot zoomed in view of movement towards setpoint
    # Create a zoomed-in inset axes outside the original plot
    axins = inset_axes(ax1, width="30%", height="30%", loc="center",
                       bbox_transform=ax1.transAxes)

    # Plot the zoomed-in tubes
    # s_zoom = s_delta_scp[plot_idxs[-1]]
    # z_zoom = z_opt_rollout[plot_idxs[-1]]
    # axins.plot(z_zoom[:-1, 0], z_zoom[:-1, 1], 'tab:red', marker='o', markevery=1)
    # r_zoom = s_zoom[:-1, -2] + 0.09 * s_zoom[:-1, -1]
    # for idxPoint in range(z_zoom.shape[0]-1):
    #     circleZoom = patches.Circle((z_zoom[idxPoint, 0], z_zoom[idxPoint, 1]), r_zoom[idxPoint],
    #                             facecolor='orange', edgecolor='black', alpha=0.5)
    #     axins.add_patch(circleZoom)

    axins.plot(z_scp[:, 3], z_scp[:, 4])
    axins.plot(pointsConnected[:, 0], pointsConnected[:, 1], '--k', alpha=1, linewidth=1)
    y_line = np.linspace(-1, 3.1, 100)
    x_line = np.linspace(-10, -6, 100)
    axins.plot(x_lb * np.ones_like(y_line), y_line, 'r', linewidth=5, zorder=1)
    axins.plot(x_line, y_ub * np.ones_like(x_line), 'r', linewidth=5, zorder=2)
    axins.scatter(z_scp[0, 3], z_scp[0, 4], color='green', marker='^', s=200)
    # # plt.scatter(z_scp[-1, 3], z_scp[-1, 4], color="red", marker="^", s=50)
    axins.set_xlim(-11, -6)
    axins.set_ylim(-1, 4)
    # axins.set_xlabel('x')
    # axins.set_ylabel('y')
    axins.scatter(zf_target[-1, 3], zf_target[-1, 4]+0.1, color='blue', marker='*', s=200, zorder=3)

    # Draw a rectangle around the zoomed-in region
    mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    plt.show()
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

    ax1.set_xlabel(r'$x$ [mm]', fontsize=14)
    ax1.set_ylabel(r'$y$ [mm]', fontsize=14)
    ax1.set_zlabel(r'$z$ [mm]', fontsize=14)
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
    ax2.plot(t_scp, z_scp[:, 3], 'tab:blue', label='SSMR (Ours)', linewidth=3)
    ax2.plot(t_target, zf_target[:, 3], '--k', alpha=1, linewidth=1, label='Target')
    if plot_rompc:
        ax2.plot(t_rompc, z_rompc[:, 3], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    ax2.plot(t_target, x_ub * np.ones_like(t_target), 'r', label='Constraint')
    ax2.plot(t_target, x_lb * np.ones_like(t_target), 'r')
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
ax2.set_xlim([0, 9])
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

    ax3.plot(t_target, y_ub * np.ones_like(t_target), 'r')
    ax3.plot(t_target, y_lb * np.ones_like(t_target), 'r')

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
ax3.set_xlim([0, 9])
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

fig5 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
ax_tube1 = fig5.add_subplot(111)
for idx in range(np.shape(s_delta_scp)[0]):
    if idx % 1 == 0:
        s_horizon = s_delta_scp[idx]
        t_horizon = t_opt_rollout[idx]
        ax_tube1.plot(t_horizon, s_horizon[:, -2] + 0.09 * s_horizon[:, -1], 'tab:green', markevery=2)
plt.ylabel(r'$||\mathbf{G}_j||L_{\mathbf{C_w}}\delta + ||\mathbf{G}_j \mathbf{C}||s}$', fontsize=14)

# Separate plots of s and delta
# fig5 = plt.figure(figsize=(14, 12), facecolor='w', edgecolor='k')
# ax_tube1 = fig5.add_subplot(211)
# for idx in range(np.shape(s_delta_scp)[0]):
#     if idx % 1 == 0:
#         s_horizon = s_delta_scp[idx]
#         t_horizon = t_opt_rollout[idx]
#         ax_tube1.plot(t_horizon, s_horizon[:, -2], 'tab:green', markevery=2)
# plt.ylabel(r'$s$', fontsize=14)
#
# ax_tube2 = fig5.add_subplot(212)
# for idx in range(np.shape(s_delta_scp)[0]):
#     if idx % 1 == 0:
#         s_horizon = s_delta_scp[idx]
#         t_horizon = t_opt_rollout[idx]
#         ax_tube2.plot(t_horizon, s_horizon[:, -1], 'tab:orange', markevery=2)
# plt.ylabel(r'$\delta$', fontsize=14)

ax_tube1.set_xlim([0, 9])
ax_tube1.tick_params(axis='both', labelsize=18)
plt.xlabel(r'$t$ [s]', fontsize=14)
# ax_tube2.set_xlim([0, 9])
# ax_tube2.tick_params(axis='both', labelsize=18)
# plt.xlabel(r'$t$ [s]', fontsize=14)

plt.show()
