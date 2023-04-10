from os.path import dirname, abspath, join
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
# import pdb
import yaml
# from sofacontrol.utils import load_data, set_axes_equal
import pickle

from sofacontrol.measurement_models import linearModel
from sofacontrol.utils import qv2x

path = dirname(abspath(__file__))
np.set_printoptions(linewidth=300)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'FreeSerif'})
plt.rcParams.update({'mathtext.fontset': 'cm'})

FONTSCALE = 1.2

plt.rc('font', size=12*FONTSCALE)          # controls default text sizes
plt.rc('axes', titlesize=15*FONTSCALE)     # fontsize of the axes title
plt.rc('axes', labelsize=13*FONTSCALE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('legend', fontsize=8*FONTSCALE)    # legend fontsize
plt.rc('figure', titlesize=15*FONTSCALE)   # fontsize of the figure title
suptitlesize = 20*FONTSCALE

plt.rc('figure', autolayout=True)

with open(join(path, "plotting_settings.yaml"), "rb") as f:
    SETTINGS = yaml.safe_load(f)

CONTROLS = [control for control in SETTINGS['show'] if SETTINGS['show'][control]]
SIM_DATA = {control: {'info': {}} for control in CONTROLS}

t_in = 1
t_out = 7
t0 = 3
for control in CONTROLS:
    with open(join(path, f'{control}_sim.pkl'), 'rb') as f:
        control_data = pickle.load(f)
    idx = np.argwhere(control_data['t'] >= t0)[0][0]
    SIM_DATA[control]['t'] = control_data['t'][idx:] - control_data['t'][idx]
    SIM_DATA[control]['z'] = control_data['z'][idx:, :]
    SIM_DATA[control]['u'] = control_data['u'][idx:, :]
    SIM_DATA[control]['info']['solve_times'] = control_data['info']['solve_times']
    SIM_DATA[control]['info']['real_time_limit'] = control_data['info']['rollout_time']

print("=== SOFA equilibrium point ===")
# Load equilibrium point
TIP_NODE = 1354
with open(join(path, 'rest_qv.pkl'), 'rb') as f:
    qv_equilibrium = pickle.load(f)['rest']
x_eq = qv2x(q=qv_equilibrium[0], v=qv_equilibrium[1])
outputModel = linearModel([TIP_NODE], 1628, vel=False)
z_eq_point = outputModel.evaluate(x_eq, qv=False)# print(rest_data.keys())
print(z_eq_point)

# Load reference/target trajectory as defined in plotting_settings.py
M, T, N, radius = (SETTINGS['target'][key] for key in ['M', 'T', 'N', 'radius'])
t_target = np.linspace(0, M*T, M*N+1)
th = np.linspace(0, M*2*np.pi, M*N+1)
zf_target = np.tile(z_eq_point, (M*N+1, 1))
zf_target[:, 0] += -radius * np.sin(th)
zf_target[:, 1] += radius * np.sin(2 * th)
z_lb = SETTINGS['target']['z_lb']
z_ub = SETTINGS['target']['z_ub']
# print(t_target)

# constrained = True
# plot_rompc = False
# if constrained:
#     y_ub = 15

# opt_controller = 'ssmr'
# print(SIM_DATA[opt_controller]['info'].keys())
# z_opt_rollout = SIM_DATA[opt_controller]['info']['z_rollout']
# t_opt_rollout = SIM_DATA[opt_controller]['info']['t_rollout']
# plot_rollouts = True
# m_w = 30


def diamond_figure8_x_vs_y():
    """Plot Figure8 via x vs. y"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='w', edgecolor='k')

    if z_lb is not None and z_ub is not None:
        ax.add_patch(
            patches.Rectangle(
                xy=(z_lb[0], z_lb[1]),  # point of origin.
                width=z_ub[0] - z_lb[0],
                height=z_ub[1] - z_lb[1],
                linewidth=2,
                color='tab:red',
                fill=False))
    
    for control in CONTROLS: # + ['target']:
        ax.plot(SIM_DATA[control]['z'][:, 3], SIM_DATA[control]['z'][:, 4],
                color=SETTINGS['color'][control],
                label=SETTINGS['display_name'][control],
                linewidth=SETTINGS['linewidth'][control],
                ls=SETTINGS['linestyle'][control], markevery=20)
    ax.plot(zf_target[:, 0], zf_target[:, 1], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.8, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)

    ax.set_xlabel(r'$x_{ee}$ [mm]')
    ax.set_ylabel(r'$y_{ee}$ [mm]')

    # Remove top and right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.legend()
    # plt.axis('off')
    # plt.legend(loc='upper left', prop={'size': 14}, borderaxespad=0, bbox_to_anchor=(0.25, 0.12))
    ax.tick_params(axis='both')

    plt.savefig(join(path, 'diamond_x_vs_y.pdf'), bbox_inches='tight')

# else:
#     ax1 = fig1.add_subplot(111, projection='3d')
#     # z_lb = np.array([-30 - 0, -30 + 127.])
#     # z_ub = np.array([30 - 0, 30 + 127.])

#     # ax1.add_patch(
#     #     patches.Rectangle(
#     #         xy=(z_lb[0], z_lb[1]),  # point of origin.
#     #         width=z_ub[0] - z_lb[0],
#     #         height=z_ub[1] - z_lb[1],
#     #         linewidth=2,
#     #         color='tab:red',
#     #         fill=False,
#     #     )
#     # )

#     ax1.plot3D(z_tpwl[:, 3], z_tpwl[:, 4], z_tpwl[:, 5], 'tab:green', marker='x', markevery=20, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
#     ax1.plot3D(z_koop[:, 3], z_koop[:, 4], z_koop[:, 5], 'tab:orange', marker='^', markevery=20, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
#     ax1.plot3D(z_scp[:, 3], z_scp[:, 4], z_scp[:, 5], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
#     ax1.plot3D(zf_target[:, 3], zf_target[:, 4], zf_target[:, 5], '--k', alpha=1, linewidth=1, label='Target')
#     if plot_rompc:
#         ax1.plot3D(z_rompc[:, 3], z_rompc[:, 4], z_rompc[:, 5], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

#     ax1.set_xlabel(r'$x_{ee}$ [mm]', fontsize=14)
#     ax1.set_ylabel(r'$y_{ee}$ [mm]', fontsize=14)
#     ax1.set_zlabel(r'$z_{ee}$ [mm]', fontsize=14)
#     set_axes_equal(ax1)

#     # startx, endx = ax1.get_xlim()
#     # starty, endy = ax1.get_ylim()
#     # startz, endz = ax1.get_zlim()
#     # ax1.xaxis.set_ticks(np.arange(int(startx),int(endx),10))
#     # ax1.yaxis.set_ticks(np.arange(int(starty),int(endy),10))
#     # ax1.zaxis.set_ticks(np.arange(int(startz),int(endz),10))


def diamond_figure8_xy_vs_t():
    """Plot controlled trajectories as function of time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='w', edgecolor='k', sharex=True)

    for ax, coord in [(ax1, 3), (ax2, 4)]:
        for control in CONTROLS: # + ['target']:
                ax.plot(SIM_DATA[control]['t'], SIM_DATA[control]['z'][:, coord],
                        color=SETTINGS['color'][control],
                        label=SETTINGS['display_name'][control],
                        linewidth=SETTINGS['linewidth'][control],
                        ls=SETTINGS['linestyle'][control], markevery=20)
        ax.plot(t_target, zf_target[:, coord-3], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.8, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
    ax1.set_ylabel(r'$x_{ee}$ [mm]')
    ax2.set_ylabel(r'$y_{ee}$ [mm]')
    ax2.set_xlabel(r'$t$ [s]')


    if SETTINGS['plot_mpc_rollouts']:
        idx = 0
        for idx in range(np.shape(z_opt_rollout)[0]):
            if idx % 2 == 0:
                z_horizon = z_opt_rollout[idx]
                t_horizon = t_opt_rollout[idx]
                ax1.plot(t_horizon, z_horizon[:, 0], 'tab:red', marker='o', markevery=2)
                ax2.plot(t_horizon, z_horizon[:, 1], 'tab:red', marker='o', markevery=2)
    
    ax2.legend()
    plt.savefig(join(path, 'diamond_xy_vs_t.pdf'), bbox_inches='tight')
    # plt.show()

    # else:
    #     ax2.plot(t_tpwl, z_tpwl[:, 4], 'tab:green', marker='x', markevery=m_w, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    #     ax2.plot(t_koop, z_koop[:, 4], 'tab:orange', marker='^', markevery=m_w, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    #     ax2.plot(t_scp, z_scp[:, 4], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    #     ax2.plot(t_target, zf_target[:, 4], '--k', alpha=1, linewidth=1, label='Target')
    #     if plot_rompc:
    #         ax2.plot(t_rompc, z_rompc[:, 4], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    #     plt.ylabel(r'$y_{ee}$ [mm]', fontsize=14)
    # ax2.set_xlim([0, 10])
    # ax2.tick_params(axis='both', labelsize=18)
    # plt.xlabel(r'$t$ [s]', fontsize=14)
    # #plt.legend(loc='upper left', prop={'size': 12})

    # if constrained:
    #     ax3.plot(t_target, y_ub * np.ones_like(t_target), 'r', label='Constraint')

    # else:
    #     ax3.plot(t_tpwl, z_tpwl[:, 5], 'tab:green', marker='x', markevery=m_w, label='TPWL ($N_r = 3$, $dt = 0.1$ s)', linewidth=1)
    #     ax3.plot(t_koop, z_koop[:, 5], 'tab:orange', marker='^', markevery=m_w, label='Koopman ($N_r = 1$, $dt = 0.05$ s)', linewidth=1)
    #     ax3.plot(t_scp, z_scp[:, 5], 'tab:blue', label='SSMR (Ours) ($N_r = 2$, $dt = 0.03$ s)', linewidth=3)
    #     ax3.plot(t_target, zf_target[:, 5], '--k', alpha=1, linewidth=1, label='Target')
    #     if plot_rompc:
    #         ax3.plot(t_rompc, z_rompc[:, 5], 'tab:red', marker='d', markevery=20, label='ROMPC CL', linewidth=1)

    #     plt.ylabel(r'$z_{ee}$ [mm]', fontsize=14)
    # ax3.set_xlim([0, 10])
    # ax3.tick_params(axis='both', labelsize=18)
    # plt.xlabel(r'$t$ [s]', fontsize=14)
    # #plt.legend(loc='upper left', prop={'size': 14}, borderaxespad=0, bbox_to_anchor=(0, 1.2))


def mse_calculations():
    """MSE calculations of desired trajectory"""

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


if __name__ == "__main__":
    diamond_figure8_x_vs_y()
    diamond_figure8_xy_vs_t()