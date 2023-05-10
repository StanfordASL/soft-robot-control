from os.path import dirname, abspath, join, exists
import os
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

SHOW_PLOTS = True

with open(join(path, "plotting_settings.yaml"), "rb") as f:
    SETTINGS = yaml.safe_load(f)

if SETTINGS['robot'] == "trunk":
    from trunk import model
elif SETTINGS['robot'] == "hardware":
    from hardware import model
else:
    raise RuntimeError("could not find model for robot specified in plotting_settings.yaml")

CONTROLS = [control for control in SETTINGS['show'] if SETTINGS['show'][control]]
SIM_DATA = {control: {'info': {}} for control in CONTROLS}

t0 = 1
for control in CONTROLS:
    with open(join(path, SETTINGS['robot'], f'{control}_sim.pkl'), 'rb') as f:
        control_data = pickle.load(f)
    idx = np.argwhere(control_data['t'] >= t0)[0][0]
    SIM_DATA[control]['t'] = control_data['t'][idx:] - control_data['t'][idx]
    SIM_DATA[control]['z'] = control_data['z'][idx:, 3:]
    SIM_DATA[control]['u'] = control_data['u'][idx:, :]
    SIM_DATA[control]['info']['solve_times'] = control_data['info']['solve_times']
    SIM_DATA[control]['info']['real_time_limit'] = control_data['info']['rollout_time']

print("=== SOFA equilibrium point ===")
# Load equilibrium point
print(model.QV_EQUILIBRIUM[0].shape, model.QV_EQUILIBRIUM[1].shape)
x_eq = qv2x(q=model.QV_EQUILIBRIUM[0], v=model.QV_EQUILIBRIUM[1])
outputModel = linearModel([model.TIP_NODE], model.N_NODES, vel=False)
Z_EQ = outputModel.evaluate(x_eq, qv=False)
print(Z_EQ)

# Load reference/target trajectory as defined in plotting_settings.py
TARGET = SETTINGS['select_target']
target_settings = SETTINGS['define_targets'][TARGET]
M, T, N, radius = (target_settings[key] for key in ['M', 'T', 'N', 'radius'])
t_target = np.linspace(0, M*T, M*N+1)
th = np.linspace(0, M*2*np.pi, M*N+1) # + np.pi/2
# zf_target = np.tile(z_eq_point, (M*N+1, 1))
zf_target = np.zeros((M*N+1, len(Z_EQ)))
if TARGET == "circle":
    zf_target[:, 0] += radius * np.cos(th)
    zf_target[:, 1] += radius * np.sin(th)
    zf_target[:, 2] += np.ones(len(t_target)) * target_settings['z_const']
elif TARGET == "figure8":
    zf_target[:, 0] += -radius * np.sin(th)
    zf_target[:, 1] += radius * np.sin(2 * th)
z_lb = target_settings['z_lb']
z_ub = target_settings['z_ub']

SAVE_DIR = join(path, SETTINGS['robot'], SETTINGS['save_dir'])
if not exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

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


def traj_x_vs_y():
    """Plot trajectory via x vs. y"""

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
        z_centered = SIM_DATA[control]['z'] - Z_EQ
        ax.plot(z_centered[:, 0], z_centered[:, 1],
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
    ax.set_aspect('equal', 'box')

    # plt.axis('off')
    # plt.legend(loc='upper left', prop={'size': 14}, borderaxespad=0, bbox_to_anchor=(0.25, 0.12))
    ax.tick_params(axis='both')

    plt.savefig(join(SAVE_DIR, f"{TARGET}_x_vs_y.{SETTINGS['file_format']}"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()

def traj_3D():

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    
    for control in CONTROLS: # + ['target']:
        z_centered = SIM_DATA[control]['z'] - Z_EQ
        ax.plot(z_centered[:, 0], z_centered[:, 1], z_centered[:, 2],
                color=SETTINGS['color'][control],
                label=SETTINGS['display_name'][control],
                linewidth=SETTINGS['linewidth'][control],
                ls=SETTINGS['linestyle'][control], markevery=20)
    ax.plot(zf_target[:, 0], zf_target[:, 1], zf_target[:, 2],
            color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.8, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)

    ax.set_xlabel(r'$x_{ee}$ [mm]')
    ax.set_ylabel(r'$y_{ee}$ [mm]')
    ax.set_zlabel(r'$z_{ee}$ [mm]')

    ax.legend()
    ax.set_aspect('equal', 'box')

    ax.tick_params(axis='both')

    plt.savefig(join(SAVE_DIR, f"figure8_3D.{SETTINGS['file_format']}"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()

def traj_xy_vs_t():
    """Plot controlled trajectories as function of time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='w', edgecolor='k', sharex=True)

    for ax, coord in [(ax1, 0), (ax2, 1)]:
        for control in CONTROLS: # + ['target']:
                z_centered = SIM_DATA[control]['z'] - Z_EQ
                ax.plot(SIM_DATA[control]['t'], z_centered[:, coord],
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
    plt.savefig(join(SAVE_DIR, f"{TARGET}_xy_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()


def traj_inputs_vs_t():
    """Plot inputs applied by controller as function of time"""
    fig, axs = plt.subplots(1, len(CONTROLS), figsize=(10, 8), facecolor='w', edgecolor='k', sharex=True)
    if len(CONTROLS) == 1:
        axs = [axs]

    for i, control in enumerate(CONTROLS):
        axs[i].plot(SIM_DATA[control]['t'], SIM_DATA[control]['u'],
                    label=SETTINGS['display_name'][control],
                    linewidth=SETTINGS['linewidth'][control],
                    ls=SETTINGS['linestyle'][control], markevery=20)
        axs[i].legend([rf"$u_{i}$" for i in range(1, SIM_DATA[control]['u'].shape[1]+1)])
        axs[i].set_ylabel(rf'$u$ ({control})')
    
    axs[-1].set_xlabel(r'$t$ [s]')
    plt.savefig(join(SAVE_DIR, f"{TARGET}_inputs_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()


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
    traj_3D()
    traj_inputs_vs_t()
    traj_x_vs_y()
    traj_xy_vs_t()