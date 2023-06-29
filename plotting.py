from os.path import dirname, abspath, join, exists, split
import os
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

# import pdb
import yaml
# from sofacontrol.utils import load_data, set_axes_equal
import pickle

from sofacontrol.measurement_models import linearModel
from sofacontrol.utils import qv2x, load_data


path = dirname(abspath(__file__))
np.set_printoptions(linewidth=300)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'FreeSerif'})
plt.rcParams.update({'mathtext.fontset': 'cm'})

FONTSCALE = 1.2
PAD_LIMS = 7

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
    from examples.trunk import model
elif SETTINGS['robot'] == "hardware":
    from examples.hardware import model
else:
    raise RuntimeError("could not find model for robot specified in plotting_settings.yaml")

CONTROLS = [control for control in SETTINGS['show'] if SETTINGS['show'][control]]
SIM_DATA = {control: {'info': {}} for control in CONTROLS}

t0 = 1
for i, control in enumerate(CONTROLS):
    with open(join(path, "examples", SETTINGS['robot'], SETTINGS['traj_dir'], f'{control}_sim.pkl'), 'rb') as f:
    # with open(join("/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_analysis/trunk_adiabatic_10ms_N=100_sparsity=0.95/020", f'ssmr_sim_{i}.pkl'), 'rb') as f:
        control_data = pickle.load(f)
    idx = np.argwhere(control_data['t'] > t0)[0][0]
    SIM_DATA[control]['t'] = control_data['t'][idx:] - control_data['t'][idx]
    SIM_DATA[control]['z'] = control_data['z'][idx:, 3:]
    SIM_DATA[control]['z'][:, 2] *= -1
    SIM_DATA[control]['u'] = control_data['u'][idx:, :]
    SIM_DATA[control]['info']['solve_times'] = control_data['info']['solve_times']
    SIM_DATA[control]['info']['real_time_limit'] = control_data['info']['rollout_time']

print("=== SOFA equilibrium point ===")
# Load equilibrium point
print(model.QV_EQUILIBRIUM[0].shape, model.QV_EQUILIBRIUM[1].shape)
x_eq = qv2x(q=model.QV_EQUILIBRIUM[0], v=model.QV_EQUILIBRIUM[1])
outputModel = linearModel([model.TIP_NODE], model.N_NODES, vel=False)
Z_EQ = outputModel.evaluate(x_eq, qv=False)
Z_EQ[2] *= -1
print(Z_EQ)

# Load reference/target trajectory as defined in plotting_settings.py
TARGET = split(SETTINGS['traj_dir'])[1]
target_settings = SETTINGS['define_targets'][TARGET]
M, T, N, radius = (target_settings[key] for key in ['M', 'T', 'N', 'radius'])
t_target = np.linspace(0, M*T, M*N+1)
th = np.linspace(0, M*2*np.pi, M*N+1) # + np.pi/2
z_target = np.zeros((M*N+1, len(Z_EQ)))
if TARGET == "circle":
    z_target[:, 0] += radius * np.cos(th)
    z_target[:, 1] += radius * np.sin(th)
    z_target[:, 2] += -np.ones(len(t_target)) * target_settings['z_const']
elif TARGET == "figure8":
    z_target[:, 0] += -radius * np.sin(th)
    z_target[:, 1] += radius * np.sin(2 * th)
    z_target[:, 2] += -np.ones(len(t_target)) * target_settings['z_const']
else:
    z_target = load_data()

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
                ls=SETTINGS['linestyle'][control], markevery=20,
                alpha=1.)
    ax.plot(z_target[:, 0], z_target[:, 1], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)

    ax.set_xlabel(r'$x_{ee}$ [mm]')
    ax.set_ylabel(r'$y_{ee}$ [mm]')

    # Remove top and right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.legend()
    ax.set_aspect('equal', 'box')

    ax.set_xlim(np.min(z_target[:, 0]) - PAD_LIMS, np.max(z_target[:, 0]) + PAD_LIMS)
    ax.set_ylim(np.min(z_target[:, 1]) - PAD_LIMS, np.max(z_target[:, 1]) + PAD_LIMS)

    ax.tick_params(axis='both')

    plt.savefig(join(SAVE_DIR, f"{TARGET}_x_vs_y.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
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
    ax.plot(z_target[:, 0], z_target[:, 1], z_target[:, 2],
            color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.8, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)

    ax.set_xlabel(r'$x_{ee}$ [mm]')
    ax.set_ylabel(r'$y_{ee}$ [mm]')
    ax.set_zlabel(r'$z_{ee}$ [mm]')

    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    ax.view_init(5, -90)

    ax.tick_params(axis='both')

    plt.savefig(join(SAVE_DIR, f"{TARGET}_3D.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
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
        ax.plot(t_target, z_target[:, coord], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.7, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=3)
        ax.set_xlim([t_target[0], t_target[-1]])
        ax.set_ylim(np.min(z_target[:, coord]) - PAD_LIMS, np.max(z_target[:, coord]) + PAD_LIMS)
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
    plt.savefig(join(SAVE_DIR, f"{TARGET}_xy_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if SHOW_PLOTS:
        plt.show()


def traj_xyz_vs_t():
    """Plot trajectories (x,y,z) as function of time"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), facecolor='w', edgecolor='k', sharex=True)

    for ax, coord in [(ax1, 0), (ax2, 1), (ax3, 2)]:
        for control in CONTROLS: # + ['target']:
                z_centered = SIM_DATA[control]['z'] - Z_EQ
                ax.plot(SIM_DATA[control]['t'], z_centered[:, coord],
                        color=SETTINGS['color'][control],
                        label=SETTINGS['display_name'][control],
                        linewidth=SETTINGS['linewidth'][control],
                        ls=SETTINGS['linestyle'][control], markevery=20)
        ax.plot(t_target, z_target[:, coord], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.7, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=2)
        ax.set_xlim([t_target[0], t_target[-1]])
        ax.set_ylim(np.min(z_target[:, coord]) - PAD_LIMS, np.max(z_target[:, coord]) + PAD_LIMS)
    ax1.set_ylabel(r'$x_{ee}$ [mm]')
    ax2.set_ylabel(r'$y_{ee}$ [mm]')
    ax3.set_ylabel(r'$z_{ee}$ [mm]')
    ax3.set_xlabel(r'$t$ [s]')


    if SETTINGS['plot_mpc_rollouts']:
        idx = 0
        for idx in range(np.shape(z_opt_rollout)[0]):
            if idx % 2 == 0:
                z_horizon = z_opt_rollout[idx]
                t_horizon = t_opt_rollout[idx]
                ax1.plot(t_horizon, z_horizon[:, 0], 'tab:red', marker='o', markevery=2)
                ax2.plot(t_horizon, z_horizon[:, 1], 'tab:red', marker='o', markevery=2)
    
    ax3.legend()
    plt.savefig(join(SAVE_DIR, f"{TARGET}_xyz_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if SHOW_PLOTS:
        plt.show()


def traj_inputs_vs_t():
    """Plot inputs applied by controller as function of time"""
    fig, axs = plt.subplots(1, len(CONTROLS), figsize=(18, 6), facecolor='w', edgecolor='k', sharey=True, )
    if len(CONTROLS) == 1:
        axs = [axs]

    for i, control in enumerate(CONTROLS):
        axs[i].plot(SIM_DATA[control]['t'], SIM_DATA[control]['u'],
                    label=SETTINGS['display_name'][control],
                    linewidth=SETTINGS['linewidth'][control],
                    ls=SETTINGS['linestyle'][control], markevery=20)
        axs[i].legend([rf"$u_{i}$" for i in range(1, SIM_DATA[control]['u'].shape[1]+1)])
        axs[i].set_xlabel(r'$t$ [s]')
        axs[i].set_title(SETTINGS['display_name'][control])
    axs[0].set_ylabel(rf'$u$')
    plt.savefig(join(SAVE_DIR, f"{TARGET}_inputs_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if SHOW_PLOTS:
        plt.show()


def rmse_calculations():
    """Compute, display and plot RMSEs for all controllers"""
    # TODO: Remove constraints from error and compute num violations / num points
    # idx represents where the constraints are active. idx_viol_x represent where
    # the constraints are violated
    # constraint_idx = 4
    # if constrained:
    #     idx_koop = np.argwhere(zd_koop[:, constraint_idx] >= y_ub)
    #     viol_koop = np.count_nonzero(z_koop[idx_koop.flatten(), constraint_idx] > y_ub + 0.2) / idx_koop.size
    #     idx_viol_koop = np.argwhere(z_koop[idx_koop.flatten(), constraint_idx] > y_ub + 0.2)
    #     zd_koop = np.delete(zd_koop, idx_viol_koop, axis=0)
    #     z_koop = np.delete(z_koop, idx_viol_koop, axis=0)

    err = {}
    rmse = {}
    solve_times = {}

    for control in CONTROLS:
        print(control)
        # TODO: distinguish depending on target traj, if error computed in 2D or 3D
        z_centered = SIM_DATA[control]['z'] - Z_EQ
        # if control == "ssmr_origin":
        #     z_centered = z_centered[:-1, :]
        if TARGET == "circle":
            # errors are to be measured in 3D
            err[control] = (z_centered - z_target)
        else:
            # errors are to be measured in 2D
            err[control] = (z_centered[:, :2] - z_target[:, :2])
        rmse[control] = np.sqrt(np.mean(np.linalg.norm(err[control], axis=1)**2, axis=0))
        solve_times[control] = 1000 * np.array(SIM_DATA[control]['info']['solve_times'])

        print(f"========= {SETTINGS['display_name'][control]} =========")
        print(f"RMSE: {rmse[control]:.3f} mm")
        print(f"Solve time: Min: {np.min(solve_times[control]):.3f} ms, Mean: {np.mean(solve_times[control]):.3f} ms, Max: {np.max(solve_times[control]):.3f} ms")

    # Plot RMSEs and solve times (barplot)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    ax1.bar(xlabels, [rmse[control] for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    ax1.set_ylabel(r'RMSE [mm]')
    ax1.set_title('RMSE')
    # ax2.bar(xlabels, [np.mean(solve_times[control]) for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    # ax2.set_ylabel(r'Solve time [ms]')
    # ax2.set_title('Average solve time')
    plt.savefig(join(SAVE_DIR, f"{TARGET}_rmse_and_solve_times.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if SHOW_PLOTS:
        plt.show()


def rmse_vs_n_models(n_models, z, solve_times=None, save_dir="", show=True):
    """Compute and plot RMSEs for different number of models"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', 'v', 'D', 'P', 'X']

    for i, control in enumerate(z.keys()):
        z_centered = np.array(z[control]) - Z_EQ
        error = z_centered[:, :, :, :2] - z_target[:, :2]
        rmse = np.sqrt(np.mean(np.linalg.norm(error, axis=-1)**2, axis=-1))
        ax.plot(n_models[control], np.mean(rmse, axis=-1), color=colors[i], marker=markers[i], label=control.upper())
        ax.fill_between(n_models[control], np.min(rmse, axis=-1), np.max(rmse, axis=-1), color=colors[i], alpha=.25)
    
    ax.set_xlabel(r'$N_{models}$')
    ax.set_ylabel(r'RMSE [mm]')
    ax.set_title('RMSE vs. number of models')
    ax.grid(True, alpha=.5)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.legend()
    
    if save_dir:
        plt.savefig(join(save_dir, f"{TARGET}_rmse_vs_n_models.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    
    # if solve_times is not None:
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    #     ax.plot(n_models, solve_times, 'k', marker='o', markevery=1)
    #     ax.set_xlabel(r'$N_{models}$')
    #     ax.set_ylabel(r'Average solve time [ms]')
    #     ax.set_title('Solve time vs. number of models')
    #     ax.grid(True, alpha=.5)
    #     if save_dir:
    #         plt.savefig(join(save_dir, f"{TARGET}_solve_times_vs_n_models.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    #     if show:
    #         plt.show()


if __name__ == "__main__":
    traj_x_vs_y()
    traj_xy_vs_t()
    traj_xyz_vs_t()
    rmse_calculations()
    traj_3D()
    traj_inputs_vs_t()