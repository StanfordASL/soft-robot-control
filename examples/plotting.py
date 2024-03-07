from os.path import dirname, abspath, join, exists, split, isdir
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

TRAJ_LINEWIDTH = 1
TRAJ_LINESTYLE = '-'
MARKER = ''

plt.rc('font', size=12*FONTSCALE)          # controls default text sizes
plt.rc('axes', titlesize=15*FONTSCALE)     # fontsize of the axes title
plt.rc('axes', labelsize=13*FONTSCALE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('legend', fontsize=10*FONTSCALE)    # legend fontsize
plt.rc('figure', titlesize=15*FONTSCALE)   # fontsize of the figure title
suptitlesize = 20*FONTSCALE

plt.rc('figure', autolayout=True)
plt.rc('grid', color="#cccccc")
plt.rc('axes', grid=True)


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
    with open(join(path, SETTINGS['robot'], SETTINGS['traj_dir'], f'{control}_sim.pkl'), 'rb') as f:
    # with open(join("/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_analysis/trunk_adiabatic_10ms_N=100_sparsity=0.95/020", f'ssmr_sim_{i}.pkl'), 'rb') as f:
        control_data = pickle.load(f)
    idx = np.argwhere(control_data['t'] > t0)[0][0]
    SIM_DATA[control]['t'] = control_data['t'][idx:] - control_data['t'][idx]
    SIM_DATA[control]['z'] = control_data['z'][idx:, 3:]
    SIM_DATA[control]['z'][:, 2] *= -1
    SIM_DATA[control]['u'] = control_data['u'][idx:, :]
    # SIM_DATA[control]['info']['solve_times'] = control_data['info']['solve_times']
    # SIM_DATA[control]['info']['real_time_limit'] = control_data['info']['rollout_time']
    # if control == "tpwl":
    #     # remove first 100 points from z and u and shift t accordingly
    #     SIM_DATA[control]['z'] = SIM_DATA[control]['z'][100:, :]
    #     SIM_DATA[control]['u'] = SIM_DATA[control]['u'][100:, :]
    #     SIM_DATA[control]['t'] = SIM_DATA[control]['t'][100:] - SIM_DATA[control]['t'][100]

print("=== SOFA equilibrium point ===")
# Load equilibrium point
print(model.QV_EQUILIBRIUM[0].shape, model.QV_EQUILIBRIUM[1].shape)
x_eq = qv2x(q=model.QV_EQUILIBRIUM[0], v=model.QV_EQUILIBRIUM[1])
outputModel = linearModel([model.TIP_NODE], model.N_NODES, vel=False)
Z_EQ = outputModel.evaluate(x_eq, qv=False)
Z_EQ[2] *= -1
print(Z_EQ)

# Load reference/target trajectory as defined in plotting_settings.py
TARGET = SETTINGS['select_target']
target_settings = SETTINGS['define_targets'][TARGET]
M, T, N, radius = (target_settings[key] for key in ['M', 'T', 'N', 'radius'])
t_target = np.linspace(0, M*T, M*N+1)
th = np.linspace(0, M*2*np.pi, M*N+1) # + np.pi/2
z_target = np.zeros((M*N+1, len(Z_EQ)))
if TARGET == "circle_z=-10":
    z_target[:, 0] += radius * np.cos(th)
    z_target[:, 1] += radius * np.sin(th)
    z_target[:, 2] += -np.ones(len(t_target)) * target_settings['z_const']
elif TARGET == "figure8":
    z_target[:, 0] += -radius * np.sin(th)
    z_target[:, 1] += radius * np.sin(2 * th)
    z_target[:, 2] += -np.ones(len(t_target)) * target_settings['z_const']
elif TARGET == "pacman":
    z_target[:, 0] += radius * np.cos(th)
    z_target[:, 1] += radius * np.sin(th)
    z_target[:, 2] += -np.ones(len(t_target)) * target_settings['z_const']
    z_target[t_target < target_settings['t_in_pacman'], :] = z_target[t_target < target_settings['t_in_pacman']][-1, :] * (t_target[t_target < target_settings['t_in_pacman']] / target_settings['t_in_pacman'])[..., None]
    z_target[t_target > T - target_settings['t_out_pacman'], :] = z_target[t_target > T - target_settings['t_out_pacman']][0, :] * (1 - (t_target[t_target > T - target_settings['t_out_pacman']] - (T - target_settings['t_out_pacman'])) / target_settings['t_out_pacman'])[..., None]
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


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def traj_x_vs_y():
    """Plot trajectory via x vs. y"""

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='w', edgecolor='k')

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
                linewidth=SETTINGS['linewidth'].get(control, TRAJ_LINEWIDTH),
                ls=SETTINGS['linestyle'].get(control, TRAJ_LINESTYLE), markevery=20,
                alpha=1.)
    ax.plot(z_target[:, 0], z_target[:, 1],
            color=SETTINGS['color']['target'], alpha=0.9,
            ls=SETTINGS['linestyle'].get('target', TRAJ_LINESTYLE),
            lw=SETTINGS['linewidth'].get('target', TRAJ_LINEWIDTH),
            label='Target', zorder=3)

    ax.set_xlabel(r'$x_{ee}$ [mm]')
    ax.set_ylabel(r'$y_{ee}$ [mm]')

    # Remove top and right border
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if TARGET == "figure8":
        bbox = (0.73, 0.5)
    elif TARGET == "pac-man":
        bbox = (0.39, 0.5)
    else:
        bbox = None

    legend = ax.legend(loc="center", ncol=1, bbox_to_anchor=bbox)
    for label in legend.get_texts():
        if label.get_text() in ["MIDW", "QPR"]:
            label.set_weight('bold')

    ax.set_aspect('equal', 'box')

    ax.set_xlim(np.min(z_target[:, 0]) - PAD_LIMS, np.max(z_target[:, 0]) + PAD_LIMS)
    ax.set_ylim(np.min(z_target[:, 1]) - PAD_LIMS, np.max(z_target[:, 1]) + PAD_LIMS)

    ax.tick_params(axis='both')

    ax.grid(False)

    plt.savefig(join(SAVE_DIR, f"{TARGET}_x_vs_y.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()

def traj_3D():

    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')
    
    for control in CONTROLS: # +     traj_xy_vs_t()
        z_centered = SIM_DATA[control]['z'] - Z_EQ
        ax.plot(z_centered[:, 0], z_centered[:, 1], z_centered[:, 2],
                color=SETTINGS['color'][control],
                label=SETTINGS['display_name'][control],
                linewidth=SETTINGS['linewidth'].get(control, TRAJ_LINEWIDTH),
                ls=SETTINGS['linestyle'].get(control, TRAJ_LINESTYLE), marker=SETTINGS['markers'].get(control, MARKER), markevery=20)
    ax.plot(z_target[:, 0], z_target[:, 1], z_target[:, 2],
            color=SETTINGS['color']['target'], alpha=1.,
            ls=SETTINGS['linestyle'].get('target', TRAJ_LINESTYLE),
            lw=SETTINGS['linewidth'].get('target', TRAJ_LINEWIDTH),
            label='Target', zorder=1)

    ax.set_xlabel(r'$x_{ee}$ [mm]')
    ax.set_ylabel(r'$y_{ee}$ [mm]')
    ax.set_zlabel(r'$z_{ee}$ [mm]')

    ax.set_zlim(-10, 10)

    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    ax.view_init(5, -90)

    ax.tick_params(axis='both')

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    plt.savefig(join(SAVE_DIR, f"{TARGET}_3D.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()

def traj_xy_vs_t():
    """Plot controlled trajectories as function of time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), facecolor='w', edgecolor='k', sharex=True)

    for ax, coord in [(ax1, 0), (ax2, 1)]:
        for control in CONTROLS: # + ['target']:
                z_centered = SIM_DATA[control]['z'] - Z_EQ
                ax.plot(SIM_DATA[control]['t'], z_centered[:, coord],
                        color=SETTINGS['color'][control],
                        label=SETTINGS['display_name'][control],
                        linewidth=SETTINGS['linewidth'].get(control, TRAJ_LINEWIDTH),
                        ls=SETTINGS['linestyle'].get(control, TRAJ_LINESTYLE), marker=SETTINGS['markers'].get(control, MARKER), markevery=20)
        ax.plot(t_target, z_target[:, coord],
                color=SETTINGS['color']['target'], alpha=0.7,
                ls=SETTINGS['linestyle'].get('target', TRAJ_LINESTYLE),
                lw=SETTINGS['linewidth'].get('target', TRAJ_LINEWIDTH),
                label='Target', zorder=3)
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
    
    # ax2.legend()
    plt.savefig(join(SAVE_DIR, f"{TARGET}_xy_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()


def traj_xyz_vs_t():
    """Plot trajectories (x,y,z) as function of time"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), facecolor='w', edgecolor='k', sharex=True)

    for ax, coord in [(ax1, 0), (ax2, 1), (ax3, 2)]:
        for control in CONTROLS: # + ['target']:
                z_centered = SIM_DATA[control]['z'] - Z_EQ
                ax.plot(SIM_DATA[control]['t'], z_centered[:, coord],
                        color=SETTINGS['color'][control],
                        label=SETTINGS['display_name'][control],
                        linewidth=SETTINGS['linewidth'].get(control, TRAJ_LINEWIDTH),
                        ls=SETTINGS['linestyle'].get(control, TRAJ_LINESTYLE), markevery=20)
        ax.plot(t_target, z_target[:, coord],
                color=SETTINGS['color']['target'], alpha=0.7,
                ls=SETTINGS['linestyle'].get('target', TRAJ_LINESTYLE),
                lw=SETTINGS['linewidth'].get('target', TRAJ_LINEWIDTH),
                label='Target', zorder=3)
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
    
    # ax3.legend()
    plt.savefig(join(SAVE_DIR, f"{TARGET}_xyz_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()


def traj_inputs_vs_t():
    """Plot inputs applied by controller as function of time"""
    fig, axs = plt.subplots(1, len(CONTROLS), figsize=(12, 3), facecolor='w', edgecolor='k', sharey=True, sharex=True)
    if len(CONTROLS) == 1:
        axs = [axs]

    # colors = ["indigo", "tab:blue"]

    for i, control in enumerate(CONTROLS):
        axs[i].plot(SIM_DATA[control]['t'], SIM_DATA[control]['u'][:, :4] / 800, # TODO: hardcoded u_max
                    linewidth=SETTINGS['linewidth'].get(control, TRAJ_LINEWIDTH),
                    color=SETTINGS['color'][control],
                    ls=SETTINGS['linestyle'].get(control, TRAJ_LINESTYLE), markevery=20)
        axs[i].plot(SIM_DATA[control]['t'], SIM_DATA[control]['u'][:, 4:] / 800, # TODO: hardcoded u_max
                    color=adjust_lightness(SETTINGS['color'][control], 0.7),
                    linewidth=SETTINGS['linewidth'].get(control, TRAJ_LINEWIDTH),
                    ls=SETTINGS['linestyle'].get(control, TRAJ_LINESTYLE), markevery=20)

        axs[i].set_xlim([t_target[0], t_target[-1]])
        axs[i].set_xlabel(r'$t$ [s]')
        axs[i].set_title(SETTINGS['display_name'][control], fontweight=('bold' if SETTINGS['display_name'][control] in ["MIDW", "QPR"] else 'normal'))

        print(f"========= {SETTINGS['display_name'][control]} =========")
        print(np.linalg.norm(SIM_DATA[control]['u'], axis=1).shape)
        print(f"mean |u(t)|: {np.mean(np.linalg.norm(SIM_DATA[control]['u'], axis=1)):.3f}")

    axs[0].set_ylabel(r'$u/u_{max}$')
    plt.savefig(join(SAVE_DIR, f"{TARGET}_inputs_vs_t.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()


def rmse_calculations(plot_solve_times=True):
    """Compute, display and plot RMSEs for all controllers"""
    err = {}
    rmse = {}
    solve_times = {}

    for control in CONTROLS:
        print(control)
        # TODO: distinguish depending on target traj, if error computed in 2D or 3D
        z_centered = SIM_DATA[control]['z'] - Z_EQ
        # if control == "ssmr_origin":
        #     z_centered = z_centered[:-1, :]
        if TARGET in ["circle", "pac-man"]:
            # errors are to be measured in 3D
            err[control] = (z_centered - z_target)
        else:
            # errors are to be measured in 2D
            err[control] = (z_centered[:, :2] - z_target[:-1, :2])
        rmse[control] = np.sqrt(np.mean(np.linalg.norm(err[control], axis=1)**2, axis=0))
        # solve_times[control] = 1000 * np.array(SIM_DATA[control]['info']['solve_times'])

        print(f"========= {SETTINGS['display_name'][control]} =========")
        print(f"RMSE: {rmse[control]:.3f} mm")
        # print(f"Solve time: Min: {np.min(solve_times[control]):.3f} ms, Mean: {np.mean(solve_times[control]):.3f} ms, Max: {np.max(solve_times[control]):.3f} ms")

    # Plot RMSEs and solve times (barplot)
    if plot_solve_times:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharex=False)
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=False)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    ax1.bar(xlabels, [rmse[control] for control in CONTROLS], width=0.7, color=[SETTINGS['color'][control] for control in CONTROLS], zorder=3)
    ax1.set_ylabel(r'RMSE [mm]')
    # ax1.set_title('RMSE')
    ax1.grid(axis="x")
 
    ax1.set_ylim(0, min(5, max([rmse[control] for control in CONTROLS]) + 0.2))
    y_max = ax1.get_ylim()[1]

    # annotate bars in barplot
    for i, v in enumerate([rmse[control] for control in CONTROLS]):
        ax1.text(i, min(v + 0.1 * (y_max / 7), 4.6), f"{v:.2f}", color='#5c5c5c', fontweight='bold', ha='center')

    # ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    if plot_solve_times:
        # ax2.bar(xlabels, [np.mean(solve_times[control]) for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS], zorder=3)
        medianprops = dict(linestyle='')
        box_plot = ax2.boxplot([solve_times[control] for control in CONTROLS], medianprops=medianprops,
                               notch=False, patch_artist=True, labels=xlabels, showfliers=False, zorder=2, vert=True)
        # for i, median in enumerate(box_plot['medians']):
        #     median.set_color(SETTINGS['color'][CONTROLS[i]])
        #     # median.set_linewidth(3)
        for i, box in enumerate(box_plot['boxes']):
            box.set_facecolor(SETTINGS['color'][CONTROLS[i]])
            box.set_edgecolor(SETTINGS['color'][CONTROLS[i]])
        print(len(box_plot['caps']))
        for i in range(len(box_plot['caps'])):
            cap = box_plot['caps'][i]
            whisker = box_plot['whiskers'][i]
            cap.set_color(SETTINGS['color'][CONTROLS[i//2]])
            whisker.set_color(SETTINGS['color'][CONTROLS[i//2]])
            cap.set_linewidth(1.5)
            whisker.set_linewidth(1.5)
        ax2.set_ylabel(r'Solve time [ms]')
        # ax2.set_title('Average solve time')
        ax2.set_ylim(0, None)
        ax2.grid(axis="x")
        # ax2.yaxis.set_label_position("right")
        # ax2.yaxis.tick_right()

    for label in ax1.get_xticklabels() + ax2.get_xticklabels():
        if label.get_text() in ["MIDW", "QPR"]:
            label.set_weight('bold')

    plt.savefig(join(SAVE_DIR, f"{TARGET}_rmse_and_solve_times.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()


def rmse_vs_n_models(baselines=False, plot_solve_times=True):
    """Compute and plot RMSEs for different number of models"""
    # import all data
    sims_per_model = 10
    z = {}
    solve_times = {}
    n_models = {}
    use_models = {}
    for control in ["modified_idw", "qp"]: # ["idw", "nn", "qp"]: # "ct", 
        result_dir = join("/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_closed-loop_analysis/10-sims-per-n-models", TARGET, f"adiabatic_ssm_{control}")
        z[control] = []
        solve_times[control] = []
        n_models[control] = []
        use_models[control] = []
        dirs = [name for name in sorted(os.listdir(result_dir)) if isdir(join(result_dir, name))]
        t0 = 1
        for dir in dirs:
            z_i = []
            solve_times_i = []
            n_models_i = []
            for j in range(sims_per_model):
                print(control, dir, j)
                with open(join(result_dir, dir, f"use_models_{j}.pkl"), "rb") as f:
                    models = pickle.load(f)
                use_models[control].append(models)
                if exists(join(result_dir, dir, f"sim_{j}.pkl")):
                    with open(join(result_dir, dir, f"sim_{j}.pkl"), "rb") as f:
                        sim = pickle.load(f)
                    idx = np.argwhere(sim['t'] > t0)[0][0]
                    z_ij = sim['z'][idx:, 3:]
                    z_ij[:, 2] *= -1
                    solve_times_ij = sim['info']['solve_times']
                elif exists(join(result_dir, dir, f"sim_{j}_failed.pkl")):
                    print("simulation failed: ", dir, j)
                    z_ij = np.full((1001, 3), np.nan)
                    solve_times_ij = np.full(251, np.nan)
                else:
                    raise RuntimeError(f"simulation not found: {dir}, {j}")
                z_i.append(z_ij)
                solve_times_i.append(solve_times_ij)
                n_models_i.append(len(models))
                # solve_times_i.append(np.mean(sim['info']['solve_times']))
            z[control].append(z_i)
            solve_times[control].append(solve_times_i)
            n_models[control].append(n_models_i[0])
            # solve_times[control].append(solve_times_i)
    # model_contribution_to_rmse(z, use_models, save_dir=split(SAVE_DIR)[0])
    # x_vs_y_bundle(z, save_dir=split(SAVE_DIR)[0])
    if plot_solve_times:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    markers = ['o', 'd', 'v', 'D', 'P', 'X']
    for i, control in enumerate(z.keys()):
        z_centered = np.array(z[control]) - Z_EQ
        if TARGET in ["pac-man", "circle"]:
            error = z_centered[:, :, :, :3] - z_target[:, :3]
        else:
            error = z_centered[:, :, :, :2] - z_target[:, :2]
        rmse = np.sqrt(np.mean(np.linalg.norm(error, axis=-1)**2, axis=-1))
        ax.boxplot(rmse.T, positions=n_models[control], widths=3, vert=True, showfliers=False, showmeans=False, whiskerprops=dict(linestyle=''), capprops=dict(linestyle=''),
                   medianprops=dict(linestyle='-', color='k'), patch_artist=True, boxprops=dict(facecolor=SETTINGS['color'][f"ssmr_adiabatic_{control}"]),
                #    whiskerprops=dict(color=colors[i]), capprops=dict(color=colors[i])
        )
        ax.plot(n_models[control], np.nanmedian(rmse, axis=-1),
                color=SETTINGS['color'][f"ssmr_adiabatic_{control}"], ls='-', markeredgecolor='k', marker=markers[i], label=SETTINGS['display_name'][f"ssmr_adiabatic_{control}"])
        ax.set_xticks(n_models[control])
        # ax.fill_between(n_models[control], np.nanmin(rmse, axis=-1), np.nanmax(rmse, axis=-1), color=colors[i], alpha=.25)
    if baselines:
        baselines_rmse = {
            "ssmr_origin": 1.83,
            # "tpwl": 2.18, 
            # "koopman": np.nan # 61.97
        }
        for i, control in enumerate(baselines_rmse.keys()):
            ax.axhline(baselines_rmse[control], color=SETTINGS['color'][control], linestyle=':', label=SETTINGS['display_name'][control])
    if not plot_solve_times:
        ax.set_xlabel(r'$N_{models}$')
    ax.set_ylabel(r'RMSE [mm]')
    # ax.set_title('RMSE vs. number of models')
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.grid(visible=True, which='both', axis='y')
    ax.set_xlim(-5, 105)
    ax.legend(loc="upper right")
    
    if plot_solve_times:
        for i, control in enumerate(z.keys()):
            solve_times[control] = 1000 * np.array(solve_times[control]).reshape(11, -1)
            print(control, solve_times[control].shape)
            print(len(n_models[control]))
            # ax2.boxplot(solve_times[control].T, positions=n_models[control], widths=3, vert=True, showfliers=False, showmeans=False, whiskerprops=dict(linestyle=''), capprops=dict(linestyle=''),
            #     medianprops=dict(linestyle='-', color='k'), patch_artist=True, boxprops=dict(facecolor=SETTINGS['color'][f"ssmr_adiabatic_{control}"]))
            ax2.plot(n_models[control], np.nanpercentile(solve_times[control], q=10, axis=-1),
                    color=SETTINGS['color'][f"ssmr_adiabatic_{control}"], ls='-', markeredgecolor='k', marker=markers[i], label=SETTINGS['display_name'][f"ssmr_adiabatic_{control}"])
            ax2.set_xlabel(r'$N_{models}$')
            ax2.set_ylabel(r'Solve time [ms]')
            ax2.set_xlim(-5, 105)
            # ax2.set_ylim(7, 32)
            ax2.grid(visible=True, which='both', axis='y')
            # ax2.yaxis.set_label_position("right")
            # ax2.yaxis.tick_right()
            ax2.legend(loc="upper left")
            ax2.set_xticks(n_models[control])
    plt.savefig(join("/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_closed-loop_analysis/10-sims-per-n-models", f"{TARGET}_rmse_and_solve_times_vs_n_models.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()


def x_vs_y_bundle():
    # import all data
    sims_per_model = 100
    N_models = 40
    z = {}
    solve_times = {}
    use_models = {}
    for control in ["modified_idw", "qp"]: # ["idw", "nn", "qp"]: # "ct", 
        result_dir = join("/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_closed-loop_analysis/100-sims-40-models", TARGET, f"adiabatic_ssm_{control}", f"{N_models:03d}")
        z[control] = []
        solve_times[control] = []
        use_models[control] = []
        t0 = 1
        for j in range(sims_per_model):
            print(control, j)
            with open(join(result_dir, f"use_models_{j}.pkl"), "rb") as f:
                models = pickle.load(f)
            use_models[control].append(models)
            if exists(join(result_dir, f"sim_{j}.pkl")):
                with open(join(result_dir, f"sim_{j}.pkl"), "rb") as f:
                    sim = pickle.load(f)
                idx = np.argwhere(sim['t'] > t0)[0][0]
                z_j = sim['z'][idx:, 3:]
                z_j[:, 2] *= -1
                solve_times_j = sim['info']['solve_times']
            elif exists(join(result_dir, f"sim_{j}_failed.pkl")):
                print("simulation failed: ", j)
                z_j = np.full((1001, 3), np.nan)
                solve_times_j = np.full(251, np.nan)
            else:
                raise RuntimeError(f"simulation not found: {dir}, {j}")
            z[control].append(z_j)
            solve_times[control].append(solve_times_j)

    fig, axs = plt.subplots(1, len(z.keys()), figsize=(len(z.keys()) * 6, 6), facecolor='w', edgecolor='k')
    if len(z.keys()) == 1:
        axs = [axs]

    for i, control in enumerate(z.keys()):
        z_centered = np.array(z[control]) - Z_EQ
        print(z_centered.shape)
        if TARGET in ["pac-man", "circle"]:
            error = z_centered[:, :, :3] - z_target[:, :3]
        else:
            error = z_centered[:, :, :2] - z_target[:, :2]
        rmse = np.sqrt(np.mean(np.linalg.norm(error, axis=-1)**2, axis=-1))
        best_traj = np.argmin(rmse, axis=-1)
        print("Best trajectory:", best_traj)
        print("Best RMSE:", np.min(rmse, axis=-1))
        print("Worst trajectory:", np.argmax(rmse, axis=-1))
        print("Worst RMSE:", np.max(rmse, axis=-1))
        axs[i].plot(z_centered[:, :, 0].T, z_centered[:, :, 1].T,
                color="#919191", # "#ba7f7f", # 
                label=control.upper(),
                linewidth=.5,
                ls='-')
        axs[i].plot(z_centered[best_traj, :, 0].T, z_centered[best_traj, :, 1].T,
                color=SETTINGS['color'][f"ssmr_adiabatic_{control}"],
                label=control.upper(),
                linewidth=3,
                ls='-') # marker='o', markevery=20,
        axs[i].plot(z_target[:, 0], z_target[:, 1],
            color=SETTINGS['color']['target'], alpha=0.9,
            ls=SETTINGS['linestyle'].get('target', TRAJ_LINESTYLE),
            lw=SETTINGS['linewidth'].get('target', TRAJ_LINEWIDTH),
            label='Target', zorder=3)

        axs[i].set_xlabel(r'$x_{ee}$ [mm]')
        axs[i].set_ylabel(r'$y_{ee}$ [mm]')
        # Remove top and right border
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].get_xaxis().tick_bottom()
        axs[i].get_yaxis().tick_left()
        axs[i].set_aspect('equal', 'box')
        axs[i].set_xlim(np.min(z_target[:, 0]) - PAD_LIMS, np.max(z_target[:, 0]) + PAD_LIMS)
        axs[i].set_ylim(np.min(z_target[:, 1]) - PAD_LIMS, np.max(z_target[:, 1]) + PAD_LIMS)
        axs[i].tick_params(axis='both')

    plt.savefig(join("/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_closed-loop_analysis/100-sims-40-models", f"{TARGET}_x_vs_y_bundle.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if SHOW_PLOTS:
        plt.show()


def model_contribution_to_rmse(z, use_models, save_dir="", show=True):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='w', edgecolor='k')
    n_models = 100

    for i, control in enumerate(z.keys()):

        model_rmse_contributions = np.zeros(n_models)

        z_centered = np.array(z[control]) - Z_EQ
        # pac-man
        error = z_centered[:, :, :, :3] - z_target[:, :3]
        # figure-8
        # error = z_centered[:, :, :, :2] - z_target[:, :2]
        rmse = np.sqrt(np.mean(np.linalg.norm(error, axis=-1)**2, axis=-1)).flatten()

        for j in range(n_models):
            occurence = 0
            for k in range(len(use_models[control])):
                if j in use_models[control][k]:
                    model_rmse_contributions[j] += rmse[k]
                    occurence += 1
            model_rmse_contributions[j] /= occurence

        ax.bar(np.arange(n_models), model_rmse_contributions - np.min(model_rmse_contributions), label=control.upper())

        print("USE_MODELS =", np.argpartition(model_rmse_contributions, 40)[:40].tolist())
    
    ax.set_xlabel(r'Model index')
    ax.set_ylabel(r'RMSE contribution')
    ax.set_title('Model contribution to RMSE')
    ax.legend()

    save_dir = ""
    if save_dir:
        plt.savefig(join(save_dir, f"{TARGET}_model_contribution_to_rmse.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    if show:
        plt.show()


if __name__ == "__main__":
    # rmse_vs_n_models()
    # x_vs_y_bundle()
    # traj_inputs_vs_t()
    # traj_x_vs_y()
    rmse_calculations(plot_solve_times=False)
    if TARGET == "figure8":
        traj_xy_vs_t()
    elif TARGET in ["circle", "pacman"]:
        traj_xyz_vs_t()
    # traj_3D()