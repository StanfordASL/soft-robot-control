from os.path import dirname, abspath, join, exists
import os
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
# import pdb
import yaml
# from sofacontrol.utils import load_data, set_axes_equal
import pickle
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatter

from sofacontrol.measurement_models import linearModel
from sofacontrol.utils import qv2x, load_data, CircleObstacle, load_full_equilibrium, add_decimal
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

path = dirname(abspath(__file__))
np.set_printoptions(linewidth=300)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'FreeSerif'})
plt.rcParams.update({'mathtext.fontset': 'cm'})

FONTSCALE = 1.1

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
    with open(join(path, SETTINGS['robot'], SETTINGS['traj_dir'], f'{control}_sim.pkl'), 'rb') as f:
        control_data = pickle.load(f)
    idx = np.argwhere(control_data['t'] >= t0)[0][0]
    SIM_DATA[control]['t'] = control_data['t'][idx:] - control_data['t'][idx]
    SIM_DATA[control]['z'] = control_data['z'][idx:, 3:]
    SIM_DATA[control]['u'] = control_data['u'][idx:, :]
    # SIM_DATA[control]['info']['solve_times'] = control_data['info']['solve_times']
    # SIM_DATA[control]['info']['real_time_limit'] = control_data['info']['rollout_time']

print("=== SOFA equilibrium point ===")
# Load equilibrium point
x_eq = load_full_equilibrium(join(path, SETTINGS['robot']))
print(x_eq.shape)

outputModel = linearModel([model.TIP_NODE], model.N_NODES, vel=False)
Z_EQ = outputModel.evaluate(x_eq, qv=False) #+ np.array([1.4, 0.0, 0.0])
if SETTINGS['robot'] == "trunk":
    Z_EQ[2] *= -1
print(Z_EQ)

# Load reference/target trajectory as defined in plotting_settings.py
TARGET = SETTINGS['select_target']
target_settings = SETTINGS['define_targets'][TARGET]
taskFile = join(path, SETTINGS['robot'], 'control_tasks', TARGET + '.pkl')
target = load_data(taskFile) # Note: target is centered, so we need to center the robot trajectory

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
    if target['X'] is not None:
        for iObs in range(len(target['X'].center)):

            circle = patches.Circle((target['X'].center[iObs][0], target['X'].center[iObs][1]), target['X'].diameter[iObs]/2, edgecolor='red', facecolor='none')
            # Add the circle to the axes
            ax.add_patch(circle)


    f = interp1d(target['t'], target['z'], axis=0)

    for control in CONTROLS:
        zf_target = f(SIM_DATA[control]['t'][:-2])

        # Don't center coordinates if koopman
        if control == "koopman":
            z_centered = SIM_DATA[control]['z']
        else:
            z_centered = SIM_DATA[control]['z'] - Z_EQ

        ax.plot(z_centered[:-2, 0], z_centered[:-2, 1],
                color=SETTINGS['color'][control],
                label=SETTINGS['display_name'][control],
                linewidth=SETTINGS['linewidth'][control],
                ls=SETTINGS['linestyle'][control], markevery=20,
                alpha=1.)
    ax.plot(zf_target[:, 0], zf_target[:, 1], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)

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

    plt.savefig(join(SAVE_DIR, f"{TARGET}_x_vs_y.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if SHOW_PLOTS:
        plt.show()

def traj_3D():

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    
    f = interp1d(target['t'], target['z'], axis=0)

    for control in CONTROLS:
        zf_target = f(SIM_DATA[control]['t'][:-2])
        
        # Don't center coordinates if koopman
        if control == "koopman":
            z_centered = SIM_DATA[control]['z'] - Z_EQ
        else:
            z_centered = SIM_DATA[control]['z'] - Z_EQ
        ax.plot(z_centered[:-2, 0], z_centered[:-2, 1], z_centered[:-2, 2],
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
    ax.grid(False)
    ax.view_init(5, -90)

    ax.tick_params(axis='both')

    plt.savefig(join(SAVE_DIR, f"{TARGET}_3D.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if SHOW_PLOTS:
        plt.show()

def traj_xy_vs_t():
    """Plot controlled trajectories as function of time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='w', edgecolor='k', sharex=True)

    f = interp1d(target['t'], target['z'], axis=0)

    target_length = len(target['t'])

    for ax, coord in [(ax1, 0), (ax2, 1)]:
        for control in CONTROLS: # + ['target']:
            # Truncate SIM_DATA[control]['t'] from the right to match the length of target['t']
            SIM_DATA[control]['t'] = SIM_DATA[control]['t'][:target_length]
            
            zf_target = f(SIM_DATA[control]['t'])

            # Don't center coordinates if koopman
            if control == "koopman":
                z_centered = SIM_DATA[control]['z']
            else:
                z_centered = SIM_DATA[control]['z'] - Z_EQ
            
            ax.plot(SIM_DATA[control]['t'], z_centered[:target_length, coord],
                        color=SETTINGS['color'][control],
                        label=SETTINGS['display_name'][control],
                        linewidth=SETTINGS['linewidth'][control],
                        ls=SETTINGS['linestyle'][control], markevery=20)
        ax.plot(SIM_DATA[control]['t'], zf_target[:, coord-3], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.8, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
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
            
            if control == "koopman":
                f = interp1d(target['t'], target['z'], axis=0)
            else:
                f = interp1d(target['t'], target['z'] - target['z'][0, :], axis=0)

            zf_target = f(SIM_DATA[control]['t'][:-2])

            # Don't center coordinates if koopman
            if control == "koopman":
                z_centered = SIM_DATA[control]['z']
            else:
                z_centered = SIM_DATA[control]['z'] - Z_EQ

            ax.plot(SIM_DATA[control]['t'][:-2], z_centered[:-2, coord],
                        color=SETTINGS['color'][control],
                        label=SETTINGS['display_name'][control],
                        linewidth=SETTINGS['linewidth'][control],
                        ls=SETTINGS['linestyle'][control], markevery=20)
        print("curr coord: ", coord)
        ax.plot(SIM_DATA[control]['t'][:-2], zf_target[:, coord], color=SETTINGS['color']['target'], ls=SETTINGS['linestyle']['target'], alpha=0.8, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
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
    
    err = {}
    rmse = {}
    solve_times = {}

    f = interp1d(target['t'], target['z'] + np.array([0., 0., Z_EQ[2]]), axis=0)

    for control in CONTROLS:
        zf_target = f(SIM_DATA[control]['t'][:-2])
        
        # Don't center coordinates if koopman
        if control == "koopman":
            z_centered = SIM_DATA[control]['z']
            z_centered[:, 2] += Z_EQ[2]
        else:
            z_centered = SIM_DATA[control]['z'] - Z_EQ

        # if control == "ssmr_origin":
        #     z_centered = z_centered[:-1, :]
        if (TARGET == "circle" and SETTINGS['robot'] == "hardware") or (TARGET == "custom" and SETTINGS['robot'] == "hardware")\
                or (TARGET == "pacman" and SETTINGS['robot'] == "trunk"):
            # errors are to be measured in 3D
            err[control] = (z_centered[:-2, :] - zf_target)
        else:
            # errors are to be measured in 2D
            err[control] = (z_centered[:-2, :2] - zf_target[:, :2])
        rmse[control] = np.sqrt(np.mean(np.linalg.norm(err[control], axis=1)**2, axis=0))
        # solve_times[control] = 1000 * np.array(SIM_DATA[control]['info']['solve_times'])

        print(f"========= {SETTINGS['display_name'][control]} =========")
        print(f"RMSE: {rmse[control]:.3f} mm")
        # print(f"Solve time: Min: {np.min(solve_times[control]):.3f} ms, Mean: {np.mean(solve_times[control]):.3f} ms, Max: {np.max(solve_times[control]):.3f} ms")

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

import numpy as np
import matplotlib.pyplot as plt

def violinplot(samples, vmax=None, legend_label=None, ax=None, show=True, color='blue'):
    samples = samples[~np.isnan(samples)].reshape(1, -1)  # Reshape samples to 2D array as expected by violinplot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 2))
    
    # Create the violin plot
    vp = ax.violinplot(samples.T, vert=True, showmeans=False, showmedians=True, showextrema=True, widths=0.5)
    
    # Customize colors
    for pc in vp['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.6)
    if 'cmeans' in vp:
        vp['cmeans'].set_color(color)
    if 'cmedians' in vp:
        vp['cmedians'].set_color('k')  # Median color
    if 'cmins' in vp:
        vp['cmins'].set_edgecolor(color)
    if 'cmaxes' in vp:
        vp['cmaxes'].set_edgecolor(color)
    if 'cbars' in vp:
        vp['cbars'].set_edgecolor(color)
    
    # Setting the legend label and x-axis limit if provided
    if legend_label is not None:
        patch = mpatches.Patch(color=color, label=legend_label)
        ax.legend(handles=[patch])
    
    if vmax is not None:
        ax.set_ylim(0, vmax)
    
    # Remove x-axis labels and ticks for clarity
    ax.xaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    
    # Show the plot or return the axis object
    if show:
        plt.show()
    else:
        return ax

def violation_calculations():
    """Compute, display and plot fraction of constraint violations for all controllers"""
    viol = {}
    max_viol = {}

    f = interp1d(target['t'], target['z'], axis=0)
    constraint = target['X']

    for control in CONTROLS:        
        # Don't center coordinates if koopman. Only care about the first two coordinates since planar constraints
        if control == "koopman":
            z_centered = SIM_DATA[control]['z'][:-2, :2]
        else:
            z_centered = SIM_DATA[control]['z'][:-2, :2] - Z_EQ[:2]
        
        viol_bool = [constraint.get_constraint_violation(x=None, z=z) for z in z_centered]
        viol_idxs = [idx for idx, val in enumerate(viol_bool) if val]

        # Consolidate values of interest
        viol[control] = len(viol_idxs) / len(viol_bool)
        max_viol[control] = max([constraint.get_constraint_violation(x=None, z=z) for z in z_centered])

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    axs[0].bar(xlabels, [viol[control] for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    axs[0].set_ylabel(r'Violation Ratio [%]')
    axs[0].set_title('Constraint Violation Ratio')
    axs[1].bar(xlabels, [max_viol[control] for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    axs[1].set_ylabel(r'Max Violation [mm]')
    axs[1].set_title('Maximum Constraint Violation')
    plt.savefig(join(SAVE_DIR, f"{TARGET}_ratio_violations.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if SHOW_PLOTS:
        plt.show()

def get_shade(color, factor):
    """Generate a lighter shade of the given color."""
    return tuple([(1 - factor) * component + factor for component in color])

def plot_bar_chart_for_multiple_dts(ax, data, rmse_threshold=280., set_threshold=False, model_comparison=False):
    # Assuming data is a dictionary of dictionaries in the format: data[control][dt]

    if model_comparison:
        CONTROLS = ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel", "ssmr_linear"]
    else:
        CONTROLS = ["ssmr_singleDelay", "ssmr_linear", "koopman", "DMD", "tpwl"]

    if SETTINGS['robot'] == "trunk":
        dts = sorted(data[CONTROLS[0]].keys()) # Assuming all controls have the same dts
    else:
        dts = [0.02]
    
    # dts = sorted(data[CONTROLS[0]].keys())

    num_dts = len(dts)
    width = 0.89 / num_dts

    if model_comparison:
        legend_name = {
        "ssmr_singleDelay": "SSMR\n(1 delay)",
        "ssmr_delays": "SSMR\n(4 delays)",
        "ssmr_posvel": "SSMR\n(pos-vel)",
        "ssmr_linear": "SSSR\n(1 delay)"
        }
    else:
        legend_name = SETTINGS['display_name_trunk'] if SETTINGS['robot'] == "trunk" else SETTINGS['display_name_hardware']

    # Choose an arbitrary color for the legend
    arbitrary_color = (0.2, 0.4, 0.6)  # This can be any color that shows shades well

    # Group bars by control
    dt_positions = [i + j * width for i in range(len(CONTROLS)) for j in range(num_dts)]
    dt_legend_handles = [mpatches.Patch(color=get_shade(arbitrary_color, idx * 0.2), label=r'$\Delta t={}$'.format(dt))
                     for idx, dt in enumerate(sorted(data[CONTROLS[0]].keys()))]
    
    for idx, dt in enumerate(dts):
        rmse_vals = [data[control][dt] for control in CONTROLS]
        dt_colors = [get_shade(matplotlib.colors.to_rgb(SETTINGS['color'][control]), (idx*0.2)) for val, control in zip(rmse_vals, CONTROLS)]
        hatches = ['' if val > rmse_threshold else '' for val in rmse_vals]
        edgecolors = [dt_colors[i] for i, hatch in enumerate(hatches)]
        positions = dt_positions[idx::num_dts]
        bars = ax.bar(positions, rmse_vals, width, color=dt_colors, label=r'$\Delta t={}$'.format(dt), edgecolor=edgecolors, zorder=2)

        # Create a shaded color for this dt
        shaded_color = get_shade(arbitrary_color, idx * 0.2)

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    
    # Configure primary and secondary x-axis labels
    # if len(dts) > 1:
    #     control_ticks = [0.2, 1.2, 2.2, 3.25, 4.3, 5.3, 6.3]
    # else:
    #     control_ticks = [0.02, 1.02, 2.02, 3.02, 4.1, 5.1, 6.1]

    if len(dts) > 1:
        control_ticks = [0.2, 1.2, 2.2, 3.25, 4.3] if not model_comparison else [0.1, 1.1, 2.1, 3.15]
    else:
        control_ticks = [0.02, 1.02, 2.02, 3.02, 4.1] if not model_comparison else [0.01, 1.01, 2.01, 3.01]

    ax.set_xticks(control_ticks)
    ax.set_xticklabels([legend_name[control] for control in CONTROLS], position=(0, 0.08), fontsize=10.)
    
    ax.set_xticks(dt_positions, minor=True)
    # ax.set_xticklabels([f'{dt}' for dt in dts] * len(CONTROLS), minor=True, fontsize=8., rotation=30)
    ax.yaxis.grid(True, color='gray', linewidth=0.5, zorder=1)

    # Adjust x-axis limits to reduce white space
    left_limit = min(dt_positions) - width
    right_limit = max(dt_positions) + width
    ax.set_xlim(left_limit, right_limit)

    # Remove the major tick lines
    ax.tick_params(axis='x', length=0, pad=20)
    sorted_rmse_vals = np.sort(rmse_vals)
    max_rmse = sorted_rmse_vals[-1]
    second_max_rmse = sorted_rmse_vals[-2]
    if set_threshold:
        threshold_val = rmse_threshold
        ax.set_ylim(0, threshold_val)
    else:
        rmse_threshold = 1.2*second_max_rmse if max_rmse > 2 * second_max_rmse else 1.2 * max_rmse
        ax.set_ylim(0, rmse_threshold)
    
    # Add metric value at the top of each bar
    for bar, rmse_val in zip(bars, rmse_vals):
        
        if np.isclose(round(rmse_val), 0.):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, 'Baseline', ha='center', va='bottom', fontsize=10, weight='bold')
        # Add an arrow pointing up above rmse that exceed threshold
        elif (rmse_val > threshold_val) and set_threshold:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.85*threshold_val, f'↑{round(rmse_val):,}%', 
                    ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.7), weight='bold')
            if SETTINGS['robot'] == "hardware":
                arrow_length = 55.0
            else:
                arrow_length = 150.0
            arrow = patches.FancyArrow(bar.get_x() + bar.get_width() / 2, ax.get_ylim()[1], 0, arrow_length, 
                                       width=0.5*bar.get_width(), head_width=0.8*bar.get_width(), 
                                       head_length=arrow_length / 1.8, length_includes_head=True, color=bar.get_facecolor(), zorder=3)
            ax.add_patch(arrow)
            arrow.set_clip_on(False)

            dots_location = ax.get_ylim()[1] + 0.5*arrow_length
            # Vertical dots
            increment = 0.4 * arrow_length
            dot_positions = [dots_location, dots_location + 0.5*increment, dots_location + 1.0*increment]
            for dot_y in dot_positions:
                ax.text(bar.get_x() + bar.get_width() / 2, dot_y, '.', ha='center', va='center', fontsize=20, color='white')
            
            # Horizontal dots
            # ax.text(bar.get_x() + bar.get_width() / 2, ax.get_ylim()[1] + 0.5*arrow_length, '...', ha='center', va='center', fontsize=20, color='white')
        elif rmse_val > 2 * second_max_rmse:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.85*rmse_threshold, f'↑{round(rmse_val)}%', ha='center', va='bottom', fontsize=10, weight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'↑{round(rmse_val)}%', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Add the legend for 'dt' values
    if len(dts) > 1:
        ax.legend(handles=dt_legend_handles, loc='best', fontsize=6)


def plotTrunkResults(dirname=None, dt_string=None, model_comparison=False, metric="rmse", control_normalizer="ssmr_singleDelay", 
                     rmse_threshold=800.0, set_threshold=True):
        
    # Define a default dictionary to store the data
    def nested_dict():
        return defaultdict(nested_dict)
    
    CONTROLS = ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel", "koopman", "ssmr_linear", "DMD", "tpwl"]
    if not model_comparison:
        controlTasks = ["ASL", "pacman"] # ["ASL", "pacman", "stanford"]
        SUBPLOT_MAPPING = {
        (0, 0): ["ssmr_singleDelay"], # ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel"]
        (0, 1): ["koopman"],
        (1, 0): ["ssmr_linear", "DMD"],
        (1, 1): ["tpwl"] # Add TPWL here
        }
        legend_name = SETTINGS['legend_name']
    else:
        controlTasks = ["ASL", "pacman", "stanford"]
        SUBPLOT_MAPPING = {
        (0, 0): ["ssmr_singleDelay"],
        (0, 1): ["ssmr_delays"],
        (1, 0): ["ssmr_posvel"],
        (1, 1): ["ssmr_linear"] # Add TPWL here
        }
        legend_name = {
        "ssmr_singleDelay": "SSMR (single delay)",
        "ssmr_delays": "SSMR (4 delays)",
        "ssmr_posvel": "SSMR (position-velocity)",
        "ssmr_linear": "SSSR (single delay)"
    }

    titles = [
        ["ASL Trajectory", "Pacman Trajectory", "Stanford Trajectory"], 
        ["", "", ""],
        ["", "", ""]]
    
    simData = nested_dict()
    rmse = nested_dict()
    targetTrajData = nested_dict()
    z_centeredData = nested_dict()

    metric_legend = {
        "rmse": r"Relative RMSE [%]",
        "ITAE": r"Relative ITAE [%]",
        "IAE": r"Relative IAE [%]",
        "ISE": r"Relative ISE [%]"
    }

    label_counter = 0
    label_list = [chr(i) for i in range(ord('a'), ord('z')+1)]

    # Go through each possible control task
    for task in controlTasks:
        if dirname is not None:
            simTaskFolder = join(path, SETTINGS['robot'], dirname, task)
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
        else:
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
            simTaskFolder = join(path, SETTINGS['robot'], task)
        
        z_target = load_data(taskFile)
        f_target = interp1d(z_target['t'], z_target['z'], axis=0)

        dt_folders = [dt_string] if dt_string is not None else os.listdir(simTaskFolder)

        # Iterate through each possible dt
        for dtFolder in dt_folders:
            # Get the dt
            dt = add_decimal(dtFolder)

            # Normalize with respect to ssmr_singleDelay
            normalizer_file_path = join(simTaskFolder, dtFolder, f"{control_normalizer}_sim.pkl")
            with open(normalizer_file_path, 'rb') as f:
                normalizer_data = pickle.load(f)
            idx_normalizer = np.argwhere(normalizer_data['t'] >= 1.0)[0][0]

            t_normalizer = normalizer_data['t'][idx_normalizer:] - normalizer_data['t'][idx_normalizer]
            zf_target_normalizer = f_target(t_normalizer[:-1])
            z_normalizer_centered = normalizer_data['z'][idx_normalizer:, 3:] - Z_EQ

            if task == "circle" or task == "star":
                error_normalize = (z_normalizer_centered[:-1, :] - zf_target_normalizer)
            else:
                error_normalize = (z_normalizer_centered[:-1, :2] - zf_target_normalizer[:, :2])

            if metric == "rmse":
                normalizer = np.sqrt(np.mean(np.linalg.norm(error_normalize, axis=1)**2, axis=0))
            elif metric == "ITAE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1) * normalizer_data['t'][:error_normalize.shape[0]])
            elif metric == "IAE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1), axis=0)
            elif metric == "ISE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1)**2, axis=0)
            
            # Iterate through each possible simulation
            for simCLfile in os.listdir(join(simTaskFolder, dtFolder)):
                sim_file_path = join(simTaskFolder, dtFolder, simCLfile)

                # Get each control simulation
                if simCLfile.split("_")[0] == "ssmr":
                    control = simCLfile.split("_")[0] + "_" + simCLfile.split("_")[1]
                else:
                    control = simCLfile.split("_")[0]

                # Load the simulation data
                with open(sim_file_path, 'rb') as f:
                    control_data = pickle.load(f)
                idx = np.argwhere(control_data['t'] >= 1.0)[0][0]
                simData[control][dt]['t'] = control_data['t'][idx:] - control_data['t'][idx]
                simData[control][dt]['z'] = control_data['z'][idx:, 3:]
                if SETTINGS['robot'] == "trunk":
                    simData[control][dt]['z'][:, 2] *= -1
                simData[control][dt]['u'] = control_data['u'][idx:, :]
                simData[control][dt]['info']['solve_times'] = control_data['info']['solve_times']
                simData[control][dt]['info']['real_time_limit'] = control_data['info']['rollout_time']

                # Extract RMSE
                zf_target = f_target(simData[control][dt]['t'])

                z_centered = simData[control][dt]['z'] - Z_EQ
                error = (z_centered[:, :2] - zf_target[:, :2])

                if metric == "rmse":
                    rmse[task][control][dt] = (np.sqrt(np.mean(np.linalg.norm(error, axis=1)**2, axis=0)) / normalizer - 1.) * 100.
                elif metric == "ITAE":
                    rmse[task][control][dt] = (np.sum(np.linalg.norm(error, axis=1) * simData[control][dt]['t'][:error.shape[0]]) / normalizer - 1.)*100.
                elif metric == "IAE":
                    rmse[task][control][dt] = (np.sum(np.linalg.norm(error, axis=1), axis=0) / normalizer - 1.)*100.
                elif metric == "ISE":
                    rmse[task][control][dt] = (np.sum(np.linalg.norm(error, axis=1)**2, axis=0) / normalizer - 1.)*100.

                # Grab target and control trajectory data for plotting later
                targetTrajData[task][control][dt] = zf_target
                z_centeredData[task][control][dt] = z_centered

    # Create main figure and gridspec
    fig = plt.figure(figsize=(10, 5))
    if not model_comparison:
        gs = gridspec.GridSpec(len(controlTasks), len(controlTasks), figure=fig, height_ratios=[1.6, 1.], hspace=0.45)  # 3x3 grid with height ratios
    else:
        gs = gridspec.GridSpec(2, len(controlTasks), figure=fig, height_ratios=[1.6, 1.], hspace=0.45)
    
    handles, labels = [], []
    y_axis_limits = None
    top_row_axes = []  # List to store all axes of the top row

    for j, task in enumerate(controlTasks):  # Loop over columns
        DT_PLOT = 0.02

        # Top row: Each plot is further divided into 2x2 grid
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, j], hspace=0.4)
        for k in range(2):
            for l in range(2):
                ax = fig.add_subplot(gs_sub[k, l])
                ax.yaxis.set_major_locator(MaxNLocator(3))  # Set the maximum number of y-axis ticks to 3
                ax.xaxis.set_major_locator(MaxNLocator(3))  # Set the maximum number of x-axis ticks to 3
                top_row_axes.append(ax)  # Add the axis to our list
                current_controls = SUBPLOT_MAPPING[(k, l)]

                # Hide y-axis for plots that are not left-most
                if l > 0:
                    ax.tick_params(labelleft=False)

                for control in current_controls:
                    # Get desired trajectory and controlled trajectory
                    desired_target = targetTrajData[task][control][DT_PLOT]
                    controlled_traj = z_centeredData[task][control][DT_PLOT]

                    ax.plot(desired_target[:, 0], desired_target[:, 1], color=SETTINGS['color']['target'], 
                            ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)

                    line, = ax.plot(controlled_traj[:, 0], controlled_traj[:, 1],
                    color=SETTINGS['color'][control],
                    label=SETTINGS['display_name'][control],
                    linewidth=SETTINGS['linewidth'][control],
                    ls=SETTINGS['linestyle'][control], markevery=20,
                    alpha=SETTINGS['alpha'][control])

                    handles.append(line)
                    labels.append(legend_name[control])

                # If it's the top-left subplot, get its y-axis limits
                if y_axis_limits is None and j == 0 and k == 0 and l == 0:
                    y_axis_limits = ax.get_ylim()

        top_row_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        top_ycoord = top_row_bbox.y0  # Get the top y-coordinate of the middle row

        # Add the label to the top left corner of each outer subplot
        ax.text(-1.52, 2.4, f"({label_list[label_counter]})", transform=ax.transAxes, 
                fontsize=12, va='top', ha='left')
        label_counter += 1

        ax = fig.add_subplot(gs[1, j])
        
        ax.set_title(titles[1][j])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        plot_bar_chart_for_multiple_dts(ax, rmse[task], model_comparison=model_comparison, rmse_threshold=rmse_threshold, set_threshold=True)

        if j == 0:
            ax.set_ylabel(metric_legend[metric])
        else:
            ax.set_ylabel('')
        
        middle_row_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        middle_ycoord = middle_row_bbox.y1

        # Add the label to the top left corner of each outer subplot
        ax.text(-0.1, 1.1, f"({label_list[label_counter]})", transform=ax.transAxes, 
                fontsize=12, va='top', ha='left')
        label_counter += 1
    
    # Legend for the top row
    handle_label_dict = dict(zip(labels, handles))
    unique_labels = list(handle_label_dict.keys())
    unique_handles = [handle_label_dict[label] for label in unique_labels]

    # Place the legend
    offset = 0.2*(top_ycoord - middle_ycoord)
    fig.legend(unique_handles, unique_labels, loc='center', 
            ncol=len(unique_labels), bbox_to_anchor=(0.5, middle_ycoord + offset),
            bbox_transform=fig.transFigure, fontsize='7.2')

    
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, f"trunk_sim_results.{SETTINGS['file_format']}"), bbox_inches='tight', 
                    dpi=400, format=SETTINGS['file_format'], transparent=True)
    # plt.show()

def plotDiamondTrials():
    # Define a default dictionary to store the data
    def nested_dict():
        return defaultdict(nested_dict)
    
    simData = nested_dict()
    rmse = nested_dict()

    CONTROLS = ["ssmr_singleDelay", "ssmr_posvel", "ssmr_linear"]

    robotPath = join(path, SETTINGS['robot'])
    trial_folders = [taskFolder for taskFolder in os.listdir(robotPath) if "_trials" in taskFolder and os.path.isdir(join(robotPath, taskFolder))]

    for taskFolder in trial_folders:
        task = taskFolder.split("_")[0]
        taskPath = join(robotPath, taskFolder)

        for trial in os.listdir(taskPath):
            trialPath = join(taskPath, trial)
            trialInt = int(trial)
            # Loop through all trials that do not include the desired trajectory
            # for simCLfile in (f for f in os.listdir(trialPath) if "sim.pkl" in f and os.path.isfile(join(trialPath, f))):
            for control in CONTROLS:
                
                # Process desired trajectory
                controlTaskFile = join(trialPath, task + '.pkl')
                z_target = load_data(controlTaskFile) 
                f_target = interp1d(z_target['t'], z_target['z'], axis=0)

                # Process simulation files
                simCLfile = control + "_sim.pkl"
                sim_file_path = join(trialPath, simCLfile)

                # Get each control simulation
                # if simCLfile.split("_")[0] == "ssmr":
                #     control = simCLfile.split("_")[0] + "_" + simCLfile.split("_")[1]
                # else:
                #     control = simCLfile.split("_")[0]

                # Load the simulation data
                with open(sim_file_path, 'rb') as f:
                    control_data = pickle.load(f)
                idx = np.argwhere(control_data['t'] >= 1.0)[0][0]
                simData[task][control][trialInt]['t'] = control_data['t'][idx:] - control_data['t'][idx]
                simData[task][control][trialInt]['z'] = control_data['z'][idx:, 3:]
                simData[task][control][trialInt]['u'] = control_data['u'][idx:, :]
                simData[task][control][trialInt]['info']['solve_times'] = control_data['info']['solve_times']
                simData[task][control][trialInt]['info']['real_time_limit'] = control_data['info']['rollout_time']

                # Extract RMSE
                zf_target = f_target(simData[task][control][trialInt]['t'][:-1])
                z_centered = simData[task][control][trialInt]['z'] - Z_EQ
                if task == "circle" or task == "star":
                    error = (z_centered[:-1, :] - zf_target)
                else:
                    error = (z_centered[:-1, :2] - zf_target[:, :2])
                rmse[task][control][trialInt] = np.sqrt(np.mean(np.linalg.norm(error, axis=1)**2, axis=0))
    
    tasks = list(rmse.keys())
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(15, 5))
    fig.suptitle('RMSE for each control and task')
    
    if len(tasks) == 1:  # In case there's only one task, make axes iterable
        axes = [axes]
    
    for i, task in enumerate(tasks):
        ax = axes[i]
        ax.set_title(f'Task: {task}')
        
        for control in CONTROLS:
            rmse_values = []
            trialInts = sorted(list(rmse[task][control].keys()))
            
            for trialInt in trialInts:
                rmse_values.append(rmse[task][control][trialInt])
                
            ax.plot(trialInts, rmse_values, SETTINGS['altlinestyle'][control], color=SETTINGS['color'][control],
                    label=SETTINGS['display_name'][control],
                    linewidth=SETTINGS['linewidth'][control],
                    alpha=SETTINGS['alpha'][control])
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('RMSE')
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_solve_times(dirname=None):

    # Define a default dictionary to store the data
    def nested_dict():
        return defaultdict(nested_dict)

    # CONTROLS = ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel", "koopman", "ssmr_linear", "DMD", "tpwl"]
    CONTROLS = ["ssmr_singleDelay", "ssmr_linear", "koopman", "DMD", "tpwl"]
    SUBPLOT_MAPPING = {
    (0, 0): ["ssmr_singleDelay"], # ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel"]
    (0, 1): ["koopman"],
    (1, 0): ["ssmr_linear", "DMD"],
    (1, 1): ["tpwl"] # Add TPWL here
    }
    if SETTINGS['robot'] == "trunk":
        controlTasks = ["ASL", "pacman", "stanford"] # ["ASL", "pacman", "stanford"]
        singleLine_display_name = {
            "ssmr_singleDelay": "SSMR\n(6D)",
            "ssmr_delays": "SSMR\n(6D)",
            "ssmr_posvel": "SSMR\n(6D)",
            "koopman": "EDMD\n(120D)",
            "ssmr_linear": "SSSR\n(6D)",
            "DMD": "DMD\n(15D)",
            "tpwl": "TPWL\n(28D)"
        }
    else:
        controlTasks = ["figure8_fast", "circle", "star"] # ["figure8_fast", "circle", "star"]
        singleLine_display_name = {
            "ssmr_singleDelay": "SSMR\n(6D)",
            "ssmr_delays": "SSMR\n(6D)",
            "ssmr_posvel": "SSMR\n(6D)",
            "koopman": "EDMD\n(66D)",
            "ssmr_linear": "SSSR\n(6D)",
            "DMD": "DMD\n(11D)",
            "tpwl": "TPWL\n(42D)"
        }
        
    simData = nested_dict()
    rmse = nested_dict()
    targetTrajData = nested_dict()
    z_centeredData = nested_dict()
    solve_times = {}

    fig, axs = plt.subplots(1, 1, figsize=(5, 3))

    # Go through each possible control task
    for task in controlTasks:
        if dirname is not None:
            simTaskFolder = join(path, SETTINGS['robot'], dirname, task)
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
        else:
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
            simTaskFolder = join(path, SETTINGS['robot'], task)
        
        z_target = load_data(taskFile)
        f_target = interp1d(z_target['t'], z_target['z'], axis=0)
        

        # Iterate through each possible dt
        for dtFolder in os.listdir(simTaskFolder):
            # Get the dt
            dt = add_decimal(dtFolder)
            
            # Iterate through each possible simulation
            for simCLfile in os.listdir(join(simTaskFolder, dtFolder)):
                sim_file_path = join(simTaskFolder, dtFolder, simCLfile)

                # Get each control simulation
                if simCLfile.split("_")[0] == "ssmr":
                    control = simCLfile.split("_")[0] + "_" + simCLfile.split("_")[1]
                else:
                    control = simCLfile.split("_")[0]

                # Load the simulation data
                with open(sim_file_path, 'rb') as f:
                    control_data = pickle.load(f)
                idx = np.argwhere(control_data['t'] >= 1.0)[0][0]
                simData[control][dt]['t'] = control_data['t'][idx:] - control_data['t'][idx]
                simData[control][dt]['z'] = control_data['z'][idx:, 3:]
                if SETTINGS['robot'] == "trunk":
                    simData[control][dt]['z'][:, 2] *= -1
                simData[control][dt]['u'] = control_data['u'][idx:, :]
                simData[control][dt]['info']['solve_times'] = control_data['info']['solve_times']
                simData[control][dt]['info']['real_time_limit'] = control_data['info']['rollout_time']

    first_subplot_created = False
    # Iterate through each control to create a boxplot for its solve times
    spacing = 0.001
    width = 0.1  # width of each violin plot
    positions = [i * (spacing + width) for i in range(len(CONTROLS))]

    for i, control in enumerate(CONTROLS):
        ax = axs
        ax.set_yscale('log')  # Set y-axis to log scale
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatter(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))

        all_solve_times = np.array([1000 * time for dt in simData[control] for time in simData[control][dt]['info']['solve_times']])

        # Creating violin plots at specific positions
        vp = ax.violinplot(all_solve_times.T, positions=[positions[i]], vert=True, showmeans=False, showmedians=False, showextrema=False, widths=width)
        
        color = SETTINGS['color'][control]
        for pc in vp['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('k')
            pc.set_alpha(0.7)
            pc.set_linewidth(1.5)  # Thicker edges for violin bodies
        if 'cmeans' in vp:
            vp['cmeans'].set_color(color)
        if 'cmedians' in vp:
            vp['cmedians'].set_color(color)  # Median color
        if 'cmins' in vp:
            vp['cmins'].set_edgecolor(color)
        if 'cmaxes' in vp:
            vp['cmaxes'].set_edgecolor(color)
        if 'cbars' in vp:
            vp['cbars'].set_edgecolor(color)
        
        # Calculate mean and annotate it
        mean_value = np.mean(all_solve_times)
        ax.annotate(f'{mean_value:.2f}', xy=(positions[i] + width/4, 1.45*mean_value), xytext=(-30,0), 
                    textcoords='offset points', ha='right', va='center', color='black', fontsize=10)
        
        # Draw a horizontal dashed line at the mean
        ax.hlines(mean_value, positions[i] - 0.5*width / 2, positions[i] + 0.5*width / 2, colors='black', linestyles='dashed', linewidth=1)

    ax.set_ylabel("Solve Times [ms]")
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.set_axisbelow(True)

    # Set x-ticks to be at the center of each violin plot
    ax.set_xticks(positions)
    ax.set_xticklabels([singleLine_display_name[control] for control in CONTROLS])  # Assuming you have names for each control

    plt.tight_layout()
    plt.title('Simulated Diamond MPC Solve Times')
    plt.savefig(join(SAVE_DIR, f"{SETTINGS['robot']}_solve_times.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    # plt.show()

def plotDiamondResults(dirname=None, dt_string=None, model_comparison=False, metric="rmse", control_normalizer="ssmr_linear",
                       rmse_threshold=280.0, set_threshold=True):

    def nested_dict():
        return defaultdict(nested_dict)
    
    SUBPLOT_MAPPING = {
        (0, 0): ["ssmr_singleDelay"],
        (0, 1): ["koopman"],
        (1, 0): ["ssmr_linear", "DMD"],
        (1, 1): ["tpwl"]
    }

    titles = [
        ["ASL Trajectory", "Pacman Trajectory", "Stanford Trajectory"], 
        ["", "", ""],
        ["", "", ""]]

    if not model_comparison:
        controlTasks = ["figure8", "figure8_fast"]
        SUBPLOT_MAPPING = {
            (0, 0): ["ssmr_singleDelay"],
            (0, 1): ["koopman"],
            (1, 0): ["ssmr_linear", "DMD"],
            (1, 1): ["tpwl"]
        }
        legend_name = SETTINGS['legend_name_hardware'] if SETTINGS['robot'] == "hardware" else SETTINGS['legend_name_trunk']
    else:
        controlTasks = ["figure8", "circle", "star"]
        SUBPLOT_MAPPING = {
            (0, 0): ["ssmr_singleDelay"],
            (0, 1): ["ssmr_delays"],
            (1, 0): ["ssmr_posvel"],
            (1, 1): ["ssmr_linear"]
        }
        legend_name = {
            "ssmr_singleDelay": "SSMR (1 delay)",
            "ssmr_delays": "SSMR (4 delays)",
            "ssmr_posvel": "SSMR (position-velocity)",
            "ssmr_linear": "SSSR (1 delay)"
        }

    metric_legend = {
        "rmse": r"Relative RMSE [%]",
        "ITAE": r"Relative ITAE [%]",
        "IAE": r"Relative IAE [%]",
        "ISE": r"Relative ISE [%]"
    }
    
    simData = nested_dict()
    rmse = nested_dict()
    targetTrajData = nested_dict()
    z_centeredData = nested_dict()

    label_counter = 0
    label_list = [chr(i) for i in range(ord('a'), ord('z')+1)]

    for task in controlTasks:
        if dirname is not None:
            simTaskFolder = join(path, SETTINGS['robot'], dirname, task)
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
        else:
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
            simTaskFolder = join(path, SETTINGS['robot'], task)

        z_target = load_data(taskFile)
        f_target = interp1d(z_target['t'], z_target['z'], axis=0)

        dt_folders = [dt_string] if dt_string is not None else os.listdir(simTaskFolder)

        for dtFolder in dt_folders:
            dt = add_decimal(dtFolder)

            normalizer_file_path = join(simTaskFolder, dtFolder, f"{control_normalizer}_sim.pkl")
            with open(normalizer_file_path, 'rb') as f:
                normalizer_data = pickle.load(f)
            idx_normalizer = np.argwhere(normalizer_data['t'] >= 1.0)[0][0]

            t_normalizer = normalizer_data['t'][idx_normalizer:] - normalizer_data['t'][idx_normalizer]
            zf_target_normalizer = f_target(t_normalizer[:-1])
            z_normalizer_centered = normalizer_data['z'][idx_normalizer:, 3:] - Z_EQ

            if task == "circle" or task == "star":
                error_normalize = (z_normalizer_centered[:-1, :] - zf_target_normalizer)
            else:
                error_normalize = (z_normalizer_centered[:-1, :2] - zf_target_normalizer[:, :2])

            if metric == "rmse":
                normalizer = np.sqrt(np.mean(np.linalg.norm(error_normalize, axis=1)**2, axis=0))
            elif metric == "ITAE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1) * normalizer_data['t'][:error_normalize.shape[0]])
            elif metric == "IAE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1), axis=0)
            elif metric == "ISE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1)**2, axis=0)
            
            for simCLfile in os.listdir(join(simTaskFolder, dtFolder)):
                sim_file_path = join(simTaskFolder, dtFolder, simCLfile)

                if simCLfile.split("_")[0] == "ssmr":
                    control = simCLfile.split("_")[0] + "_" + simCLfile.split("_")[1]
                else:
                    control = simCLfile.split("_")[0]

                with open(sim_file_path, 'rb') as f:
                    control_data = pickle.load(f)
                idx = np.argwhere(control_data['t'] >= 1.0)[0][0]
                simData[control][dt]['t'] = control_data['t'][idx:] - control_data['t'][idx]
                simData[control][dt]['z'] = control_data['z'][idx:, 3:]
                simData[control][dt]['u'] = control_data['u'][idx:, :]
                simData[control][dt]['info']['solve_times'] = control_data['info']['solve_times']
                simData[control][dt]['info']['real_time_limit'] = control_data['info']['rollout_time']

                zf_target = f_target(simData[control][dt]['t'][:-1])
                z_centered = simData[control][dt]['z'] - Z_EQ
                if task == "circle" or task == "star":
                    error = (z_centered[:-1, :] - zf_target)
                else:
                    error = (z_centered[:-1, :2] - zf_target[:, :2])
                
                if metric == "rmse":
                    rmse[task][control][dt] = (np.sqrt(np.mean(np.linalg.norm(error, axis=1)**2, axis=0)) / normalizer - 1.) * 100.
                elif metric == "ITAE":
                    rmse[task][control][dt] = (np.sum(np.linalg.norm(error, axis=1) * simData[control][dt]['t'][:error.shape[0]]) / normalizer - 1.)*100.
                elif metric == "IAE":
                    rmse[task][control][dt] = (np.sum(np.linalg.norm(error, axis=1), axis=0) / normalizer - 1.)*100.
                elif metric == "ISE":
                    rmse[task][control][dt] = (np.sum(np.linalg.norm(error, axis=1)**2, axis=0) / normalizer - 1.)*100.

                targetTrajData[task][control][dt] = zf_target
                z_centeredData[task][control][dt] = z_centered

    fig = plt.figure(figsize=(10, 5))
    if not model_comparison:
        gs = gridspec.GridSpec(len(controlTasks), len(controlTasks), figure=fig, height_ratios=[1.6, 1.], hspace=0.45)
    else:
        gs = gridspec.GridSpec(2, len(controlTasks), figure=fig, height_ratios=[1.6, 1.], hspace=0.45)
    
    handles, labels = [], []
    y_axis_limits = None
    top_row_axes = []

    for j, task in enumerate(controlTasks):
        DT_PLOT = 0.02

        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, j], wspace=0.25, hspace=0.3)
        for k in range(2):
            for l in range(2):
                ax = fig.add_subplot(gs_sub[k, l])
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_locator(MaxNLocator(3))
                top_row_axes.append(ax)

                current_controls = SUBPLOT_MAPPING[(k, l)]

                if l > 0:
                    ax.tick_params(labelleft=False)
                
                if task == "circle":
                    DT_PLOT = 0.05

                for control in current_controls:
                    desired_target = targetTrajData[task][control][DT_PLOT]
                    controlled_traj = z_centeredData[task][control][DT_PLOT]
                    
                    if task == "circle" or task == "star":
                        ax.plot(desired_target[:, 1], desired_target[:, 2], color=SETTINGS['color']['target'], 
                            ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
                        controlled_traj = controlled_traj[:, 1:]
                        if task == "circle":
                            ax.set_ylim(-1., 35)
                            ax.set_xlim(-25., 25)
                        elif task == "star":
                            ax.set_ylim(-1., 39.)
                            ax.set_xlim(-15., 12.)
                    else:
                        ax.plot(desired_target[:, 0], desired_target[:, 1], color=SETTINGS['color']['target'], 
                            ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
                        controlled_traj = controlled_traj[:, :2]
                        if task == "figure8_fast":
                            ax.set_ylim(-17., 17.)
                            ax.set_xlim(-17., 17.)

                    line, = ax.plot(controlled_traj[:, 0], controlled_traj[:, 1],
                    color=SETTINGS['color'][control],
                    label=SETTINGS['display_name'][control],
                    linewidth=SETTINGS['linewidth'][control],
                    ls=SETTINGS['linestyle'][control], markevery=20,
                    alpha=SETTINGS['alpha'][control])

                    handles.append(line)
                    labels.append(legend_name[control])

        top_row_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        top_ycoord = top_row_bbox.y0
        # Add the label to the top left corner of each outer subplot
        ax.text(-1.57, 2.4, f"({label_list[label_counter]})", transform=ax.transAxes, 
                fontsize=12, va='top', ha='left')
        label_counter += 1

        ax = fig.add_subplot(gs[1, j])
        
        ax.set_title(titles[1][j])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        plot_bar_chart_for_multiple_dts(ax, rmse[task], model_comparison=model_comparison, rmse_threshold=rmse_threshold, set_threshold=True)

        if j == 0:
            ax.set_ylabel(metric_legend[metric])
        else:
            ax.set_ylabel('')
        
        middle_row_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        middle_ycoord = middle_row_bbox.y1

        # Add the label to the top left corner of each outer subplot
        ax.text(-0.1, 1.1, f"({label_list[label_counter]})", transform=ax.transAxes, 
                fontsize=12, va='top', ha='left')
        label_counter += 1
        
    handle_label_dict = dict(zip(labels, handles))
    unique_labels = handle_label_dict.keys()
    unique_handles = [handle_label_dict[label] for label in unique_labels]

    offset = 0.1 * (top_ycoord - middle_ycoord)
    fig.legend(unique_handles, unique_labels, loc='center', 
            ncol=len(unique_labels), bbox_to_anchor=(0.5, middle_ycoord + offset),
            bbox_transform=fig.transFigure, fontsize='7.2')

    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, f"diamond_sim_results.{SETTINGS['file_format']}"), bbox_inches='tight', 
                    dpi=400, format=SETTINGS['file_format'], transparent=True)
    # plt.show()

def is_dominated(point, points):
    eps = [1.0, 1.0] # TODO: Currently epsilon-domination set to 1.0 (normal pareto-dominance)
    for pt in points:
        if pt[0] <= point[0] * eps[0] and pt[1] <= point[1] * eps[1] and (pt[0] < point[0] * eps[0] or pt[1] < point[1] * eps[1]):
            return True
    return False

def get_pareto_front(points):
    pareto_points = []
    for point in points:
        if not is_dominated(point, points):
            pareto_points.append(point)
    # Sort the points to plot them correctly
    pareto_points.sort(key=lambda x: x[0])  # Sort by solve time

    # Create stair-step points
    stair_step_points = []
    for i in range(len(pareto_points)):
        if i == 0:
            stair_step_points.append((pareto_points[i][0], 10**12))
        else:
            stair_step_points.append((pareto_points[i][0], pareto_points[i-1][1]))
        stair_step_points.append(pareto_points[i])

    # Append the desired points to the end of the stair_step_points list
    if pareto_points:
        stair_step_points.append((100, pareto_points[-1][1]))
    
    return zip(*stair_step_points)  # Unzip into separate lists for plotting

def pareto_plot_diamond(metric, models, dirname=None, dt_string=["002"]):

    # Define a default dictionary to store the data
    def nested_dict():
        return defaultdict(nested_dict)
    
    if SETTINGS['robot'] == "hardware":
        controlTasks = ["figure8", "figure8_fast", "circle"] # ["circle", "star"]
        task_legend = {
            "circle": "Fast Circle",
            "figure8_fast": "Fast Figure 8",
            "star": "Star",
            "figure8": "Slow Figure 8"
        }
        robot = "Simulated Diamond"
        legend_name = SETTINGS['legend_name_hardware']
    else:
        controlTasks = ["ASL", "pacman", "stanford"]
        task_legend = {
            "ASL": "ASL",
            "pacman": "Pacman",
            "stanford": "Stanford"
        }
        robot = "Simulated Trunk"
        legend_name = SETTINGS['legend_name_trunk']

    metric_legend = {
        "rmse": "RMSE [mm]",
        "ITAE": r"ITAE [m s$^2$]",
        "IAE": r"IAE [m s]",
        "ISE": r"ISE [m$^2$ s]"
    }

    marker_style = {
        controlTasks[0]: 'o',
        controlTasks[1]: 's',
        controlTasks[2]: '^'
    }

    linestyle_legend = {
        controlTasks[0]: '-.',
        controlTasks[1]: '--',
        controlTasks[2]: ':'
    }

    simData = nested_dict()
    rmse = nested_dict()
    solve_times = nested_dict()
    targetTrajData = nested_dict()
    z_centeredData = nested_dict()

    control_normalizer = "ssmr_singleDelay"

    plt.figure(figsize=(5, 3))
    added_labels = set()
    legend_elements = []

    all_points_all_tasks = []

    # Go through each possible control task
    for task in controlTasks:
        all_points_per_task = []

        legend_elements.append(Line2D([0], [0], marker=marker_style[task], color='w', label=task_legend[task],
                                  markerfacecolor='gray', markersize=10))  # Change 'black' to any appropriate color

        if dirname is not None:
            simTaskFolder = join(path, SETTINGS['robot'], dirname, task)
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
        else:
            taskFile = join(path, SETTINGS['robot'], dirname, 'control_tasks', task + '.pkl')
            simTaskFolder = join(path, SETTINGS['robot'], task)

        z_target = load_data(taskFile)
        f_target = interp1d(z_target['t'], z_target['z'], axis=0)

        dt_folders = dt_string if dt_string is not None else os.listdir(simTaskFolder)

        # Iterate through each possible dt
        for dtFolder in dt_folders:
            # Get the dt
            dt = add_decimal(dtFolder)

            # Normalize with respect to ssmr_singleDelay
            normalizer_file_path = join(simTaskFolder, dtFolder, f"{control_normalizer}_sim.pkl")
            with open(normalizer_file_path, 'rb') as f:
                normalizer_data = pickle.load(f)
            idx_normalizer = np.argwhere(normalizer_data['t'] >= 1.0)[0][0]

            t_normalizer = normalizer_data['t'][idx_normalizer:] - normalizer_data['t'][idx_normalizer]
            zf_target_normalizer = f_target(t_normalizer[:-1])
            z_normalizer_centered = normalizer_data['z'][idx_normalizer:, 3:] - Z_EQ

            if task == "circle" or task == "star":
                error_normalize = (z_normalizer_centered[:-1, :] - zf_target_normalizer)
            else:
                error_normalize = (z_normalizer_centered[:-1, :2] - zf_target_normalizer[:, :2])

            if metric == "rmse":
                normalizer = np.sqrt(np.mean(np.linalg.norm(error_normalize, axis=1)**2, axis=0))
            elif metric == "ITAE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1) * normalizer_data['t'][:error_normalize.shape[0]])
            elif metric == "IAE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1), axis=0)
            elif metric == "ISE":
                normalizer = np.sum(np.linalg.norm(error_normalize, axis=1)**2, axis=0)
            
            # Iterate through each possible simulation
            for simCLfile in os.listdir(join(simTaskFolder, dtFolder)):

                # Get each control simulation
                if simCLfile.split("_")[0] == "ssmr":
                    control = simCLfile.split("_")[0] + "_" + simCLfile.split("_")[1]
                else:
                    control = simCLfile.split("_")[0]
                
                # Check if the control is in the models we want to compare
                if control in models:
                    sim_file_path = join(simTaskFolder, dtFolder, simCLfile)
                else:
                    continue

                # Load the simulation data
                with open(sim_file_path, 'rb') as f:
                    control_data = pickle.load(f)
                idx = np.argwhere(control_data['t'] >= 1.0)[0][0]
                simData[control][dt]['t'] = control_data['t'][idx:] - control_data['t'][idx]
                simData[control][dt]['z'] = control_data['z'][idx:, 3:]
                simData[control][dt]['u'] = control_data['u'][idx:, :]
                simData[control][dt]['info']['solve_times'] = control_data['info']['solve_times']
                simData[control][dt]['info']['real_time_limit'] = control_data['info']['rollout_time']

                # Extract RMSE
                zf_target = f_target(simData[control][dt]['t'][:-1])
                z_centered = simData[control][dt]['z'] - Z_EQ
                if task == "circle" or task == "star":
                    error = (z_centered[:-1, :] - zf_target)
                else:
                    error = (z_centered[:-1, :2] - zf_target[:, :2])
                
                if metric == "rmse":
                    rmse[task][control][dt] = np.sqrt(np.mean(np.linalg.norm(error, axis=1)**2, axis=0))
                elif metric == "ITAE":
                    rmse[task][control][dt] = np.sum(np.linalg.norm(error, axis=1) * simData[control][dt]['t'][:error.shape[0]])
                elif metric == "IAE":
                    rmse[task][control][dt] = np.sum(np.linalg.norm(error, axis=1), axis=0)
                elif metric == "ISE":
                    rmse[task][control][dt] = np.sum(np.linalg.norm(error, axis=1)**2, axis=0)

                # Grab target and control trajectory data for plotting later
                targetTrajData[task][control][dt] = zf_target
                z_centeredData[task][control][dt] = z_centered
                solve_times[task][control][dt] = np.mean(simData[control][dt]['info']['solve_times']) / dt

                all_points_per_task.append((solve_times[task][control][dt], rmse[task][control][dt]))
                all_points_all_tasks.append((solve_times[task][control][dt], rmse[task][control][dt]))

                # Plotting within the loop
                # Only label the first occurrence
                label = SETTINGS['display_name'][control]
                if label not in added_labels:
                    plt.scatter(np.array(solve_times[task][control][dt]), rmse[task][control][dt], 
                            label=f"{legend_name[control]}", color=SETTINGS['color'][control], alpha=0.7, marker=marker_style[task])
                    added_labels.add(label)
                else:
                    plt.scatter(np.array(solve_times[task][control][dt]), rmse[task][control][dt], 
                            color=SETTINGS['color'][control], alpha=0.7, marker=marker_style[task])
            
        # Calculate and plot the Pareto front
        pareto_x, pareto_y = get_pareto_front(all_points_per_task)
        plt.plot(pareto_x, pareto_y, color='black', linestyle=linestyle_legend[task], label=f'Pareto Front for {task_legend[task]}', alpha=0.5)

    plt.title(f'Pareto Log-plot for {robot}', fontsize=10)
    # plt.axvline(x=1, ymin=-10**6, ymax=10**6, color='red', linewidth=2, label="Real-time Limit")  # Adjust the color and linewidth as needed

    plt.xlabel('Solve Time to Control Period Ratio')
    plt.ylabel(f'{metric_legend[metric]}')
    # plt.ylim(0.0, 20.)
    plt.xscale('log')  # Set x-axis to log scale
    plt.yscale('log')  # Set y-axis to log scale

    # Take first element of each point in all_points to get the solve times
    all_solve_times = [point[0] for point in all_points_all_tasks]
    all_metric_vals = [point[1] for point in all_points_all_tasks]
    xmin, _ = plt.xlim()
    ymin, _ = plt.ylim()

    plt.xlim(xmin, max(all_solve_times)*2.0)
    plt.ylim(ymin, max(all_metric_vals)*2.0)

    # Create the primary legend and save it to a variable
    primary_legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=2.5)

    # Add the additional custom legend
    custom_legend = plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1, -0.02), 
                               ncol=len(controlTasks), fontsize=5.7, frameon=False)

    # Re-add the primary legend using add_artist()
    plt.gca().add_artist(primary_legend)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, f"{SETTINGS['robot']}_pareto_plot.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    # TODO: Fix this using interpolation of the 
    # violation_calculations()
    # rmse_calculations()
    # traj_3D()
    # traj_inputs_vs_t()
    # traj_x_vs_y()
    # traj_xy_vs_t()
    # traj_xyz_vs_t()

    # plotTrunkResults(dirname="trunk_results", dt_string="002", metric="ISE")
    # plotTrunkResults(dirname="trunk_results", model_comparison=True)
    
    plotDiamondResults(dirname="diamond_results", metric="ISE", control_normalizer="ssmr_singleDelay")
    # plotDiamondResults(dirname="diamond_results", model_comparison=True)

    # plotDiamondTrials()

    # plot_solve_times(dirname="trunk_results")
    # plot_solve_times(dirname="diamond_results")

    # models = ["ssmr_singleDelay", "ssmr_linear", "koopman", "DMD", "tpwl"]
    # pareto_plot_diamond("ISE", models, dirname="diamond_results", dt_string=["002"])
    # pareto_plot_diamond("ISE", models, dirname="trunk_results", dt_string=["002"])