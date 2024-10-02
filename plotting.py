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
from sofacontrol.utils import qv2x, load_data, CircleObstacle, load_full_equilibrium, confidence_interval, get_metric_value
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

path = dirname(abspath(__file__))
np.set_printoptions(linewidth=300)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'FreeSerif'})
plt.rcParams.update({'mathtext.fontset': 'cm'})

FONTSCALE = 0.6

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

print("=== SOFA equilibrium point ===")
# Load equilibrium point
x_eq = load_full_equilibrium(join(path, "examples", SETTINGS['robot']))
print(x_eq.shape)

outputModel = linearModel([model.TIP_NODE], model.N_NODES, vel=False)
Z_EQ = outputModel.evaluate(x_eq, qv=False) #+ np.array([1.4, 0.0, 0.0])
if SETTINGS['robot'] == "trunk":
    Z_EQ[2] *= -1
print(Z_EQ)

# Load reference/target trajectory as defined in plotting_settings.py
TARGET = SETTINGS['select_target']

SAVE_DIR = join(path, SETTINGS['robot'], SETTINGS['save_dir'])
if not exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

metric_legend = {
    "rmse": r"Relative RMSE [%]",
    "ITAE": r"Relative ITAE [%]",
    "IAE": r"Relative IAE [%]",
    "ISE": r"Relative ISE [%]"
}

def get_metric_value(chosen_metric, error_val, ts=None):
    if chosen_metric == "rmse":
        metric_val = np.sqrt(np.mean(np.linalg.norm(error_val, axis=-1)**2, axis=-1))
    elif chosen_metric == "ITAE":
        metric_val = np.sum(np.linalg.norm(error_val, axis=-1)) * ts
    elif chosen_metric == "IAE":
        metric_val = np.sum(np.linalg.norm(error_val, axis=-1), axis=-1)
    elif chosen_metric == "ISE":
        metric_val = np.sum(np.linalg.norm(error_val, axis=-1)**2, axis=-1)
    
    return metric_val

def rmse_and_violations_MC(z, t, z_target, taskParams, save_dir="", metric="rmse", show=True):

    tickSize = 8

    SETTINGS = {'display_name': 
                {"ssm": "SSMR (6D)", 
                 "koopman": "Koopman/\nEDMD (120D)", 
                 "tpwl": "TPWL (42D)", 
                 "linear": "SSSR (6D)"},
                'color':
                {
                    "ssm": "tab:orange",
                    "koopman": "tab:green",
                    "tpwl": "tab:olive",
                    "linear": "tab:purple",
                    'target': 'black'},
                'linestyle': {
                    'target' : '--'
                },
                'linewidth' : {
                    'ssm': 2,
                    'linear': 2,
                    'tpwl': 2,
                    'koopman': 2,
                    'target': 1
                },
                'file_format': "pdf"
                }
    CONTROLS = z.keys()
    SUBPLOT_MAPPING = {
    (0, 0): ["ssm"], 
    (0, 1): ["linear"],
    (1, 0): ["koopman"],
    (1, 1): ["tpwl"]
    }

    """Compute and plot RMSEs for different number of models"""
    rmse = {}
    z_centered = {}
    z_best = {}
    idx_best = {}
    viol = {}
    mean_viol = {}
    ci_num_viol = {}
    ci_max_viol = {}
    ci_rmse = {}

    label_counter = 0
    label_list = [chr(i) for i in range(ord('a'), ord('z')+1)]

    # Get normalizer
    z_centered_normalizer = np.array(z["ssm"]) - Z_EQ
    error_normalizer = z_centered_normalizer[:, :, :2] - z_target["ssm"][:, :2]
    
    # Find the index of the trajectory with the smallest RMSE
    ts = t["ssm"] if metric == "ITAE" else None
    normalizer_error = get_metric_value(metric, error_normalizer, ts=ts)
    min_rmse_idx = np.nanargmin(normalizer_error)
    normalizer = np.mean(normalizer_error)

    for i, control in enumerate(CONTROLS):
        z_centered[control] = np.array(z[control]) - Z_EQ
        error = z_centered[control][:, :, :2] - z_target[control][:, :2]
        # rmse[control] = get_metric_value(metric, error, ts=t) # np.sqrt(np.mean(np.linalg.norm(error, axis=-1)**2, axis=-1))
        metric_error = get_metric_value(metric, error, ts=ts)

        # Store the trajectory with the smallest RMSE in z_best
        z_best[control] = z_centered[control][min_rmse_idx]

        rmse[control] = (metric_error / normalizer - 1.) * 100    
        ci_rmse[control] = confidence_interval(rmse[control])

        viol_values = np.array([[constraint.get_constraint_violation(x=None, z=z) for z in z_centered[control][:, :, :2][idx]] 
                                    for idx, constraint in enumerate(taskParams['X_list'])]) # num of simulations x num of points in trajectory
        
        viol_idxs = [idx for idx, val in enumerate(np.concatenate(viol_values)) if val]

        # Consolidate values of interest
        viol[control] = len(viol_idxs) / len(np.concatenate(viol_values)) * 100
        ci_num_viol[control] = confidence_interval(np.count_nonzero(viol_values, axis=1) / viol_values.shape[1])
        mean_viol[control] = np.mean(np.max(viol_values, axis=1))
        ci_max_viol[control] = confidence_interval(np.max(viol_values, axis=1))
    
    # Create the main figure with a 1x2 grid layout
    fig = plt.figure(figsize=(7, 3))  # Adjust the figure size as needed
    main_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4, 3])

    # Left side: Nested GridSpec for the 2x2 grid plot
    top_row_axes = []  # List to store all axes of the top row
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=main_gs[0], wspace=0.2)
    for k in range(2):
        for l in range(2):
            ax = fig.add_subplot(left_gs[k, l])
            ax.yaxis.set_major_locator(MaxNLocator(3))  # Set the maximum number of y-axis ticks to 3
            ax.xaxis.set_major_locator(MaxNLocator(3))  # Set the maximum number of y-axis ticks to 3
            ax.tick_params(axis='y', labelsize=tickSize)  # Adjust y-axis tick font size
            ax.tick_params(axis='x', labelsize=tickSize)  # Adjust x-axis tick font size
            top_row_axes.append(ax)  # Add the axis to our list
            current_controls = SUBPLOT_MAPPING[(k, l)]

            # Hide y-axis for plots that are not left-most
            # if l > 0:
            #     ax.tick_params(labelleft=False)
            ax.set_xlim(-40, 30)
            
            for control in current_controls:
                ax.plot(z_best[control][:, 0], z_best[control][:, 1], color=SETTINGS['color'][control], 
                        linewidth=SETTINGS['linewidth'][control])
                ax.plot(z_target[control][:, 0], z_target[control][:, 1], color=SETTINGS['color']['target'], 
                            ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
            
            curr_obsIdx = min_rmse_idx
            for iObs in range(len(taskParams['X_list'][curr_obsIdx].center)):
                circle = patches.Circle((taskParams['X_list'][curr_obsIdx].center[iObs][0], taskParams['X_list'][curr_obsIdx].center[iObs][1]), 
                                        taskParams['X_list'][curr_obsIdx].diameter[iObs]/2, edgecolor='red', facecolor='none')
                # Add the circle to the axes
                ax.add_patch(circle)
    
    # Add the label to the top left corner of each outer subplot
    ax.text(-1.4, 2.4, f"({label_list[0]})", transform=ax.transAxes, 
            fontsize=12, va='top', ha='left')
    label_counter += 1

    # Process error bars for rmse, num_violations, and max_violations
    error_below_rmse, error_above_rmse = zip(*[ci_rmse[control] for control in CONTROLS])
    error_below_numviol, error_above_numviol = zip(*[ci_num_viol[control] for control in CONTROLS])
    error_below_maxviol, error_above_maxviol = zip(*[ci_max_viol[control] for control in CONTROLS])
    
    # Right side: Nested GridSpec for the original 1x3 plot
    right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[1], hspace=1.0)
    ax_rmse = fig.add_subplot(right_gs[0])
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]

    rmse_vals = [np.nanmean(rmse[control], axis=-1) for control in CONTROLS]
    rmse_bar = ax_rmse.bar(xlabels, rmse_vals, color=[SETTINGS['color'][control] for control in CONTROLS])
    ax_rmse.errorbar(xlabels[1:], [np.nanmean(rmse[control], axis=-1) for control in list(CONTROLS)[1:]], 
                yerr=(error_below_rmse[1:], error_above_rmse[1:]),
                color="black", alpha=.25, fmt='o', capsize=5, markersize=3) # capsize=5
    ax_rmse.set_ylabel(f'{metric_legend[metric]}')
    ax_rmse.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    # ax_rmse.set_title('RMSE')

    # Setting the threshold for error metric to be nicely fit all bar plots
    sorted_rmse_vals = np.sort(rmse_vals)
    max_rmse = sorted_rmse_vals[-1]
    second_max_rmse = sorted_rmse_vals[-2]
    rmse_threshold = 1.2*(max_rmse + np.max(error_above_rmse))
    ax_rmse.set_ylim(0, rmse_threshold)
    # top_of_rmse_bar = [error_above_rmse[idx] + rmse[control] for idx, control in enumerate(CONTROLS)]

    for bar, rmse_val, upper_bound_rmse in zip(rmse_bar, rmse_vals, error_above_rmse):
        
        if np.isclose(round(rmse_val), 0.):
            ax_rmse.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, 'Baseline', ha='center', va='bottom', fontsize=10, fontweight='bold')
        elif rmse_val > 2 * second_max_rmse:
            ax_rmse.text(bar.get_x() + bar.get_width() / 2, 0.85*rmse_threshold, f'↑{round(rmse_val)}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax_rmse.text(bar.get_x() + bar.get_width() / 2, rmse_val + upper_bound_rmse + 0.1, f'↑{round(rmse_val)}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_viol = fig.add_subplot(right_gs[1])
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    viol_ratio_values = [viol[control] for control in CONTROLS]
    viol_bars = ax_viol.bar(xlabels, viol_ratio_values, color=[SETTINGS['color'][control] for control in CONTROLS])
    ax_viol.errorbar(xlabels, viol_ratio_values, 
                yerr=(error_below_numviol, error_above_numviol),
                color="black", alpha=.25, fmt='o', capsize=5, markersize=3)
    ax_viol.set_ylabel(f'Violation Ratio\n[%]')
    ax_viol.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    # ax_viol.set_title('Constraint Violation Ratio')

    for viol_bar, viol_val, upper_bound_numviol in zip(viol_bars, viol_ratio_values, error_above_numviol):
        ax_viol.text(viol_bar.get_x() + viol_bar.get_width() / 2, viol_val + upper_bound_numviol + 0.1, 
                     f'{viol_val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    
    ax_max_viol = fig.add_subplot(right_gs[2])
    maxviol_values = [mean_viol[control] for control in CONTROLS]
    maxviol_bars = ax_max_viol.bar(xlabels, maxviol_values, color=[SETTINGS['color'][control] for control in CONTROLS])
    ax_max_viol.errorbar(xlabels, maxviol_values, 
                yerr=(error_below_maxviol, error_above_maxviol),
                color="black", alpha=.25, fmt='o', capsize=5, markersize=3)
    ax_max_viol.set_ylabel(f'Max Violation\n[mm]')
    ax_max_viol.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    # ax_max_viol.set_title('Average Max Constraint Violation')

    for maxviol_bar, maxviol_val, upper_bound_maxviol in zip(maxviol_bars, maxviol_values, error_above_maxviol):
        ax_max_viol.text(maxviol_bar.get_x() + maxviol_bar.get_width() / 2, maxviol_val + upper_bound_maxviol + 0.1, 
                     f'{maxviol_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for ax in [ax_rmse, ax_viol, ax_max_viol]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add the label to the top left corner of each outer subplot
        ax.text(-0.08, 1.35, f"({label_list[label_counter]})", transform=ax.transAxes, 
                fontsize=10, va='top', ha='left')
        label_counter += 1
    
    fig.tight_layout()
    
    if save_dir:
        plt.savefig(join(save_dir, f"trunk_with_constraints.{SETTINGS['file_format']}"), bbox_inches='tight', 
                        dpi=300, format=SETTINGS['file_format'], transparent=True)
    if show:
        plt.show()