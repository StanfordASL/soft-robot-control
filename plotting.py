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
from sofacontrol.utils import qv2x, load_data, CircleObstacle, load_full_equilibrium, confidence_interval
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

def rmse_and_violations_MC(z, z_target, taskParams, save_dir="", show=True):

    tickSize = 8

    SETTINGS = {'display_name': 
                {"ssm": "SSMR", 
                 "koopman": "EDMD", 
                 "tpwl": "TPWL", 
                 "linear": "SSSR"},
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
                'file_format': "png"
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

    for i, control in enumerate(CONTROLS):
        z_centered[control] = np.array(z[control]) - Z_EQ
        error = z_centered[control][:, :, :2] - z_target[control][:, :2]
        rmse[control] = np.sqrt(np.mean(np.linalg.norm(error, axis=-1)**2, axis=-1))
        ci_rmse[control] = confidence_interval(rmse[control])

        viol_values = np.array([[constraint.get_constraint_violation(x=None, z=z) for z in z_centered[control][:, :, :2][idx]] 
                                    for idx, constraint in enumerate(taskParams['X_list'])]) # num of constraints x num of points in trajectory
        
        viol_idxs = [idx for idx, val in enumerate(np.concatenate(viol_values)) if val]

        # Consolidate values of interest
        viol[control] = len(viol_idxs) / len(np.concatenate(viol_values))
        ci_num_viol[control] = confidence_interval(np.count_nonzero(viol_values, axis=1) / viol_values.shape[1])
        mean_viol[control] = np.mean(np.max(viol_values, axis=1))
        ci_max_viol[control] = confidence_interval(np.max(viol_values, axis=1))

        # Find the index of the trajectory with the smallest RMSE
        min_rmse_idx = np.nanargmin(rmse[control])
        
        # Store the trajectory with the smallest RMSE in z_best
        z_best[control] = z_centered[control][min_rmse_idx]
        idx_best[control] = min_rmse_idx
    
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
            
            for control in current_controls:
                ax.plot(z_best[control][:, 0], z_best[control][:, 1], color=SETTINGS['color'][control], 
                        linewidth=SETTINGS['linewidth'][control])
                ax.plot(z_target[control][:, 0], z_target[control][:, 1], color=SETTINGS['color']['target'], 
                            ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
            
            curr_obsIdx = idx_best[current_controls[0]]
            for iObs in range(len(taskParams['X_list'][curr_obsIdx].center)):
                circle = patches.Circle((taskParams['X_list'][curr_obsIdx].center[iObs][0], taskParams['X_list'][curr_obsIdx].center[iObs][1]), 
                                        taskParams['X_list'][curr_obsIdx].diameter[iObs]/2, edgecolor='red', facecolor='none')
                # Add the circle to the axes
                ax.add_patch(circle)
        
    # Process error bars for rmse, num_violations, and max_violations
    error_below_rmse, error_above_rmse = zip(*[ci_rmse[control] for control in CONTROLS])
    error_below_numviol, error_above_numviol = zip(*[ci_num_viol[control] for control in CONTROLS])
    error_below_maxviol, error_above_maxviol = zip(*[ci_max_viol[control] for control in CONTROLS])
    
    # Right side: Nested GridSpec for the original 1x3 plot
    right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[1], hspace=0.3)
    ax_rmse = fig.add_subplot(right_gs[0])
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    ax_rmse.bar(xlabels, [np.nanmean(rmse[control], axis=-1) for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    ax_rmse.errorbar(xlabels, [np.nanmean(rmse[control], axis=-1) for control in CONTROLS], 
                yerr=(error_below_rmse, error_above_rmse),
                color="black", alpha=.25, fmt='o', capsize=5, markersize=3) # capsize=5
    ax_rmse.set_ylabel(r'RMSE [mm]')
    # ax_rmse.set_title('RMSE')

    ax_viol = fig.add_subplot(right_gs[1])
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    ax_viol.bar(xlabels, [viol[control] for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    ax_viol.errorbar(xlabels, [viol[control] for control in CONTROLS], 
                yerr=(error_below_numviol, error_above_numviol),
                color="black", alpha=.25, fmt='o', capsize=5, markersize=3)
    ax_viol.set_ylabel(r'Violation Ratio [%]')
    # ax_viol.set_title('Constraint Violation Ratio')
    
    ax_max_viol = fig.add_subplot(right_gs[2])
    ax_max_viol.bar(xlabels, [mean_viol[control] for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    ax_max_viol.errorbar(xlabels, [mean_viol[control] for control in CONTROLS], 
                yerr=(error_below_maxviol, error_above_maxviol),
                color="black", alpha=.25, fmt='o', capsize=5, markersize=3)
    ax_max_viol.set_ylabel(r'Max Violation [mm]')
    # ax_max_viol.set_title('Average Max Constraint Violation')

    for ax in [ax_rmse, ax_viol, ax_max_viol]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    fig.tight_layout()
    
    if save_dir:
        plt.savefig(join(save_dir, f"trunk_with_constraints.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if show:
        plt.show()