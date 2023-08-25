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
import matplotlib.ticker as mticker


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
    SETTINGS = {'display_name': 
                {"ssm": "SSMR", 
                 "koopman": "Koopman", 
                 "tpwl": "TPWL", 
                 "linear": "Linear"},
                'color':
                {
                    "ssm": "tab:orange",
                    "koopman": "tab:green",
                    "tpwl": "tab:red",
                    "linear": "tab:purple"},
                'file_format': "png"
                }
    CONTROLS = z.keys()

    """Compute and plot RMSEs for different number of models"""
    rmse = {}
    z_centered = {}
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
    
    # Process error bars for rmse, num_violations, and max_violations
    error_below_rmse, error_above_rmse = zip(*[ci_rmse[control] for control in CONTROLS])
    error_below_numviol, error_above_numviol = zip(*[ci_num_viol[control] for control in CONTROLS])
    error_below_maxviol, error_above_maxviol = zip(*[ci_max_viol[control] for control in CONTROLS])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    ax.bar(xlabels, [np.nanmean(rmse[control], axis=-1) for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    ax.errorbar(xlabels, [np.nanmean(rmse[control], axis=-1) for control in CONTROLS], 
                yerr=(error_below_rmse, error_above_rmse),
                color="black", alpha=.25, fmt='o', capsize=5)
    ax.set_ylabel(r'RMSE [mm]')
    ax.set_title('RMSE')

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    xlabels = [SETTINGS['display_name'][control] for control in CONTROLS]
    axs[0].bar(xlabels, [viol[control] for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    axs[0].errorbar(xlabels, [viol[control] for control in CONTROLS], 
                yerr=(error_below_numviol, error_above_numviol),
                color="black", alpha=.25, fmt='o', capsize=5)
    axs[0].set_ylabel(r'Violation Ratio [%]')
    axs[0].set_title('Constraint Violation Ratio')
    axs[1].bar(xlabels, [mean_viol[control] for control in CONTROLS], color=[SETTINGS['color'][control] for control in CONTROLS])
    axs[1].errorbar(xlabels, [mean_viol[control] for control in CONTROLS], 
                yerr=(error_below_maxviol, error_above_maxviol),
                color="black", alpha=.25, fmt='o', capsize=5)
    axs[1].set_ylabel(r'Max Violation [mm]')
    axs[1].set_title('Average Max Constraint Violation')
    plt.savefig(join(SAVE_DIR, f"{TARGET}_ratio_violations.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    
    if save_dir:
        plt.savefig(join(save_dir, f"{TARGET}_rmse_vs_n_models.{SETTINGS['file_format']}"), bbox_inches='tight', dpi=200)
    if show:
        plt.show()