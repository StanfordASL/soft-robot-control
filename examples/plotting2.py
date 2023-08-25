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


from sofacontrol.measurement_models import linearModel
from sofacontrol.utils import qv2x, load_data, CircleObstacle, load_full_equilibrium, add_decimal
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
    if SETTINGS['robot'] == "trunk":
        SIM_DATA[control]['z'][:, 2] *= -1
    SIM_DATA[control]['u'] = control_data['u'][idx:, :]
    SIM_DATA[control]['info']['solve_times'] = control_data['info']['solve_times']
    SIM_DATA[control]['info']['real_time_limit'] = control_data['info']['rollout_time']

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
            z_centered = SIM_DATA[control]['z']- Z_EQ
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

    for ax, coord in [(ax1, 0), (ax2, 1)]:
        for control in CONTROLS: # + ['target']:
            zf_target = f(SIM_DATA[control]['t'])

            # Don't center coordinates if koopman
            if control == "koopman":
                z_centered = SIM_DATA[control]['z'] - Z_EQ
            else:
                z_centered = SIM_DATA[control]['z'] - Z_EQ
            
            ax.plot(SIM_DATA[control]['t'][:-2], z_centered[:-2, coord],
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

    f = interp1d(target['t'], target['z'], axis=0)

    for ax, coord in [(ax1, 0), (ax2, 1), (ax3, 2)]:
        for control in CONTROLS: # + ['target']:
            zf_target = f(SIM_DATA[control]['t'][:-2])

            # Don't center coordinates if koopman
            if control == "koopman":
                z_centered = SIM_DATA[control]['z'] - Z_EQ
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

    f = interp1d(target['t'], target['z'], axis=0)

    for control in CONTROLS:
        zf_target = f(SIM_DATA[control]['t'][:-1])
        
        # Don't center coordinates if koopman
        if control == "koopman":
            z_centered = SIM_DATA[control]['z'] - Z_EQ
        else:
            z_centered = SIM_DATA[control]['z'] - Z_EQ

        # if control == "ssmr_origin":
        #     z_centered = z_centered[:-1, :]
        if (TARGET == "circle" and SETTINGS['robot'] == "hardware") or (TARGET == "custom" and SETTINGS['robot'] == "hardware"):
            # errors are to be measured in 3D
            err[control] = (z_centered[:-1, :] - zf_target)
        else:
            # errors are to be measured in 2D
            err[control] = (z_centered[:-1, :2] - zf_target[:, :2])
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

def plot_bar_chart_for_multiple_dts(ax, data, rmse_threshold=10, set_threshold=True):
    # Assuming data is a dictionary of dictionaries in the format: data[control][dt]

    CONTROLS = ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel", "koopman", "ssmr_linear", "DMD", "tpwl"]
    dts = sorted(data[CONTROLS[0]].keys()) # Assuming all controls have the same dts
    num_dts = len(dts)
    width = 0.8 / num_dts

    # Group bars by control
    dt_positions = [i + j * width for i in range(len(CONTROLS)) for j in range(num_dts)]
    for idx, dt in enumerate(dts):
        rmse_vals = [data[control][dt] for control in CONTROLS]
        dt_colors = [get_shade(matplotlib.colors.to_rgb(SETTINGS['color'][control]), (idx*0.2)) if val <= rmse_threshold 
                     else get_shade((0.8, 0.8, 0.8), 0.2) for val, control in zip(rmse_vals, CONTROLS)]
        hatches = ['//' if val > rmse_threshold else '' for val in rmse_vals]
        edgecolors = ['darkgray' if hatch else dt_colors[i] for i, hatch in enumerate(hatches)]
        positions = dt_positions[idx::num_dts]
        bars = ax.bar(positions, rmse_vals, width, color=dt_colors, label=f'dt={dt}', edgecolor=edgecolors)

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    
    # Configure primary and secondary x-axis labels
    control_ticks = [0.2, 1.2, 2.2, 3.3, 4.3, 5.3, 6.3]
    ax.set_xticks(control_ticks)
    ax.set_xticklabels([SETTINGS['display_name'][control] for control in CONTROLS], position=(0, -0.01), fontsize=9)
    
    ax.set_xticks(dt_positions, minor=True)
    ax.set_xticklabels([f'{dt}' for dt in dts] * len(CONTROLS), minor=True, fontsize=8, rotation=30)

    # Remove the major tick lines
    ax.tick_params(axis='x', length=0, pad=20)
    if set_threshold:
        ax.set_ylim(0, rmse_threshold)

def plotTrunkResults():
    # Define a default dictionary to store the data
    def nested_dict():
        return defaultdict(nested_dict)
    
    CONTROLS = ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel", "koopman", "ssmr_linear", "DMD", "tpwl"]
    SUBPLOT_MAPPING = {
    (0, 0): ["ssmr_singleDelay"], # ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel"]
    (0, 1): ["koopman"],
    (1, 0): ["ssmr_linear", "DMD"],
    (1, 1): ["tpwl"] # Add TPWL here
    }
    controlTasks = ["ASL", "pacman", "stanford"]
    titles = [
        ["ASL Trajectory", "Pacman Trajectory", "Stanford Trajectory"], 
        ["", "", ""],
        ["", "", ""]]
    
    simData = nested_dict()
    rmse = nested_dict()
    targetTrajData = nested_dict()
    z_centeredData = nested_dict()
    solve_times = {}

    # Go through each possible control task
    for task in controlTasks:
        taskFile = join(path, SETTINGS['robot'], 'control_tasks', task + '.pkl')
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

                # Extract RMSE
                zf_target = f_target(simData[control][dt]['t'])
                z_centered = simData[control][dt]['z'] - Z_EQ
                error = (z_centered[:, :2] - zf_target[:, :2])
                rmse[task][control][dt] = np.sqrt(np.mean(np.linalg.norm(error, axis=1)**2, axis=0))

                # Grab target and control trajectory data for plotting later
                targetTrajData[task][control][dt] = zf_target
                z_centeredData[task][control][dt] = z_centered

    # Create main figure and gridspec
    
    fig = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)  # 3x3 grid
    DT_PLOT = 0.02
    
    handles, labels = [], []
    y_axis_limits = None
    top_row_axes = []  # List to store all axes of the top row

    for j, task in enumerate(controlTasks):  # Loop over columns

        # Top row: Each plot is further divided into 2x2 grid
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, j])
        for k in range(2):
            for l in range(2):
                ax = fig.add_subplot(gs_sub[k, l])
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
                    labels.append(SETTINGS['legend_name'][control])


                # If it's the top-left subplot, get its y-axis limits
                if y_axis_limits is None and j == 0 and k == 0 and l == 0:
                    y_axis_limits = ax.get_ylim()

        # Middle row (Your bar plots)
        ax = fig.add_subplot(gs[1, j])
        ax.set_title(titles[1][j])
        plot_bar_chart_for_multiple_dts(ax, rmse[task])

        # Set y-axis label for the first column and hide it for the others in the MIDDLE ROW
        if j == 0:  # First column
            ax.set_ylabel('RMSE [mm]')
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

        # Bottom row
        boxplot_data = {}
        avg_solve_times = {}
        for control in CONTROLS:
            all_solve_times = [1000 * time for dt in simData[control] for time in simData[control][dt]['info']['solve_times']] # Display in ms
            boxplot_data[control] = all_solve_times
            avg_solve_times[control] = np.mean(all_solve_times)  # Compute and store the average
        
        ax = fig.add_subplot(gs[2, :])
        ax.set_yscale('log')
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_scientific(False)

        ax.boxplot(boxplot_data.values(), vert=True, patch_artist=True, notch=True)
        ax.set_xticklabels(SETTINGS['display_name'][control] for control in boxplot_data.keys())
        ax.set_ylabel('Solve Times [ms]', fontweight='normal')

        # Annotate each boxplot with its average solve time
        for i, control in enumerate(CONTROLS):
            avg_time = avg_solve_times[control]
            ax.annotate(f"{avg_time:.2f}", (i + 1, avg_time),
                        textcoords="offset points", xytext=(61,0), ha='center', 
                        fontsize=10, color='#1f77b4', fontweight='bold')

    
    # Set the ylim for the top row subplots
    for ax in top_row_axes:
        ax.set_ylim(y_axis_limits)
        
    # Legend for the top row
    handle_label_dict = dict(zip(labels, handles))
    unique_labels = list(handle_label_dict.keys())
    unique_handles = [handle_label_dict[label] for label in unique_labels]
    fig.legend(unique_handles, unique_labels, loc='upper center', 
           ncol=len(unique_labels), bbox_to_anchor=(0.5, 0.693),  # Adjust these values for legend placement
           bbox_transform=fig.transFigure,  # This ensures the bbox_to_anchor coordinates are in figure fraction
           fontsize='small')
    
    plt.tight_layout()
    plt.show()

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


def plotDiamondResults():
    # Define a default dictionary to store the data
    def nested_dict():
        return defaultdict(nested_dict)
    
    CONTROLS = ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel", "koopman", "ssmr_linear", "DMD", "tpwl"]
    SUBPLOT_MAPPING = {
    (0, 0): ["ssmr_singleDelay"], # ["ssmr_singleDelay", "ssmr_delays", "ssmr_posvel"]
    (0, 1): ["koopman"],
    (1, 0): ["ssmr_linear", "DMD"],
    (1, 1): ["tpwl"] # Add TPWL here
    }
    controlTasks = ["figure8", "circle", "star"]
    titles = [
        ["ASL Trajectory", "Pacman Trajectory", "Stanford Trajectory"], 
        ["", "", ""],
        ["", "", ""]]
    
    simData = nested_dict()
    rmse = nested_dict()
    targetTrajData = nested_dict()
    z_centeredData = nested_dict()

    # Go through each possible control task
    for task in controlTasks:
        taskFile = join(path, SETTINGS['robot'], 'control_tasks', task + '.pkl')
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
                rmse[task][control][dt] = np.sqrt(np.mean(np.linalg.norm(error, axis=1)**2, axis=0))

                # Grab target and control trajectory data for plotting later
                targetTrajData[task][control][dt] = zf_target
                z_centeredData[task][control][dt] = z_centered

    # Create main figure and gridspec
    
    fig = plt.figure(figsize=(30, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)  # 3x3 grid
    
    handles, labels = [], []
    y_axis_limits = None
    top_row_axes = []  # List to store all axes of the top row

    for j, task in enumerate(controlTasks):  # Loop over columns
        DT_PLOT = 0.02

        # Top row: Each plot is further divided into 2x2 grid
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, j])
        for k in range(2):
            for l in range(2):
                ax = fig.add_subplot(gs_sub[k, l])
                top_row_axes.append(ax)  # Add the axis to our list
                current_controls = SUBPLOT_MAPPING[(k, l)]

                # Hide y-axis for plots that are not left-most
                if l > 0:
                    ax.tick_params(labelleft=False)
                
                # TODO: Refactor this. Should return best performing model
                if task == "circle":
                    DT_PLOT = 0.05

                for control in current_controls:
                    # Get desired trajectory and controlled trajectory
                    desired_target = targetTrajData[task][control][DT_PLOT]
                    controlled_traj = z_centeredData[task][control][DT_PLOT]
                    
                    if task == "circle" or task == "star":
                        ax.plot(desired_target[:, 1], desired_target[:, 2], color=SETTINGS['color']['target'], 
                            ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
                        controlled_traj = controlled_traj[:, 1:]
                    else:
                        ax.plot(desired_target[:, 0], desired_target[:, 1], color=SETTINGS['color']['target'], 
                            ls=SETTINGS['linestyle']['target'], alpha=.9, linewidth=SETTINGS['linewidth']['target'], label='Target', zorder=1)
                        controlled_traj = controlled_traj[:, :2]

                    line, = ax.plot(controlled_traj[:, 0], controlled_traj[:, 1],
                    color=SETTINGS['color'][control],
                    label=SETTINGS['display_name'][control],
                    linewidth=SETTINGS['linewidth'][control],
                    ls=SETTINGS['linestyle'][control], markevery=20,
                    alpha=SETTINGS['alpha'][control])

                    handles.append(line)
                    labels.append(SETTINGS['legend_name'][control])


                # If it's the top-left subplot, get its y-axis limits
                if y_axis_limits is None and j == 0 and k == 0 and l == 0:
                    y_axis_limits = ax.get_ylim()

        # Middle row (Your bar plots)
        ax = fig.add_subplot(gs[1, j])
        ax.set_title(titles[1][j])
        plot_bar_chart_for_multiple_dts(ax, rmse[task], set_threshold=False)

        # Set y-axis label for the first column and hide it for the others in the MIDDLE ROW
        if j == 0:  # First column
            ax.set_ylabel('RMSE [mm]')
        else:
            ax.set_ylabel('')
            # ax.tick_params(labelleft=False)

        # Bottom row
        boxplot_data = {}
        avg_solve_times = {}
        for control in CONTROLS:
            all_solve_times = [1000 * time for dt in simData[control] for time in simData[control][dt]['info']['solve_times']] # Display in ms
            boxplot_data[control] = all_solve_times
            avg_solve_times[control] = np.mean(all_solve_times)  # Compute and store the average
        
        ax = fig.add_subplot(gs[2, :])
        ax.set_yscale('log')
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_scientific(False)

        ax.boxplot(boxplot_data.values(), vert=True, patch_artist=True, notch=True)
        ax.set_xticklabels(SETTINGS['display_name'][control] for control in boxplot_data.keys())
        ax.set_ylabel('Solve Times [ms]', fontweight='normal')

        # Annotate each boxplot with its average solve time
        for i, control in enumerate(CONTROLS):
            avg_time = avg_solve_times[control]
            ax.annotate(f"{avg_time:.2f}", (i + 1, avg_time),
                        textcoords="offset points", xytext=(65,5), ha='center', 
                        fontsize=10, color='#1f77b4', fontweight='bold')

    
    # Set the ylim for the top row subplots
    # for ax in top_row_axes:
    #     ax.set_ylim(y_axis_limits)
        
    # Legend for the top row
    handle_label_dict = dict(zip(labels, handles))
    unique_labels = list(handle_label_dict.keys())
    unique_handles = [handle_label_dict[label] for label in unique_labels]
    fig.legend(unique_handles, unique_labels, loc='upper center', 
           ncol=len(unique_labels), bbox_to_anchor=(0.5, 0.693),  # Adjust these values for legend placement
           bbox_transform=fig.transFigure,  # This ensures the bbox_to_anchor coordinates are in figure fraction
           fontsize='small')
    
    plt.tight_layout()
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

    # plotTrunkResults()
    # plotDiamondResults()
    plotDiamondTrials()