import os
from datetime import datetime
"""
Provides direct interface with SOFA and is either opened from sofa gui or run in the following command style:
$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py
Imports problem_specification to add specified robot (FEM model) and specific controller.

Automatically:
    - collects decay trajectories based on a given input sequence
    - saves the collected trajectories to the directory specified in settings.yaml
"""
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import numpy as np
from tqdm.auto import tqdm

import numpy as np
import yaml
import pickle

import sys
from multiprocessing import Process

from os.path import dirname, join, isdir, split, exists
from os import listdir
from shutil import copy
from sofacontrol.utils import generateModel, createTargetTrajectory, createObstacleConstraint, generateObstacles, createControlConstraint, save_data, load_data

# import plotting as plot

path = dirname(os.path.abspath(__file__))

# SETTINGS
METHOD = "linear" # "tpwl" # "koopman" # "ssm" # "ssmr" (full delay) # "ssmr_1delay", "ssmr_origin" (pos-vel)
SAVE_DIR = "/home/jalora/Desktop/CL_trunk_obstacle_analysis" #TODO: Add special directory for RMSE evaluations
NUM_SIMS = 100
ROBOT = "trunk"
DT = 0.02

# Default nodes are the "end effector (51)" and the "along trunk (22, 37) = (4th, 7th) top link "
DEFAULT_OUTPUT_NODES = [51, 22, 37]
TIP_NODE = 51
N_NODES = 709

######## User Options ########
saveControlTask = False
createNewTask = False
createNewObstacles = False

# Control Task Params
controlTask = "custom" # figure8, circle, pacman, or custom
trajAmplitude = 30
pathToTraceImage = "/home/jalora/Desktop/ETH.png"

# Constrol Constraints
if ROBOT == "hardware":
    u_min, u_max = 0.0, 2500.0
    du_max = None
else:
    u_min, u_max = 0.0, 800.0
    du_max = None

num_obstacles = 20
min_diameter, max_diameter = 8, 12
min_distance_from_origin = 10

# Define target trajectory paths
mainPath = join(path, "examples", ROBOT)
trajDir = join(mainPath, "control_tasks")
taskFile = join(trajDir, controlTask + "_" + str(NUM_SIMS) + "_" + "obstacles" + ".pkl")
singleTaskFile = join(trajDir, controlTask + ".pkl")

if METHOD == "koopman":
    from examples.trunk.trunk_koopman import run_koopman as run_scp
    sim_prefix_save = "koopman_sim"
elif METHOD == "tpwl":
    from examples.trunk.trunk import run_scp
    sim_prefix_save = "tpwl_sim"
elif "ssm" in METHOD:
    from examples.trunk.trunk_SSM import run_scp_call as run_scp
    sim_prefix_save = "ssmr_sim"
    if METHOD == "ssm": # TODO: Hack to not break the random obstacle generation functionality. Refactor!
        sim_prefix_save = "ssmr_sim"
    else:
        sim_prefix_save = METHOD + "_sim"
elif METHOD == "linear":
    from examples.trunk.trunk_SSM import run_scp_call as run_scp
    sim_prefix_save = "linear_sim"
else:
    raise ValueError("Invalid method")

def createScene_CL(rootNode, model, dt, T):
    # Start building scene
    rootNode.addObject("RequiredPlugin", name="SoftRobots", printLog=False)
    rootNode.addObject("RequiredPlugin", name="SofaPython3", printLog=False)
    rootNode.addObject("RequiredPlugin", name="SofaSparseSolver")
    rootNode.addObject("RequiredPlugin", name="SofaPreconditioner")
    rootNode.addObject('RequiredPlugin', pluginName='SofaOpenglVisual')
    # rootNode.addObject('RequiredPlugin', pluginName='SofaMatrix')

    rootNode.addObject("VisualStyle", displayFlags="showBehavior")
    rootNode.addObject("FreeMotionAnimationLoop")
    # rootNode.addObject('DefaultAnimationLoop', name='loop')
    rootNode.addObject("GenericConstraintSolver", maxIterations=100, tolerance=1e-5)
    rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
    rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
    rootNode.addObject('DefaultVisualManagerLoop')

    prob = run_scp(METHOD, model, dt, T)
    prob.checkDefinition()

    # Set the gravity and simulation time step
    rootNode.gravity = prob.Robot.gravity
    rootNode.dt = prob.Robot.dt

    # Define filename prefix for saving
    if prob.opt.get('save_prefix') is None:
        prob.opt['save_prefix'] = datetime.now().strftime("%Y%m%d_%H%M")

    robot = rootNode.addChild(prob.Robot.robot)

    if robot.min_force > [0] * len(robot.actuator_list):
        print('[PROBLEM WARNING]   Minimal force for 1 or more actuators set to {}, which is higher than 0. '
              'Undesired behavior might occur'.format(max(robot.min_force)))

    rootNode.addObject(prob.ControllerClass(rootNode=rootNode,
                                            robot=robot,
                                            snapshots=prob.snapshots,
                                            controller=prob.controller,
                                            measurement_model=prob.measurement_model,
                                            output_model=prob.output_model,
                                            simdata_dir=prob.simdata_dir,
                                            snapshots_dir=prob.snapshots_dir,
                                            opt=prob.opt))
    
    rootNode.autopaused = False  # Enables terminating simulation at the end when running from command line
    return rootNode

def generate_task_params_obstacles():

    taskParams = {}
    ######## Generate model and setup control task ########
    # Set directory for SSM Models
    # pathToModel = "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic/origin/SSMmodel_delay-embedding" # join(path, "SSMmodels", "model_004")
    pathToModel = "/home/jalora/Desktop/trunk_origin/000/SSMmodel_delay-embedding_ROMOrder=3_localV" # join(path, "SSMmodels", "model_004")
    model = generateModel(mainPath, pathToModel, [TIP_NODE], N_NODES)

    ######## Define new control task ########
    if createNewTask:
        if "ssm" in METHOD:
            ######## Define the trajectory ########
            zf_target, t = createTargetTrajectory(controlTask, 'trunk', model.y_eq, model.output_dim, amplitude=trajAmplitude, pathToImage=pathToTraceImage)
            z = model.zfyf_to_zy(zf=zf_target)
        else:
            raise RuntimeError('Creating new control tasks only implemented for SSM')
    
    ######## Define new control constraint ########
    U, dU = createControlConstraint(u_min, u_max, model.input_dim, du_max=du_max)

    ######## Generate new obstacles for each desired simulation ########
    X_list = []
    if createNewObstacles:
        if "ssm" in METHOD:
            for sim_idx in tqdm(range(NUM_SIMS)):
                obstacles = generateObstacles(num_obstacles, min_diameter, max_diameter, min_distance_from_origin, min_distance_between_obstacles=3)
                obstacleDiameter = [d[0] for d in obstacles]
                obstacleLoc = [d[1] for d in obstacles]

                # Create obstacle constraint for current simulation
                X = createObstacleConstraint(model.output_dim, model.y_ref, obstacleDiameter, obstacleLoc)
                X_list.append(X)
        else:
            raise RuntimeError('Creating new obstacles only implemented for SSM')
    else:
        taskParams = load_data(taskFile)
        if taskParams['z'].shape[1] < model.output_dim:
            taskParams['z'] = np.hstack((taskParams['z'], np.zeros((taskParams['z'].shape[0], model.output_dim - taskParams['z'].shape[1]))))
        
        # If a new task is desired, then store it as the target trajectory
        if createNewTask:
            taskParams['z'] = z

    # TODO: Assume constraints are CircleObstacle type
    ######## Save Target Trajectory and Constraints ########
    taskParams = {'z': z, 't': t, 'X': None, 'X_list': X_list, 'U': U, 'dU': dU, 'obstacleDiameter': obstacleDiameter, 'obstacleLoc': obstacleLoc}
    if saveControlTask:
        if "ssm" in METHOD:
            save_data(taskFile, taskParams)
        else:
            raise RuntimeError('Saving task parameters only supported for SSMR')

def doSimulationsRandomizeObstacles():
    #  Allows executing from terminal directly
    #  Requires adjusting to own path
    sofa_lib_path = "/home/jalora/sofa/build/lib"

    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    # create a flag to indicate if the script failed
    with open("/home/jalora/soft-robot-control/gusto_failed_flag.pkl", "wb") as f:
        pickle.dump(False, f)
    
    taskParams = load_data(taskFile)
    sim_save_dir = join(SAVE_DIR, METHOD)

    ######## Generate model and setup control task ########
    # Set directory for SSM Models
    # pathToModel = "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic/origin/SSMmodel_delay-embedding" # join(path, "SSMmodels", "model_004")
    # model = None
    if "ssm" in METHOD:
        pathToModel = "/home/jalora/Desktop/trunk_origin/000/SSMmodel_delay-embedding_ROMOrder=3_localV" # join(path, "SSMmodels", "model_004")
        # model = generateModel(mainPath, pathToModel, [TIP_NODE], N_NODES, modelType=METHOD.split("_", 1)[1])
        model = generateModel(mainPath, pathToModel, [TIP_NODE], N_NODES)

        if taskParams['z'].shape[1] < model.output_dim:
            taskParams['z'] = np.hstack((taskParams['z'], np.zeros((taskParams['z'].shape[0], model.output_dim - taskParams['z'].shape[1]))))
    elif METHOD == "linear":
        pathToModel = "/home/jalora/Desktop/trunk_origin/000/SSMmodel_delay-embedding_ROMOrder=3_localV" # join(path, "SSMmodels", "model_004")
        model = generateModel(mainPath, pathToModel, [TIP_NODE], N_NODES, modelType="linear", isLinear=True)
        # model = generateModel(mainPath, pathToModel, [TIP_NODE], N_NODES)

        if taskParams['z'].shape[1] < model.output_dim:
            taskParams['z'] = np.hstack((taskParams['z'], np.zeros((taskParams['z'].shape[0], model.output_dim - taskParams['z'].shape[1]))))

    for j in tqdm(range(0, len(taskParams['X_list']))):
        if exists(join(sim_save_dir, f"{sim_prefix_save}_{j}.pkl")) or exists(join(sim_save_dir, f"{sim_prefix_save}_{j}_failed.pkl")):
            continue
        # Create save directory if it does not exist
        if not os.path.exists(sim_save_dir):
            os.makedirs(sim_save_dir)

        if "ssm" in METHOD or METHOD == "linear":
            from examples.trunk.trunk_SSM import run_gusto_solver_call
            run_solver = run_gusto_solver_call
        elif METHOD == "koopman":
            from examples.trunk.trunk_koopman import run_koopman_solver_call
            run_solver = run_koopman_solver_call
        elif METHOD == "tpwl":
            from examples.trunk.trunk import run_gusto_solver_call
            run_solver = run_gusto_solver_call
        
        # run closed loop simulation with an adiabatic model interpolating the models in use_models
        # start process running gusto_solver with argument z
        taskParams['X'] = taskParams['X_list'][j]
        p_gusto = Process(target=run_solver, args=(taskParams, model))
        p_gusto.start()
        # start SOFA simulation
        root = Sofa.Core.Node()
        rootNode = createScene_CL(root, model, DT, T=11.)
        Sofa.Simulation.init(root)
        while True:
            with open("/home/jalora/soft-robot-control/gusto_failed_flag.pkl", "rb") as f:
                gusto_fail = pickle.load(f)
            if gusto_fail:
                raise RuntimeError("gusto_solver failed")
            Sofa.Simulation.animate(root, root.dt.value)
            if rootNode.autopaused == True:
                break  # Terminates simulation when autopaused
        # terminate process running gusto_solver
        p_gusto.terminate()
        p_gusto.join()
        p_gusto.close()
        # copy the resulting trajectory data to save_dir. TODO: If simulation fails and file is not saved, then this will just copy the previous file.
        copy(os.path.join("/home/jalora/soft-robot-control/examples/trunk", f"{sim_prefix_save}.pkl"), join(sim_save_dir, f"{sim_prefix_save}_{j}.pkl"))

        os.execv(sys.executable, ['python'] + sys.argv)

    print('All simulations finished, exiting...')

def plotResults():
    import plotting as plot
    from scipy.interpolate import interp1d

    sim_prefix = {"ssm": "ssmr_sim", "linear": "linear_sim", "koopman": "koopman_sim", "tpwl": "tpwl_sim"}

    t0 = 1.
    z = {}
    z_target = {}
    # solve_times = {}

    taskParams = load_data(taskFile)

    for control in ["ssm", "linear", "koopman", "tpwl"]: # ["idw", "nn", "qp"]: # "ct", 
        z[control] = []
        z_target[control] = []
        z_interp = interp1d(taskParams['t'], taskParams['z'], axis=0)
        for j in range(NUM_SIMS):
            sim_save_dir = join(SAVE_DIR, control)

            if exists(join(sim_save_dir, f"{sim_prefix[control]}_{j}.pkl")):
                with open(join(sim_save_dir, f"{sim_prefix[control]}_{j}.pkl"), "rb") as f:
                    sim = pickle.load(f)
                # Recenter time
                idx = np.argwhere(sim['t'] > t0)[0][0]
                sim['t'] = sim['t'][idx:] - sim['t'][idx]
                z_target[control] = z_interp(sim['t'])
                z_j = sim['z'][idx:, 3:]
                z_j[:, 2] *= -1
            elif exists(join(sim_save_dir, f"{sim_prefix[control]}_{j}_failed.pkl")):
                print("simulation failed: ", dir, j)
                z_j = np.full((1001, 3), np.nan) #TODO: Currently hard-coded
                z_target[control] = np.full((1001, 3), np.nan)
            else:
                raise RuntimeError(f"simulation not found: {dir}, {j}")
            # solve_times_i.append(np.mean(sim['info']['solve_times']))
            z[control].append(z_j)

    plot.rmse_and_violations_MC(z, z_target, taskParams, save_dir=split(sim_save_dir)[0])


if __name__ == '__main__':
    # generate_task_params_obstacles()
    # doSimulationsRandomizeObstacles()
    plotResults()