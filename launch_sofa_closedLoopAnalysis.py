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

import plotting as plot

path = dirname(os.path.abspath(__file__))

# SETTINGS
MODEL_DIR = "/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms_N=100_sparsity=0.95"
ADD_MODELS_PER_STEP = 10
SIMS_PER_MODEL = 5
ORIGIN_MODEL_IDX = 0

METHOD = "adiabatic_ssm" # "tpwl" # "koopman" # "ssm"
SAVE_DIR = join("/media/jonas/Backup Plus/jonas_soft_robot_data/trunk_closed-loop_analysis", METHOD)


def createScene_CL(rootNode, z, T):
    # Start building scene
    rootNode.addObject("RequiredPlugin", name="SoftRobots", printLog=False)
    rootNode.addObject("RequiredPlugin", name="SofaPython3", printLog=False)
    rootNode.addObject("RequiredPlugin", name="SofaSparseSolver")
    rootNode.addObject("RequiredPlugin", name="SofaPreconditioner")
    rootNode.addObject('RequiredPlugin', pluginName='SofaOpenglVisual')

    rootNode.addObject('RequiredPlugin', pluginName='SofaMatrix')

    rootNode.addObject("VisualStyle", displayFlags="showBehavior")
    rootNode.addObject("FreeMotionAnimationLoop")
    # rootNode.addObject('DefaultAnimationLoop', name='loop')
    rootNode.addObject("GenericConstraintSolver", maxIterations=100, tolerance=1e-5)
    rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
    rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
    rootNode.addObject('DefaultVisualManagerLoop')

    if METHOD == "koopman":
        from examples.trunk.trunk_koopman import run_koopman as run_scp
    elif METHOD == "tpwl":
        from examples.trunk.trunk_tpwl import run_scp
    elif METHOD == "adiabatic_ssm":
        from examples.trunk.trunk_adiabaticSSM import run_scp
    elif METHOD == "ssm":
        from examples.trunk.trunk_SSM import run_scp
    else:
        raise ValueError("Invalid method")

    prob = run_scp(z, T)
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


def doSimulations(method=METHOD):
    #  Allows executing from terminal directly
    #  Requires adjusting to own path
    sofa_lib_path = "/home/jonas/Projects/stanford/sofa/build/lib"

    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    model_names = [name for name in sorted(listdir(MODEL_DIR)) if isdir(join(MODEL_DIR, name))]
    # model_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 36, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 51, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 76, 77, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99]
    # model_names = [f"{i:03d}" for i in model_names]

    assert ORIGIN_MODEL_IDX == 0

    np.random.seed(0)

    # create a flag to indicate if the script failed
    with open("/home/jonas/Projects/stanford/soft-robot-control/gusto_failed_flag.pkl", "wb") as f:
        pickle.dump(False, f)
    
    if method == "adiabatic_ssm":


        for i in tqdm(range(0, len(model_names) + 1, ADD_MODELS_PER_STEP)): # range(ADD_MODELS_PER_STEP, len(model_names) + 1, ADD_MODELS_PER_STEP)
            for j in tqdm(range(SIMS_PER_MODEL)):
                # add the next {add_models_per_step} models to use_models
                use_models = np.random.choice([int(name) for name in model_names], size=i, replace=False).tolist()
                # save use_models as .pkl file
                save_dir_i = join(SAVE_DIR, f"{i:03d}")
                if exists(join(save_dir_i, f"ssmr_sim_{j}.pkl")) or exists(join(save_dir_i, f"ssmr_sim_{j}_failed.pkl")):
                    continue
                # Create save directory if it does not exist
                if not os.path.exists(save_dir_i):
                    os.makedirs(save_dir_i)
                else:
                    with open(os.path.join(save_dir_i, f"use_models_{j}.pkl"), "wb") as f:
                        pickle.dump(use_models, f)
                    try:
                        with open(os.path.join(MODEL_DIR, "use_models.pkl"), "wb") as f:
                            pickle.dump(use_models, f)
                        # import trunk_adiabaticSSM here to create the interpolated model with the currently used models
                        from examples.trunk.trunk_adiabaticSSM import run_gusto_solver
                        # run closed loop simulation with an adiabatic model interpolating the models in use_models
                        # start process running gusto_solver with argument z
                        p_gusto = Process(target=run_gusto_solver, args=(None, None,))
                        p_gusto.start()
                        # start SOFA simulation
                        root = Sofa.Core.Node()
                        rootNode = createScene_CL(root, z=None, T=11.)
                        Sofa.Simulation.init(root)
                        while True:
                            with open("/home/jonas/Projects/stanford/soft-robot-control/gusto_failed_flag.pkl", "rb") as f:
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
                        # copy the resulting trajectory data to save_dir
                        copy(os.path.join("/home/jonas/Projects/stanford/soft-robot-control/examples/trunk", "ssmr_sim.pkl"), join(save_dir_i, f"ssmr_sim_{j}.pkl"))
                        # save use_models as .pkl file

                    except Exception as e:
                        # closed_loop_simulation failed
                        print(e)
                        with open(os.path.join(save_dir_i, f"ssmr_sim_{j}_failed.pkl"), "wb") as f:
                            pickle.dump({}, f)

                    os.execv(sys.executable, ['python'] + sys.argv)

    elif method == "koopman":
        pass
    elif method == "tpwl":
        pass
    elif method == "ssm":
        pass

    print('All simulations finished, exiting...')


def plotResults():
    z = {}
    # solve_times = {}
    n_models = {}
    for control in ["qp"]: # ["idw", "nn", "qp"]: # "ct", 
        z[control] = []
        # solve_times[control] = []
        n_models[control] = []
        result_dirs = [name for name in sorted(listdir(SAVE_DIR + f"_{control}")) if isdir(join(SAVE_DIR + f"_{control}", name))]
        t0 = 1
        for dir in result_dirs:
            z_i = []
            # solve_times_i = []
            n_models_i = []
            for j in range(SIMS_PER_MODEL):
                print(control, dir, j)
                with open(join(SAVE_DIR + f"_{control}", dir, f"use_models_{j}.pkl"), "rb") as f:
                    use_models = pickle.load(f)
                if exists(join(SAVE_DIR + f"_{control}", dir, f"ssmr_sim_{j}.pkl")):
                    with open(join(SAVE_DIR + f"_{control}", dir, f"ssmr_sim_{j}.pkl"), "rb") as f:
                        sim = pickle.load(f)
                    idx = np.argwhere(sim['t'] > t0)[0][0]
                    z_ij = sim['z'][idx:, 3:]
                    z_ij[:, 2] *= -1
                elif exists(join(SAVE_DIR + f"_{control}", dir, f"ssmr_sim_{j}_failed.pkl")):
                    print("simulation failed: ", dir, j)
                    z_ij = np.full((1001, 3), np.nan)
                else:
                    raise RuntimeError(f"simulation not found: {dir}, {j}")
                z_i.append(z_ij)
                n_models_i.append(len(use_models))
                # solve_times_i.append(np.mean(sim['info']['solve_times']))
            z[control].append(z_i)
            n_models[control].append(n_models_i[0])
            # solve_times[control].append(solve_times_i)
    plot.rmse_vs_n_models(n_models, z, baselines=True, save_dir=split(SAVE_DIR)[0])


if __name__ == '__main__':
    # for method in ["adiabatic_ssm"]:
    #     doSimulations(method)
    plotResults()
