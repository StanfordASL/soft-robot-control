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
import scipy.io as spio
from itertools import combinations, permutations, product
from tqdm.auto import tqdm
from pathlib import Path

import numpy as np
import itertools
import yaml
import pickle

from psutil import virtual_memory
import sys

path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path, "settings.yaml"), "rb") as f:
    SETTINGS = yaml.safe_load(f)['collectDecayData']


def createScene(rootNode, q0=None, save_filepath="", input=None, pre_tensioning=None):
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

    if SETTINGS['robot'] == 'diamond':
        from examples.diamond import diamond as platform
    elif SETTINGS['robot'] == 'trunk':
        from examples.trunk import trunk as platform
    else:
        raise RuntimeError("Please specify a valid platform to be used in settings.yaml / ['diamond', 'trunk']")
    
    if save_filepath:
        save_data = True
    else:
        save_data = False
    
    prob = platform.apply_constant_input(input, pre_tensioning, q0=q0, save_data=save_data, filepath=save_filepath)
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


def collectDecayTrajectories():
    #  Allows executing from terminal directly
    #  Requires adjusting to own path
    sofa_lib_path = "/home/jonas/Projects/stanford/sofa/build/lib"

    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    save_dir = SETTINGS['save_dir']
    max_pre_tensioning = SETTINGS['max_pre_tensioning']
    n_grid = SETTINGS['n_grid']
    combine_pre_tensionings = np.linspace(-max_pre_tensioning, max_pre_tensioning, n_grid)
    n_samples = SETTINGS['n_samples']
    u_dim = SETTINGS['u_dim']
    n_trajs = SETTINGS['n_traj']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(os.path.join(save_dir, "pre_tensionings.pkl")):
        with open(os.path.join(save_dir, "pre_tensionings.pkl"), "rb") as f:
            pre_tensionings = pickle.load(f)
    else:
        rng = np.random.default_rng(seed=42)
        pre_tensionings = [np.zeros(u_dim)]
        while len(pre_tensionings) < n_samples:
            sampled_pre_tensioning = np.random.choice(combine_pre_tensionings, size=u_dim)
            if not np.any([np.allclose(pre_tensioning, sampled_pre_tensioning) for pre_tensioning in pre_tensionings]):
                pre_tensionings.append(sampled_pre_tensioning.astype(float))
        with open(os.path.join(save_dir, "pre_tensionings.pkl"), "wb") as f:
            pickle.dump(pre_tensionings, f)
    assert len(pre_tensionings) == n_samples
    
    for i, pre_tensioning in enumerate(tqdm(pre_tensionings)):
        model_dir = os.path.join(save_dir, f"{i:03}/decay/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(os.path.join(model_dir, "pre_tensioning.pkl"), "wb") as f:
            pickle.dump(pre_tensioning, f)
        combine_inputs = SETTINGS['combine_inputs']
        if os.path.exists(os.path.join(model_dir, "inputs.pkl")):
            with open(os.path.join(model_dir, "inputs.pkl"), "rb") as f:
                inputs = pickle.load(f)
        else:
            rng = np.random.default_rng(seed=42)
            inputs = []
            while len(inputs) < n_trajs:
                sampled_input = np.random.choice(combine_inputs, size=u_dim)
                sampled_input = np.clip(sampled_input, SETTINGS['u_min'], SETTINGS['u_max'])
                if not np.any([np.allclose(input, sampled_input) for input in inputs]) and np.any(sampled_input > 0):
                    inputs.append(sampled_input.astype(float))
            with open(os.path.join(model_dir, "inputs.pkl"), "wb") as f:
                pickle.dump(inputs, f)
        already_simulated_files = sorted([fname for fname in os.listdir(model_dir) if "decayTraj" in fname])
        if already_simulated_files:
            collect_from = int(already_simulated_files[-1].split('_')[1]) + 1
            print("Collecting data starting at trajectory index:", collect_from)
            inputs = inputs[collect_from:]
        else:
            collect_from = 0
        # Simulate different amplitudes and directions
        print(f"Simulating and saving {len(inputs)} different decay trajectories with pre-tensioning {pre_tensioning}")
        for j, input in enumerate(tqdm(inputs)):
            save_filepath = f"{model_dir}/decayTraj_{j+collect_from:02d}"

            root = Sofa.Core.Node()
            rootNode = createScene(root, q0=None, save_filepath=save_filepath, input=input, pre_tensioning=pre_tensioning)
            Sofa.Simulation.init(root)
            
            while True:
                Sofa.Simulation.animate(root, root.dt.value)
                if rootNode.autopaused == True:
                    break    
            # Restart if RAM usage is too high (there is a memory leak somewhere, can't find it)
            ram_usage_percent = virtual_memory()[2]
            print("RAM usage:", ram_usage_percent)
            if ram_usage_percent > 80:
                os.execv(sys.executable, ['python'] + sys.argv)

    print('All simulations finished, exiting...')


if __name__ == '__main__':
    collectDecayTrajectories()