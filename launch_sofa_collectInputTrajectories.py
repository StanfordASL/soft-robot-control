"""
Provides direct interface with SOFA and is either opened from sofa gui or run in the following command style:
$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py
Imports problem_specification to add specified robot (FEM model) and specific controller.

Automatically:
    - collects open-loop controlled trajectories based on a random LHS (latin hypercube sampling) input sequence
    - saves the collected trajectories to the directory specified in settings.yaml
"""
import os
from datetime import datetime

import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import numpy as np
from itertools import combinations, permutations, product
from tqdm.auto import tqdm
from pathlib import Path

import numpy as np
import yaml

path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path, "settings.yaml"), "rb") as f:
    SETTINGS = yaml.safe_load(f)['collectInputData']


def createScene(rootNode, q0=None, save_filepath=""):
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
    prob = platform.collect_open_loop_data(u_max=SETTINGS['u_max'], pre_tensioning=SETTINGS['pre_tensioning'], q0=q0, save_data=save_data, filepath=save_filepath)
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


def collectInputTrajectories():
    #  Allows executing from terminal directly
    #  Requires adjusting to own path
    sofa_lib_path = "/home/jonas/Projects/stanford/sofa/build/lib"
    path = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.join(path, SETTINGS['save_dir']) # origin"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    print(f"Simulating open-loop input trajectories")
        
    save_filepath = f"{save_dir}/controlled_traj"

    root = Sofa.Core.Node()
    rootNode = createScene(root, q0=None, save_filepath=save_filepath)
    Sofa.Simulation.init(root)
    
    while True:
        Sofa.Simulation.animate(root, root.dt.value)
        if rootNode.autopaused == True:
            break
    print('Simulation finished, exiting...')


if __name__ == '__main__':
    collectInputTrajectories()