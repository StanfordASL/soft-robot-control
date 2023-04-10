import os
from datetime import datetime

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


"""
Provides direct interface with SOFA and is either opened from sofa gui or run in the following command style:
$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py
Imports problem_specification to add specified robot (FEM model) and specific controller.

Automatically:
    - collects decay trajectories based on a given inpute sequence (see arguments of main())
    - saves the collected trajectories to "./snapshots/decayData/"
"""

def createScene(rootNode, q0=None, save_filepath="", input=np.ones(4)*1000, pre_tensioning=np.array([900, 1200, 0, 0])):
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

    from examples.diamond import diamond

    if save_filepath:
        save_data = True
    else:
        save_data = False
    prob = diamond.apply_constant_input(input, pre_tensioning, q0=q0, save_data=save_data, filepath=save_filepath)
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
    path = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.join(path, "examples/diamond/dataCollection/controlled") # origin"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    # global _runAsPythonScript
    # _runAsPythonScript = True
    pre_tensioning = np.zeros(4)
    # pre_tensioning = np.array([900, 1200, 0, 0])
    u_max = 3000
    u_dim = 4
    # inputs = []
    # for u in u_max:
    #     inputs += [np.array(i) for i in itertools.product([0, u], repeat=u_dim)][1:]
    inputs = [np.array(i) for i in itertools.product([0, 1500, 3000], repeat=u_dim)][1:]
    # # sample {nTraj} random inputs from the hyper-rectangle [0, u_max]^u_dim
    # for i in range(nTraj-1):
    #     u_sample = np.random.uniform(0.8, 1.0, size=u_dim) * u_max
    #     inputs.append(u_sample)

    # print(inputs)

    # Simulate different modes, amplitudes, and directions
    print(f"Simulating and saving {len(inputs)} different decay trajectories with pre-tensioning {pre_tensioning}")
    keep_trajs_up_to = 0
    for i, input in enumerate(tqdm(inputs)):
        
        save_filepath = f"{save_dir}/decayTraj_{i+keep_trajs_up_to:02d}"

        root = Sofa.Core.Node()
        rootNode = createScene(root, q0=None, save_filepath=save_filepath, input=input, pre_tensioning=pre_tensioning)
        Sofa.Simulation.init(root)
        
        while True:
            Sofa.Simulation.animate(root, root.dt.value)
            if rootNode.autopaused == True:
                break

    print('Simulation finished, exiting...')


if __name__ == '__main__':
    collectDecayTrajectories()