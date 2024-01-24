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


"""
Provides direct interface with SOFA and is either opened from sofa gui or run in the following command style:
$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py
Imports problem_specification to add specified robot (FEM model) and specific controller.
"""

robotType = "diamond"

def createDataCollectionScene(rootNode, q0, save_filename, amplitude):
    # Start building scene
    rootNode.addObject("RequiredPlugin", name="SoftRobots")
    rootNode.addObject("RequiredPlugin", name="SofaSparseSolver")
    rootNode.addObject("RequiredPlugin", name="SofaPreconditioner")
    rootNode.addObject('RequiredPlugin', pluginName='SofaOpenglVisual')
    rootNode.addObject("VisualStyle", displayFlags="showBehavior")

    rootNode.addObject("FreeMotionAnimationLoop")
    # rootNode.addObject('DefaultAnimationLoop', name='loop')
    rootNode.addObject("GenericConstraintSolver", maxIterations=100, tolerance=1e-5)

    rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
    rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
    rootNode.addObject('DefaultVisualManagerLoop')

    input_const = np.array([800., 0., 0., 0., 0., 0., 0., 0.])

    import problem_specification

    # Rotation in degrees
    prob = problem_specification.problem(input=input_const)
    #prob = problem_specification.problem()
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
                                            opt=prob.opt)
                                            )
    rootNode.autopaused = False  # Enables terminating simulation at the end when running from command line
    return rootNode

def createScene(rootNode):
    # Start building scene
    rootNode.addObject("RequiredPlugin", name="SoftRobots")
    rootNode.addObject("RequiredPlugin", name="SofaSparseSolver")
    rootNode.addObject("RequiredPlugin", name="SofaPreconditioner")
    rootNode.addObject('RequiredPlugin', pluginName='SofaOpenglVisual')
    rootNode.addObject('RequiredPlugin', pluginName='SofaMatrix')
    rootNode.addObject("VisualStyle", displayFlags="showBehavior")

    rootNode.addObject("FreeMotionAnimationLoop")
    # rootNode.addObject('DefaultAnimationLoop', name='loop')
    rootNode.addObject("GenericConstraintSolver", maxIterations=100, tolerance=1e-5)

    # rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
    rootNode.addObject('BackgroundSetting', color='1 1 1')

    rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")

    # Define the specific instance via the problem_specification script
    path = os.path.dirname(os.path.abspath(__file__))
    import problem_specification

    if robotType == "trunk":
        # Order of control: 1 - front, clockwise (full extension); 5 - front, clockwise (midpoint)
        amp1 = 400.
        amp2 = 800.
        # input_const = np.array([0, 0, 0, amp1, 0, amp2, 0, 0])
        input_const = np.array([amp2, 0, 0, 0, 0, 0, 0, 0])
    else:
        amplitude = 3000.
        input_const = np.array([0., amplitude, amplitude, 0.])

    prob = problem_specification.problem(input=input_const, save_data=False)
    # prob = problem_specification.problem(q0=q0, save_data=False, scale_mode=amplitude, filename=save_filename)
    #prob = problem_specification.problem()
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
                                            opt=prob.opt)
                                            )
    rootNode.autopaused = False  # Enables terminating simulation at the end when running from command line
    return rootNode

def main():
    #  Allows executing from terminal directly
    #  Requires adjusting to own path
    #sofa_lib_path = "/home/jjalora/sofa/build/lib"
    sofa_lib_path = "/home/jalora/sofa/build/lib"


    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    global _runAsPythonScript
    _runAsPythonScript = True


    root = Sofa.Core.Node()
    rootNode = createScene(root)
    Sofa.Simulation.init(root)

    while True:
        Sofa.Simulation.animate(root, root.dt.value)
        if rootNode.autopaused == True:
            break

    print('Simulation finished, exiting...')


if __name__ == '__main__':
    main()