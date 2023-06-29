import os
from datetime import datetime

import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import numpy as np
"""
Provides direct interface with SOFA and is either opened from sofa gui or run in the following command style:
$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so $REPO_ROOT/launch_sofa.py
Imports problem_specification to add specified robot (FEM model) and specific controller.
"""

def createScene(rootNode, prob=None):
    # Start building scene
    rootNode.addObject("RequiredPlugin", name="SoftRobots")
    rootNode.addObject("RequiredPlugin", name="SofaSparseSolver")
    rootNode.addObject("RequiredPlugin", name="SofaPreconditioner")
    rootNode.addObject('RequiredPlugin', pluginName='SofaOpenglVisual')
    rootNode.addObject("VisualStyle", displayFlags="showBehavior")
    rootNode.addObject('RequiredPlugin', pluginName='SofaMatrix')

    rootNode.addObject("FreeMotionAnimationLoop")
    # rootNode.addObject('DefaultAnimationLoop', name='loop')
    rootNode.addObject("GenericConstraintSolver", maxIterations=100, tolerance=1e-5)

    rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
    rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")

    # Define the specific instance via the problem_specification script
    if prob is None:
        import problem_specification
        prob = problem_specification.problem()
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


def main(prob):
    print("Running launch_sofa.main()...")
    #  Allows executing from terminal directly
    #  Requires adjusting to own path
    sofa_lib_path = "/home/jalora/sofa/build/lib"
    # sofa_lib_path = "/home/jonas/Projects/stanford/sofa/build/lib"
    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    global _runAsPythonScript
    _runAsPythonScript = True
    root = Sofa.Core.Node()

    rootNode = createScene(root, prob)
    Sofa.Simulation.init(root)

    while True:
        Sofa.Simulation.animate(root, root.dt.value)
        if rootNode.autopaused == True:
            print('Simulation finished, exiting...')
            break


if __name__ == '__main__':
    main(None)
