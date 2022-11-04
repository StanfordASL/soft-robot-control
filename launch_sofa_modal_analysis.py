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

    # Define the specific instance via the problem_specification script
    # Run modal analysis or collect decaying trajectories
    # input_const = np.array([1500., 1500., 0., 0.])

    import problem_specification

    # Rotation in degrees
    # prob = problem_specification.problem(input=input_const)
    prob = problem_specification.problem(q0=q0, save_data=True, scale_mode=amplitude, filename=save_filename)
    # prob = problem_specification.problem()
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

    rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
    rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")

    # Define the specific instance via the problem_specification script
    # Run modal analysis or collect decaying trajectories
    # input_const = np.array([1., 0., 0., 0.])
    # input_const = np.array([0., 1., 0., 0.])
    # input_const = np.array([0., 2000., 0., 2000.])
    path = os.path.dirname(os.path.abspath(__file__))

    modeName1 = 'mode3'
    datafile1 = path + '/robots/data/' + modeName1 + '.mat'

    modeName2 = 'mode2'
    datafile2 = path + '/robots/data/' + modeName2 + '.mat'

    # TODO: Setting directions modal displacements
    # Set corresponding direction to zero if desiring just one mode (Takes int values 0, 1, or -1)
    direction1 = 0
    direction2 = 0

    amplitude = 2000

    modal_direction = ["", "pos", "neg"]
    signMode1 = modal_direction[direction1]
    signMode2 = modal_direction[direction2]

    if signMode1 == "":
        save_filename = signMode2 + modeName2 + "_" + str(amplitude)
        print("Now simulating " + modeName2 + " with " + str(amplitude) + " amplitude")
    elif signMode2 == "":
        save_filename = signMode1 + modeName1 + "_" + str(amplitude)
        print("Now simulating " + modeName1 + " with " + str(amplitude) + " amplitude")
    else:
        save_filename = signMode1 + modeName1 + "_" + signMode2 + modeName2 + "_" + str(amplitude)
        print("Now simulating " + modeName1 + " and " + modeName2 + " with " + str(amplitude) + " amplitude")
    # Single trajectory
    q0 = direction1 * spio.loadmat(datafile1)[modeName1].T + direction2 * spio.loadmat(datafile2)[modeName2].T

    import problem_specification

    # Rotation in degrees
    # TODO: To save data, need to specify file
    input_const = np.array([amplitude, 0, 0, 0])
    prob = problem_specification.problem(q0=q0, input=input_const, save_data=True, filename=save_filename)
    # prob = problem_specification.problem(q0=q0, save_data=False, scale_mode=amplitude, filename=save_filename)
    # prob = problem_specification.problem()
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


def main(sofa_lib_path=None):
    #  Allows executing from terminal directly
    #  Requires adjusting to own path
    # sofa_lib_path = "/home/jjalora/sofa/build/lib"
    # sofa_lib_path = "/home/jalora/sofa/build/lib"
    sofa_lib_path = "/home/jason/sofa/build/v22.06/lib"
    runSingleSimulation = True

    if not os.path.exists(sofa_lib_path):
        raise RuntimeError('Path non-existent, sofa_lib_path should be modified to point to local SOFA installation'
                           'in main() in launch_sofa.py')

    SofaRuntime.PluginRepository.addFirstPath(sofa_lib_path)
    SofaRuntime.importPlugin("SofaComponentAll")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    global _runAsPythonScript
    _runAsPythonScript = True

    # modesList = list(combinations(["mode2", "mode3"], 2))
    modesList = list(combinations(["mode1", "mode2", "mode3"], 3))
    amplitudeList = [1500, 2000]
    # amplitudeList = [1000]
    directionList = list(filter(lambda elem: elem != (0, 0, 0), [p for p in product([1, 0, -1], repeat=3)]))

    if runSingleSimulation:
        root = Sofa.Core.Node()
        rootNode = createScene(root)
        Sofa.Simulation.init(root)

        while True:
            Sofa.Simulation.animate(root, root.dt.value)
            if rootNode.autopaused == True:
                break
    else:
        # Simulate different modes, amplitudes, and directions
        for (modes, directions, amplitude) in tqdm(product(modesList, directionList, amplitudeList)):
            root = Sofa.Core.Node()

            path = os.path.dirname(os.path.abspath(__file__))

            modeName1 = modes[0]
            datafile1 = path + '/robots/data/' + modeName1 + '.mat'

            modeName2 = modes[1]
            datafile2 = path + '/robots/data/' + modeName2 + '.mat'

            modeName3 = modes[2]
            datafile3 = path + '/robots/data/' + modeName3 + '.mat'

            # Set corresponding direction to zero if desiring just one mode (Takes int values 0, 1, or -1)
            direction1 = directions[0]
            direction2 = directions[1]
            direction3 = directions[2]

            modal_direction = ["", "pos", "neg"]
            signMode1 = modal_direction[direction1]
            signMode2 = modal_direction[direction2]
            signMode3 = modal_direction[direction3]

            if signMode1 == "":
                if signMode3 == "":
                    save_filename = signMode2 + modeName2 + "_" + str(amplitude)
                    print("Now simulating " + modeName2 + " with " + str(amplitude) + " amplitude")
                elif signMode2 == "":
                    save_filename = signMode3 + modeName3 + "_" + str(amplitude)
                    print("Now simulating " + modeName3 + " with " + str(amplitude) + " amplitude")
                else:
                    save_filename = signMode2 + modeName2 + "_" + signMode3 + modeName3 + "_" + str(amplitude)
                    print(
                        "Now simulating " + modeName2 + " and " + modeName3 + " with " + str(amplitude) + " amplitude")
            elif signMode2 == "":
                if signMode3 == "":
                    save_filename = signMode1 + modeName1 + "_" + str(amplitude)
                    print("Now simulating " + modeName1 + " with " + str(amplitude) + " amplitude")
                else:
                    save_filename = signMode1 + modeName1 + "_" + signMode3 + modeName3 + "_" + str(amplitude)
                    print(
                        "Now simulating " + modeName1 + " and " + modeName3 + " with " + str(amplitude) + " amplitude")
            elif signMode3 == "":
                save_filename = signMode1 + modeName1 + "_" + signMode2 + modeName2 + "_" + str(amplitude)
                print("Now simulating " + modeName1 + " and " + modeName2 + " with " + str(amplitude) + " amplitude")
            else:
                save_filename = signMode1 + modeName1 + "_" + signMode2 + modeName2 + "_" + signMode3 + modeName3 + "_" + str(
                    amplitude)
                print("Now simulating " + modeName1 + " and " + modeName2 + " and " + modeName3 + " with " + str(
                    amplitude) + " amplitude")

            # Single trajectory
            q0 = direction1 * spio.loadmat(datafile1)[modeName1].T + direction2 * spio.loadmat(datafile2)[
                modeName2].T + direction3 * spio.loadmat(datafile3)[modeName3].T

            currentModalFile = Path(path + "/examples/diamond/dataCollection/" + save_filename + '_snapshots' + '.pkl')
            if currentModalFile.exists():
                print(save_filename + " exists! Skipping to next initial condition")
                continue

            rootNode = createDataCollectionScene(root, q0, save_filename, amplitude)
            Sofa.Simulation.init(root)

            while True:
                Sofa.Simulation.animate(root, root.dt.value)
                if rootNode.autopaused == True:
                    break

    print('Simulation finished, exiting...')


if __name__ == '__main__':
    main()
