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

from psutil import virtual_memory
import sys
from examples.trunk.trunk_adiabaticSSM import run_scp, run_gusto_solver
from multiprocessing import Process

path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path, "settings.yaml"), "rb") as f:
    SETTINGS = yaml.safe_load(f)['collectDecayData']


def createScene_OL(rootNode, q0=None, save_filepath="", input=None, pre_tensioning=None):
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

    # Load settings
    save_dir = SETTINGS['save_dir']

    u_dim = SETTINGS['u_dim']
    n_trajs = SETTINGS['n_traj']

    # Create save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 
    # If pre_tensionings have been sampled before, load them, otherwise sample them randomly and save them for future runs
    if os.path.exists(os.path.join(save_dir, "pre_tensionings.pkl")):
        with open(os.path.join(save_dir, "pre_tensionings.pkl"), "rb") as f:
            pre_tensionings = pickle.load(f)
            
    else:
        if SETTINGS['find_pre_tensionings'] == "grid":
            # Find the right pre_tensionings by applying closed-loop controller
            pre_tensionings = []
            x = np.linspace(SETTINGS['x_range'][0], SETTINGS['x_range'][1], SETTINGS['n_grid']['x'])
            y = np.linspace(SETTINGS['y_range'][0], SETTINGS['y_range'][1], SETTINGS['n_grid']['y'])
            z = np.linspace(SETTINGS['z_range'][0], SETTINGS['z_range'][1], SETTINGS['n_grid']['z'])
            N = 300
            t = np.linspace(0, N*0.01, N)
            i = 0
            for zi in z:
                for yi in y:
                    for xi in x:
                        # If pre-tensioning exists
                        if os.path.exists(os.path.join(save_dir, "tmp_pretensionings", f"pre_tensioning_{i:03}.pkl")):
                            # Load pre-tensioning
                            with open(os.path.join(save_dir, "tmp_pretensionings", f"pre_tensioning_{i:03}.pkl"), "rb") as f:
                                pre_tensioning = pickle.load(f)
                        else:
                            zi_constr = -(195 - np.sqrt(195**2 - xi**2 - yi**2)) + zi
                            z = np.tile(np.hstack([xi, yi, zi_constr]), (N, 1))
                            # ramp inputs to make trajectory more well-behaved
                            z *= np.concatenate([np.linspace(0, 1, N//2), np.ones(N//2)])[:, None]

                            print(z)
                            # start process running gusto_solver with argument z
                            p_gusto = Process(target=run_gusto_solver, args=(t, z,))
                            p_gusto.start()
                            # start SOFA simulation
                            root = Sofa.Core.Node()
                            rootNode = createScene_CL(root, z=z, T=1+N*0.01)
                            Sofa.Simulation.init(root)
                            while True:
                                Sofa.Simulation.animate(root, root.dt.value)
                                if rootNode.autopaused == True:
                                    break  # Terminates simulation when autopaused
                            # terminate process running gusto_solver
                            p_gusto.terminate()
                            p_gusto.join()
                            p_gusto.close()
                            # Load the resulting trajectory data
                            with open(os.path.join("/home/jonas/Projects/stanford/soft-robot-control/examples/trunk", "ssmr_sim.pkl"), "rb") as f:
                                # Extract the pre-tensioning
                                sim_data = pickle.load(f)
                            pre_tensioning = sim_data['u'][-1]
                            
                            assert len(pre_tensioning) == u_dim
                            # Save the pre-tensioning
                            with open(os.path.join(save_dir, "tmp_pretensionings", f"pre_tensioning_{i:03}.pkl"), "wb") as f:
                                pickle.dump(pre_tensioning, f)
                            os.execv(sys.executable, ['python'] + sys.argv)
                        i += 1
                        print(f"Pre-tensioning at ({xi}, {yi}, {zi}): {pre_tensioning}")
                        pre_tensionings.append(pre_tensioning.astype(float))
        
        elif SETTINGS['find_pre_tensionings'] == "random":
            # Sample pre-tensionings randomly
            combine_pre_tensionings = SETTINGS['combine_pre_tensionings']
            n_samples = SETTINGS['n_samples']
            rng = np.random.default_rng(seed=42)
            pre_tensionings = [np.zeros(u_dim)]
            while len(pre_tensionings) < n_samples:
                probs = np.concatenate([[SETTINGS['sparsity']], np.ones(len(combine_pre_tensionings) - 1) * (1 - SETTINGS['sparsity']) / (len(combine_pre_tensionings) - 1)])
                sampled_pre_tensioning = np.random.choice(combine_pre_tensionings, size=u_dim, p=probs)
                if not np.any([np.allclose(pre_tensioning, sampled_pre_tensioning) for pre_tensioning in pre_tensionings]):
                    pre_tensionings.append(sampled_pre_tensioning.astype(float))

        elif SETTINGS['find_pre_tensionings'] == "specified":
            # Use pre-tensionings specified in settings.yaml
            pre_tensionings = np.array(SETTINGS['pre_tensionings'])

        else:
            raise ValueError(f"Invalid value for find_pre_tensionings: {SETTINGS['find_pre_tensionings']}")

        # Save the pre-tensionings for future runs
        with open(os.path.join(save_dir, "pre_tensionings.pkl"), "wb") as f:
            pickle.dump(pre_tensionings, f)
    
    # Do one short simulation per pre-tensioning to collect rest configurations
    if os.path.exists(os.path.join(save_dir, "rest_qs.pkl")):
        with open(os.path.join(save_dir, "rest_qs.pkl"), "rb") as f:
            rest_qs = pickle.load(f)
    else:
        print("========= Collecting rest configurations =========")
        rest_qs = []
        for pre_tensioning in tqdm(pre_tensionings):
            root = Sofa.Core.Node()
            print(pre_tensioning)
            rootNode = createScene_OL(root, q0=None, save_filepath=os.path.join(save_dir, "temp_rest_traj"), input=np.zeros(u_dim), pre_tensioning=pre_tensioning)
            Sofa.Simulation.init(root)
            while True:
                Sofa.Simulation.animate(root, root.dt.value)
                if rootNode.autopaused == True:
                    break
            with open(os.path.join(save_dir, "temp_rest_traj_snapshots.pkl"), "rb") as f:
                rest_q = pickle.load(f)['q'][-1]
            print("rest_q:", rest_q)
            rest_qs.append(rest_q)
        with open(os.path.join(save_dir, "rest_qs.pkl"), "wb") as f:
            pickle.dump(rest_qs, f)
        # assert len(rest_qs) == n_samples

    # Do n_trajs simulations per pre-tensioning to collect decay trajectories
    for i, pre_tensioning in enumerate(tqdm(pre_tensionings)):
        model_dir = os.path.join(save_dir, f"{i:03}/decay/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(os.path.join(model_dir, "..", "pre_tensioning.pkl"), "wb") as f:
            pickle.dump(pre_tensioning, f)
        with open(os.path.join(model_dir, "..", "rest_q.pkl"), "wb") as f:
            pickle.dump(rest_qs[i], f)
        combine_inputs = SETTINGS['combine_inputs']
        # If inputs have been sampled before, load them, otherwise sample them randomly and save them for future runs
        if os.path.exists(os.path.join(model_dir, "inputs.pkl")):
            with open(os.path.join(model_dir, "inputs.pkl"), "rb") as f:
                inputs = pickle.load(f)
        else:
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
            rootNode = createScene_OL(root, q0=None, save_filepath=save_filepath, input=input, pre_tensioning=pre_tensioning)
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