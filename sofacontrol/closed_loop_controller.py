#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import Sofa
import Sofa.Simulation
import numpy as np
from scipy.sparse import csc_matrix

from sofacontrol import utils


class ClosedLoopController(Sofa.Core.Controller):
    """
    Controller interface with SOFA engine. onAnimateBeginEvent, onAnimateEndEvent and onKeypressedEvent are
    integrated within SOFA and the first two are called at the start and end of each timestep in the simulation
    respectively.

    Requires a controller, measurement model and output model to be specified in problem_specification.py
    """

    def __init__(self, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.rootNode = kwargs.get("rootNode")
        self.robot = kwargs.get("robot")
        self.preconditioner = self.robot.preconditioner
        self.opt = kwargs.get("opt", dict())  # for extra options
        self.simdata_dir = kwargs.get("simdata_dir")
        self.save_prefix = self.opt.get("save_prefix", datetime.now().strftime("%Y%m%d_%H%M"))
        self.dt = self.rootNode.dt.value

        #TODO: For now, store q. Include this as an argument later
        self.store_q = False
        self.store_x = True

        # Define controller, measurement model, output model, etc.
        self.controller = kwargs.get("controller")
        self.measurement = kwargs.get("measurement_model")
        self.output = kwargs.get("output_model")
        self.actuators = self.robot.actuator_list
        self.m = len(self.actuators)

        # Tell the controller what the sim time step is (for observer in particular)
        self.controller.set_sim_timestep(self.dt)

        # Initialize Variables
        self.t = None
        self.sim_data = {
            "dt": self.dt,
            "t": [],
            "z": [],
            "u": [],
            "z_hat": [],
            "q": [],
            "x": []
        }
        self.auto_paused = False

        # Optional arguments
        self.sim_duration = self.opt.get('sim_duration')

    # @catch_exceptions(ValueError, KeyError)
    def onAnimateBeginEvent(self, params):
        u_prev = self.get_command()
        x = utils.get_x(self.robot)
        y = self.measurement.evaluate(x)
        self.t = round(self.rootNode.time.value, 6)
        u = self.controller.evaluate(self.t, y, x, u_prev)
        self.apply_command(u)

        if self.simdata_dir is not None:
            self.sim_data["t"].append(self.t)
            self.sim_data["u"].append(self.get_command())
            self.sim_data["z"].append(self.output.evaluate(x))

            # Stores q and x data
            if self.store_q:
                q = utils.get_q(self.robot)
                self.sim_data["q"].append(q)
            if self.store_x:
                self.sim_data["x"].append(x)

            # Evaluate belief state of observer
            self.sim_data["z_hat"].append(self.controller.observer.z)

    def onAnimateEndEvent(self, params):
        if self.sim_duration is not None:
            if self.t > self.sim_duration and not self.auto_paused:
                print('Reached specified simulation duration.')
                self.rootNode.animate.value = False
                self.auto_paused = True
                self.rootNode.autopaused = True  # Terminates simulation when run from command line

                if self.simdata_dir is not None:
                    self.save_data()

    def onKeypressedEvent(self, c):
        key = c['key']
        if key == "S" and self.simdata_dir is not None:
            self.save_data()

    def apply_command(self, u):
        u *= self.dt
        for (i, actuator) in enumerate(self.actuators):
            actuator.value[0] = u[i]

    def get_command(self):
        u = []
        for (i, actuator) in enumerate(self.actuators):
            u.append(actuator.value[0])
        u = np.array(u) / self.dt
        u = np.maximum(u, self.robot.min_force)

        return np.atleast_1d(u.squeeze())

    def save_data(self):
        print('Saving simulation data.')
        filename = os.path.join(self.simdata_dir, self.save_prefix + '_sim.pkl')
        save_data = self.sim_data.copy()

        save_data["t"] = np.asarray(save_data["t"])
        save_data["u"] = np.asarray(save_data["u"])
        save_data["z"] = np.asarray(save_data["z"])
        save_data["z_hat"] = np.asarray(save_data["z_hat"])
        save_data["Hf"] = csc_matrix(self.output.C)
        if not 'info' in self.sim_data:
            save_data['info'] = self.controller.save_controller_info()
            try:
                save_data['info']['robot'] = self.opt.get('robot_name')
                save_data['info']['target_file'] = self.opt.get('target_file')
                save_data['info']['rom_file'] = self.opt.get('rom_file')
            except AttributeError:
                pass
        utils.save_data(filename, save_data)
        print('Done.')


class TemplateController:
    """
    Minimal functionality required for controller class to interface with closed_loop_controller
    """

    def __init__(self):
        pass

    def save_controller_info(self):
        """
        Enables saving controller info from the simulation (e.g. cost matrix values, dimension of system, observer info)
        :return: dict with controller info
        """
        info = dict()
        return info

    def evaluate(self, time, y, x, u_prev):
        """
        :param time: Timestep of the simulation (= total time elapsed)
        :param y: Measurement of system, as defined by measurement model
        :param x: Full state of the full order system
        :param u_prev: Previous input actually applied to system (considering actuator limits)
        :return: Input u to the system
        """
        raise NotImplementedError('TemplateController must be subclassed')

    def set_sim_timestep(self, dt):
        """
        :param dt: simulation time step
        """
        self.sim_dt = dt
