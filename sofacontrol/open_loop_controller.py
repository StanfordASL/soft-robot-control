#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import os
from datetime import datetime

import Sofa
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix

import sofacontrol.utils as scutils


class OpenLoopController(Sofa.Core.Controller):
    """
    Controller interface with SOFA engine. onAnimateBeginEvent, onAnimateEndEvent and onKeypressedEvent are
    integrated within SOFA and the first two are called at the start and end of each timestep in the simulation
    respectively.

    Handles saving full snapshot data, without any processing

    Requires a controller to be specified in problem_specification.py
    """

    def __init__(self, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.opt = kwargs.get("opt", dict())  # for extra options

        # Sofa related variables
        self.rootNode = kwargs.get("rootNode")
        self.dt = self.rootNode.dt.value
        self.robot = kwargs.get("robot")
        self.actuators = self.robot.actuator_list
        self.m = len(self.actuators)

        # Get open loop control sequence
        self.controller = kwargs.get('controller')

        self.simdata_dir = kwargs.get('simdata_dir')
        self.snapshots_dir = kwargs.get('snapshots_dir')

        # For old code so you can also just pass in the u and save tuple
        if isinstance(self.controller, tuple):
            self.controller = OpenLoop(self.m, self.controller[2], self.controller[0], self.controller[1])

        # Data saving variables
        self.save_prefix = self.opt.get("save_prefix", datetime.now().strftime("%Y%m%d_%H%M"))
        self.output = kwargs.get("output_model")
        self.sim_data = {
            "dt": self.dt,
            "t": [],
            "z": [],
            "u": []
        }
        self.snapshots = kwargs.get("snapshots")  # the object for storing snapshots

        if self.snapshots is not None:
            self.LDL_dir = scutils.get_snapshot_dir()

        # Other
        self.step = 0
        self.t = 0
        self.next_save_idx = 0
        self.auto_paused = False
        self.prev_point = None
        self.data_saved = False

    def onAnimateBeginEvent(self, params):
        self.t = round(self.rootNode.time.value, 6)

        # Define control
        u = self.controller.evaluate(self.t)
        self.apply_command(u)

        # Saving simulation data
        if self.simdata_dir is not None:
            self.sim_data_saving()

        # Check if snapshot should be saved
        self.save_point = False
        if self.snapshots is not None:
            # Always keep around both the previous and current points to help
            # evaluate whether the current point should be saved
            self.point = scutils.Point()
            self.point.t = self.t
            self.point.dt = self.dt
            self.point.q = self.robot.tetras.position.value.flatten().copy()
            self.point.v = self.robot.tetras.velocity.value.flatten().copy()
            # self.point.f = self.robot.tetras.force.value.flatten().copy() # Not currently used
            self.point.u = self.get_command()

            if self.save_snapshot() and self.snapshots.save_snapshot(self.point, self.prev_point):
                self.save_point = True
                if self.snapshots.save_dynamics:
                    scutils.turn_on_LDL_saver(self.robot.preconditioner,
                                              os.path.join(self.LDL_dir, 'temp/LDL_%05d.txt'))
                    self.point.H = scutils.extract_H_matrix(self.robot)

            self.prev_point = copy.copy(self.point)

    def onAnimateEndEvent(self, params):
        scutils.turn_off_LDL_saver(self.robot.preconditioner)

        # Gather remaining information for saving the current point
        if self.save_point:
            self.point.q_next = self.robot.tetras.position.value.flatten().copy()
            self.point.v_next = self.robot.tetras.velocity.value.flatten().copy()
            if self.snapshots.save_dynamics:
                dv = self.point.v_next - self.point.v
                K, D, M, b, f, S = scutils.extract_KDMb(self.robot, self.LDL_dir, self.step, params['dt'], dv,
                                                        self.point)
                self.point.K = K
                self.point.D = D
                self.point.M = M
                self.point.b = b
                self.point.f = f
                self.point.S = S
            self.snapshots.add_point(self.point)

        # Turn off animation at the end of the defined sequence
        if self.t >= self.controller.t_seq[-1] and not self.auto_paused:
            print('Reached the end of the sequence.')
            self.rootNode.animate.value = False
            self.auto_paused = True
            self.rootNode.autopaused = True  # Terminates simulation when run from command line

            # Let the snapshots object clean up
            if self.snapshots is not None:
                filename = os.path.join(self.snapshots_dir, self.save_prefix + '_snapshots.pkl')
                self.snapshots.simulation_end(filename)  # May add additional arguments to this later

        self.step += 1

    def sim_data_saving(self):
        """
        Saves simulation data at time points specified in self.controller.save_seq
        """
        if self.t <= self.controller.t_seq[-1]:
            self.sim_data["t"].append(self.t)
            self.sim_data["z"].append(self.output.evaluate(scutils.get_x(self.robot)))
        if self.t >= self.controller.t_seq[-1] and not self.data_saved:
            self.sim_data["u"] = np.atleast_2d(self.controller.u_seq.T)  # Transpose required to have N x n_u
            self.sim_data["t"] = np.asarray(self.sim_data["t"])
            self.sim_data["z"] = np.asarray(self.sim_data["z"])
            self.sim_data["Hf"] = csc_matrix(self.output.C)
            print('Saving simulation data.')
            filename = os.path.join(self.simdata_dir, self.save_prefix + '_sim.pkl')
            scutils.save_data(filename, self.sim_data)
            self.data_saved = True

    def save_snapshot(self):
        """
        Function that returns true if a snapshot should be taken at the current time
        """
        save = False
        if self.t <= self.controller.t_seq[-1]:
            t_next_save = self.controller.t_seq[self.next_save_idx]  # time of next saving
            if self.t >= round(t_next_save, 6):
                if self.controller.save_seq[self.next_save_idx]:
                    save = True
            self.next_save_idx += 1
        return save

    def apply_command(self, u):
        u = self.dt * np.atleast_1d(u)
        for (i, actuator) in enumerate(self.actuators):
            actuator.value[0] = u[i]

    def get_command(self):
        u = []
        for (i, actuator) in enumerate(self.actuators):
            u.append(actuator.value[0])
        u = np.array(u) / self.dt
        u = np.maximum(u, self.robot.min_force)

        return np.atleast_1d(u.squeeze())


class OpenLoop:
    """
    Minimal functionality required for controller class to interface with open_loop_controller
    :t_sequence: array of times corresponding to u and save sequences
    :u_sequence: 
    """

    def __init__(self, m, t_sequence, u_sequence, save_sequence):
        self.m = m
        self.t_seq = t_sequence
        self.save_seq = save_sequence
        self.u_seq = self.convert_u_standard_form(u_sequence)
        self.u_interp = interp1d(self.t_seq, self.u_seq)

    def save_controller_info(self):
        """
        This function is only called if this class is used for closed_loop_controller
        """
        return {'m': self.m, 't': self.t_seq, 'u': self.u_seq, 'save': self.save_seq}

    def evaluate(self, *args):
        """
        :param args[0]: t, simulation time (seconds)
        :param args[1]: y, (only for closed_loop_controller), measurement of system, as defined by measurement model
        :param args[2]: x, (only for closed_loop_controller), full state of the full order system
        :return: Input u to the system
        """
        t = args[0]
        if t <= self.t_seq[-1]:
            return self.u_interp(t)
        else:
            return np.zeros(self.m)

    def convert_u_standard_form(self, u):
        if u.ndim == 1:
            u = u.reshape(1, -1)

        if u.shape[0] != self.m and u.shape[1] == self.m:
            print('Transposing the control sequence to be {} x {}'.format(self.m, u.shape[0]))
            u = u.T
        elif u.shape[0] != self.m and u.shape[1] != self.m:
            print('Control sequence ({} x {}) does not specify proper number of inputs ({} x -)'.format(
                u.shape[0], u.shape[1], self.m))
            print('Setting control to zero')
            u = np.zeros((self.m, 1))
        return u
