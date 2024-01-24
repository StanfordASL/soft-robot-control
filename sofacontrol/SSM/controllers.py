import numpy as np
from scipy.interpolate import interp1d
import time

from sofacontrol.SSM.SSM_utils import SSMData
from sofacontrol.SSM.observer import SSMObserver, DiscreteEKFObserver
from sofacontrol import closed_loop_controller
from sofacontrol import open_loop_controller
from sofacontrol.scp.ros import GuSTOClientNode
from sofacontrol.utils import vq2qv, CircleObstacle
from sofacontrol.lqr.lqr import dare

"""
Functions provide different control techniques for optimal control tasks such as Trajectory Optimization, Trajectory 
Tracking and setpoint reaching. Interfaces with SOFA through closed_loop_controller.py
"""


class TemplateController(closed_loop_controller.TemplateController):
    """
    Basic control sequences for open loop control
    :param dyn_sys:
    :cost_params:
    :param dt: Timestep considered for control (usually corresponds to timestep selected in closed_loop_controller)
    :param delay: Delay before control is considered active. Comprises settling time of simulation + hypothetical
                  observer convergence
    """

    def __init__(self, dyn_sys, cost_params, dt=0.01, delay=2, u0=None, **kwargs):
        super(TemplateController, self).__init__()
        self.dyn_sys = dyn_sys
        self.dt = dt
        self.input_dim = self.dyn_sys.get_input_dim()
        self.state_dim = self.dyn_sys.get_state_dim()

        self.cost_params = cost_params

        # Define observer
        self.observer = kwargs.pop('EKF', SSMObserver(dyn_sys))

        self.data = SSMData(self.dyn_sys.delays, self.dyn_sys.y_eq)

        self.t_delay = delay
        if u0 is not None:
            self.u0 = u0
        else:
            self.u0 = np.zeros(self.input_dim)

        self.t_compute = 0.
        self.u = self.u0

        self.Y = kwargs.pop('Y', None)

    def validate_problem(self):
        raise NotImplementedError('Must be subclassed')

    def set_sim_timestep(self, dt):
        """
        This function is called in closed_loop_controller on initialization
        """
        self.sim_dt = dt

    def recompute_policy(self, t_step):
        """
        For generic controller, policy is solely computed on first timestep, requires overwriting for receding horizon
        controller implementations
        :param t_step: Timestep since controller became active
        :return: Boolean
        """
        return True if t_step == 0 else False

    def compute_policy(self, t_step, x_belief):
        """
        Computes updated policy pi (feedforward and feedback terms) for controller. Updates class instance variables
        :param t_step: Timestep since controller started
        :param x_belief: Belief state
        """
        raise NotImplementedError('Must be subclassed')

    def compute_input(self, t_step, x_belief):
        """
        Computes input at current t_step. Called at approximate frequency of controller.
        :param t_step: Timestep since controller started
        :param x_belief: Belief state
        :return: Updates self.u (control input passed to simulator)
        """
        raise NotImplementedError('Must be subclassed')

    def evaluate(self, sim_time, y, x, u_prev):
        """
        Update observer at each step of simulation (each function call). Updates controller at controller frequency
        :param time: Time in the simulation [s]
        :param y: Measurement at current time - passed in (v,q) format (raw and uncentered)
        :param x: Full order state at time. Only used at start of control if using an observer, for initialization
        """

        # Update observer using previous control and current measurement, note that on the first
        # step self.u is set to self.u0 automatically in __init__
        sim_time = round(sim_time, 4)

        # TODO: Hardcoded for now, but basically, if y doesn't include velocities, then don't flip
        # Also assumes that we're only measuring the tip. Definitely modify this later
        if y.size > 3:
            y = vq2qv(y)
        #     y = y[0:3]

        # TODO: Automate this option
        # Project to interior of constraint set. Ensures LOCP always solvable
        if self.Y is not None and self.Y.contains(y[:2]):
            if type(self.Y) == CircleObstacle:
                print("Projecting to boundary of circle obstacle")
                y = np.hstack((self.Y.project_to_boundary(y[:2]), y[2]))
            else:
                y = self.Y.project_to_polyhedron(y)

        # Store current observations for delay embedding ()
        self.data.add_measurement(y, u_prev)

        # TODO: Debugging
        # print('y: ', y)

        # Startup portion of controller, before OCP controller is activated
        if round(sim_time, 4) < round(self.t_delay, 4):
            self.u = self.u0

            # TODO: When delay embeddings are included, this needs to be commented
            # y_belief = self.data.get_y_delay()
            # self.observer.update(u_prev, y_belief, self.dt)

        # Optimal controller is active
        else:
            # Updating controller (self.u) and/or policy (if first step or receding horizon)
            # Note: waiting to start simulation allows us to also populate the history to satisfy time-delay reqs
            #print('debug position: ', y)
            #print('debug past positions: ', self.data.y[-15:] + self.dyn_sys.y_ref)
            if round(sim_time - self.t_delay, 4) >= round(self.t_compute, 4):  # self.t_compute set to 0
                # TODO: Add lifting function to get delay embeddings
                y_belief = self.data.get_y_delay()

                # TODO: Debugging
                #print('y_belief: ', y_belief + self.dyn_sys.y_ref)

                # print('debug belief positions ', y_belief + np.tile(self.dyn_sys.y_ref, self.dyn_sys.delays + 1))
                # print('Error between true output and estimated output on manifold', np.linalg.norm(self.dyn_sys.W_map(self.dyn_sys.V_map(y_belief)) + self.dyn_sys.y_ref - y))
                # TODO: Update estimated state based on past measurements

                self.observer.update(u_prev, y_belief, self.dt)

                if self.recompute_policy(self.t_compute):
                    self.compute_policy(self.t_compute, self.observer.x)

                self.u = self.compute_input(self.t_compute, self.observer.x)

                self.t_compute += self.dt  # Increment t_compute
                self.t_compute = round(self.t_compute, 4)

        self.u = np.atleast_1d(self.u)
        return self.u.copy()  # Returns copy of self.u

    def save_controller_info(self):
        """
        Should include any key identifiable information on the system being controlled, to be saved at end of simulation
        :return: dict with key identifiable information
        """
        info = dict()
        info['cost_params'] = self.cost_params
        if self.dyn_sys is not None:
            info['dyn_sys_params'] = self.dyn_sys.get_sim_params()
            info['state_dim'] = self.dyn_sys.get_state_dim()
            info['input_dim'] = self.dyn_sys.get_input_dim()
        return info


class scp(TemplateController):
    def __init__(self, dyn_sys, cost, dt, N_replan=None, delay=2, u0=None, wait=True, **kwargs):
        super().__init__(dyn_sys, None, dt=dt, delay=delay, u0=u0, **kwargs)

        if N_replan is not None:
            self.N_replan = N_replan
        else:
            self.N_replan = 1

        self.t_opt = None
        self.u_opt = None
        self.x_opt = None
        self.u_bar = None
        self.x_bar = None

        self.z_opt_horizon = []

        self.wait = wait

        self.initialized = False

        self.solve_times = []
        self.t_opt_horizon = []
        self.cost = cost

        # Set up the ROS client node
        self.GuSTO = GuSTOClientNode()
        self.feedback = kwargs.pop('feedback', False)

    def compute_policy(self, t_step, x_belief):
        """
        Policy computed online based on observer belief state and current time
        """
        print('t_sim = {:.3f}'.format(t_step))
        print(f'x_belief = {x_belief}')

        # If the controller hasn't been initialized yet start with x_belief and solve
        if not self.initialized:
            # x_belief = self.dyn_sys.rom.compute_RO_state(xf=self.dyn_sys.rom.x_ref)
            self.run_GuSTO(t_step, x_belief, wait=True)  # Upon instantiation always wait
            self.update_policy(init=True)
            self.initialized = True
        else:
            self.run_GuSTO(t_step, x_belief, wait=self.wait)
            self.update_policy()

    def run_GuSTO(self, t0, x0, wait):
        # Instantiate the GuSTO problem over the horizon
        self.GuSTO.send_request(t0, x0, wait=wait)

    def recompute_policy(self, t_step):
        step = round(round(t_step, 4) / self.dt)
        i = int(step % self.N_replan)
        return True if i == 0 else False  # Recompute if rollout horizon reached (more explicit than: not i)

    def update_policy(self, init=False):
        # Query whether the solution is ready
        if not self.GuSTO.check_if_done():  # If running with wait=True, this is always False
            print('GuSTO cannot provide real-time compatibility, consider modifying problem')
            self.GuSTO.force_wait()

        t_opt_p, u_opt_p, x_opt_p, t_solve = self.GuSTO.get_solution(self.state_dim, self.input_dim)

        self.solve_times.append(t_solve)

        # Downsample and only take the first N_replan dt steps
        u_opt_intp = interp1d(t_opt_p, np.vstack((u_opt_p, u_opt_p[-1, :])), axis=0)
        x_opt_intp = interp1d(t_opt_p, x_opt_p, axis=0)

        if init:
            t_opt_new = self.dt * np.arange(self.N_replan + 1)
            u_opt_new = u_opt_intp(t_opt_new)
            x_opt_new = x_opt_intp(t_opt_new)
            self.t_opt = t_opt_new
            self.u_opt = u_opt_new
            self.x_opt = x_opt_new
        else:
            t_opt_new = self.t_opt[-1] + self.dt * np.arange(self.N_replan + 1)
            u_opt_new = u_opt_intp(np.round(t_opt_new, 4))
            x_opt_new = x_opt_intp(np.round(t_opt_new, 4))
            self.t_opt = np.concatenate((self.t_opt, t_opt_new[1:]))
            self.u_opt = np.concatenate((self.u_opt[:-1, :], u_opt_new))
            self.x_opt = np.concatenate((self.x_opt, x_opt_new[1:, :]))

        # Define short time optimal horizon solutions
        self.z_opt_horizon.append(self.dyn_sys.x_to_zfyf(x_opt_p))
        self.t_opt_horizon.append(t_opt_p)

        # Define interpolation functions for new optimal trajectory, note
        # that these include traj from time t = 0 onward
        self.u_bar = interp1d(self.t_opt, self.u_opt, axis=0)
        self.x_bar = interp1d(self.t_opt, self.x_opt, axis=0)

        # Define optimal points to linearize about
        self.x_opt_current = x_opt_p
        self.u_opt_current = u_opt_p

    # TODO: Refactor. Use x_bar(t_step) instead of x_opt_current. Don't need any of the i_near business.
    def compute_input(self, t_step, x_belief):
        self.GuSTO.force_spin()  # Periodic querying of client node
        if self.feedback:
            # Compute LQR
            x_dist = np.linalg.norm(self.x_opt_current - x_belief, axis=1)
            i_near = np.argmin(x_dist)

            x_near = self.x_opt_current[i_near]
            A_d, B_d, _ = self.dyn_sys.get_jacobians(x_near, u=self.u_opt_current[i_near],
                                                     dt=self.dt)
            Hk, _ = self.dyn_sys.get_observer_jacobians(x_near)
            K, _ = dare(A_d, B_d, Hk.T @ self.cost.Q @ Hk, self.cost.R)
            u = self.u_bar(t_step) + K @ (x_belief - x_near)
        else:
            u = self.u_bar(t_step)

        return u

    def save_controller_info(self):
        """
        """
        info = dict()
        info['t_opt'] = self.t_opt
        info['u_opt'] = self.u_opt
        info['z_opt'] = self.dyn_sys.x_to_zfyf(self.x_opt, zf=True)
        info['solve_times'] = self.solve_times
        info['rollout_time'] = self.N_replan * self.dt
        info['z_rollout'] = self.z_opt_horizon
        info['t_rollout'] = self.t_opt_horizon
        return info


class OpenLoop(open_loop_controller.OpenLoop):
    """
    Run an open loop control but include an observer for evaluation
    """

    def __init__(self, m, t_sequence, u_sequence, save_sequence, delay=1):
        super(OpenLoop, self).__init__(m, t_sequence, u_sequence, save_sequence)
        self.observer = None
        self.u = np.zeros(self.m)
        self.delay = delay

    def add_observer(self, observer):
        self.observer = observer

    def set_sim_timestep(self, dt):
        """
        This function is called in closed_loop_controller on initialization
        """
        self.sim_dt = dt

    def evaluate(self, t, y, x, u_prev):
        """
        :param t: simulation time (seconds)
        :param y: measurement of system, as defined by measurement model
        :param x: reduced state of the system
        :return: Input u to the system
        """

        # Update observer using previous control and current measurement
        self.observer.update(self.u, y, self.sim_dt, x=x)

        t_compute = t - self.delay

        # Initialize observer on first step
        if t_compute < 0:
            self.u = np.zeros(self.m)

        else:
            # Set control u to be applied at current time step
            if t_compute < self.t_seq[-1]:
                self.u = self.u_interp(t)
            else:
                self.u = np.zeros(self.m)

        return self.u.copy()