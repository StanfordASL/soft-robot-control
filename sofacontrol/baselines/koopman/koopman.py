import numpy as np
from scipy.interpolate import interp1d

from sofacontrol.baselines.koopman.koopman_utils import KoopmanData
from sofacontrol.baselines.ros import MPCClientNode
from sofacontrol.closed_loop_controller import TemplateController
from sofacontrol.lqr.traj_tracking_lqr import TrajTrackingLQR


# NOTE: Only tested on a single output node, position only (3 dim), and a delay of 1. To test with larger delays and
# compare with MATLAB code

class KoopmanMPC(TemplateController):
    """
    Koopman MPC interface with Sofa. Similar style to sofacontrol/tpwl/controllers.py. Interfaces with Sofa and ROS
    through client node. Not yet designed to be real-time capable
    """

    def __init__(self, dyn_sys, delay=2, u0=None, wait=True, **kwargs):
        super().__init__()

        # Related to Controllers.py file
        self.dyn_sys = dyn_sys

        self.input_dim = self.dyn_sys.m
        self.state_dim = self.dyn_sys.N

        self.dt = self.dyn_sys.Ts

        self.observer = KoopmanObserver()

        self.Y = kwargs.get('Y')
        if u0 is not None:
            self.u0 = u0
        else:
            self.u0 = np.zeros(self.input_dim)

        self.t_compute = 0.
        self.u = self.u0

        self.solve_times = []

        self.data = KoopmanData(self.dyn_sys.scale, self.dyn_sys.delays, inputInFeatures=self.dyn_sys.inputInFeatures)

        # self.planning_horizon = kwargs.get('planning_horizon', 10)
        self.rollout_horizon = kwargs.get('rollout_horizon', 1)
        # Consider interpolation or input hold, used if sim dt is different than system dt
        self.input_hold = kwargs.get('input_hold', False)

        # Related to MPC Solver Node
        self.t_opt = None
        self.u_opt = None
        self.x_opt = None
        self.u_bar = None
        self.x_bar = None

        self.x_opt_full = None

        self.wait = wait

        self.t_next_solve = 0
        self.initiailzed = False

        self.MPC = MPCClientNode()

        self.z_opt_horizon = []
        self.t_opt_horizon = []

        self.t_delay = delay

    def set_sim_timestep(self, dt):
        """
        Required for interfacing with Sofa
        """
        self.sim_dt = dt

    def compute_policy(self, t_step, zeta_belief):
        # Projects to "dominant" Koopman modes if 'W' is defined. Else 'W' is identity
        xlift_curr = np.dot(self.dyn_sys.W, np.asarray(self.dyn_sys.lift_data(*zeta_belief)))  # x_belief is directly y_belief in this case
        self.MPC.send_request(round(t_step, 4), xlift_curr, wait=True)

        if not self.MPC.check_if_done():  # If running with wait=True, this is always False
            print('GuSTO cannot provide real-time compatibility, consider modifying problem')
            self.MPC.force_wait()

        t_opt_p, u_opt_p, x_opt_p, t_solve = self.MPC.get_solution(self.state_dim,
                                                                   self.input_dim)  # Why do we need state dim and input dim here??
        t_opt_p = np.round(t_opt_p, 4)
        u_opt_p = self.data.scaling.scale_up(u=u_opt_p)

        self.solve_times.append(t_solve)

        # Downsample and only take the first N_replan dt steps
        u_opt_intp = interp1d(t_opt_p, np.vstack((u_opt_p, u_opt_p[-1, :])), axis=0)
        x_opt_intp = interp1d(t_opt_p, x_opt_p, axis=0)

        if self.t_opt is None:
            t_opt_new = self.dt * np.arange(self.rollout_horizon + 1)
            u_opt_new = u_opt_intp(t_opt_new)
            x_opt_new = x_opt_intp(t_opt_new)
            self.t_opt = t_opt_new
            self.u_opt = u_opt_new
            self.x_opt = x_opt_new
            self.x_opt_full = np.expand_dims(x_opt_p, axis=0)
        else:
            t_opt_new = self.t_opt[-1] + self.dt * np.arange(self.rollout_horizon + 1)
            u_opt_new = u_opt_intp(t_opt_new)
            x_opt_new = x_opt_intp(t_opt_new)
            self.t_opt = np.concatenate((self.t_opt, t_opt_new[1:]))
            self.t_opt = np.round(self.t_opt, 4)
            self.u_opt = np.concatenate((self.u_opt[:-1, :], u_opt_new))
            self.x_opt = np.concatenate((self.x_opt, x_opt_new[1:, :]))
            self.x_opt_full = np.concatenate((self.x_opt_full, np.expand_dims(x_opt_p, axis=0)))

        # Define short time optimal horizon solutions
        self.z_opt_horizon.append(self.data.scaling.scale_up(y=(self.dyn_sys.H @ x_opt_p.T).T))
        self.t_opt_horizon.append(t_opt_p)

        # Define interpolation functions for new optimal trajectory, note
        # that these include traj from time t = 0 onward

        if self.input_hold:
            self.u_bar = interp1d(self.t_opt, self.u_opt, kind='previous', axis=0)
            self.x_bar = interp1d(self.t_opt, self.x_opt, kind='previous', axis=0)
        else:
            self.u_bar = interp1d(self.t_opt, self.u_opt, axis=0)
            self.x_bar = interp1d(self.t_opt, self.x_opt, axis=0)

    def recompute_policy(self, t_step):
        step = round(round(t_step, 4) / self.dt)
        i = int(step % self.rollout_horizon)
        return True if i == 0 else False  # Recompute if rollout horizon reached (more explicit than: not i)

    def compute_input(self, t_step, z_belief):
        self.MPC.force_spin()

        u = self.u_bar(t_step)
        return u

    def evaluate(self, sim_time, y, x, u_prev):
        """
        Update observer at each step of simulation (each function call). Updates controller at controller frequency
        :param time: Time in the simulation [s]
        :param y: Measurement at time time
        :param x: Full order state at time time. Only used at start of control if using an observer, for initialization
        """
        # print(sim_time)
        sim_time = round(sim_time, 4)
        # Startup portion of controller, before OCP controller is activated
        self.observer.update(None, y, None)

        if self.Y is not None and not self.Y.contains(y):
            y = self.Y.project_to_polyhedron(y)
            
        self.data.add_measurement(y, u_prev)
        if round(sim_time, 4) < round(self.t_delay, 4):
            self.u = self.u0
        # Optimal controller is active
        else:
            # Updating controller (self.u) and/or policy (if first step or receding horizon)
            if round(sim_time - self.t_delay, 4) >= round(self.t_compute, 4):  # self.t_compute set to
                # Gets lifted Koopman state based on current/past measurements
                zeta_belief = self.data.get_zeta()

                if self.recompute_policy(self.t_compute):
                    self.compute_policy(self.t_compute, zeta_belief)

                self.u = self.compute_input(self.t_compute, zeta_belief)

                self.t_compute += self.dt  # Increment t_compute
                self.t_compute = round(self.t_compute, 4)
        self.u = np.atleast_1d(self.u)
        return self.u.copy()  # Returns copy of self.u

    def save_controller_info(self):
        info = dict()
        info['t_opt'] = self.t_opt
        info['u_opt'] = self.u_opt
        info['z_opt'] = self.data.scaling.scale_up(y=(self.dyn_sys.H @ self.x_opt.T).T)
        info['zopt_full'] = self.data.scaling.scale_up(
            y=np.einsum("ij, klj -> ikl", self.dyn_sys.H, self.x_opt_full).T).transpose((1, 0, 2))

        info['z_rollout'] = self.z_opt_horizon
        info['t_rollout'] = self.t_opt_horizon

        info['solve_times'] = self.solve_times
        info['rollout_time'] = self.rollout_horizon * self.dt
        return info

class TrajTracking(TemplateController):

    """
    Implements trajectory following for non-linear systems, which is an extension to a classic LQR problem.
    Requires a trajectory as input
    """

    def __init__(self, dyn_sys, cost_params, target, u_lb, u_ub, delay=2., u0=None, **kwargs):

        super().__init__()

        self.target = target
        self.dyn_sys = dyn_sys
        self.cost_params = cost_params
        self.t_delay = delay
        self.dt = self.dyn_sys.Ts

        self.final_time = self.target.t[-1]
        self.t_compute = 0.

        self.observer = KoopmanObserver()
        self.data = KoopmanData(self.dyn_sys.scale, self.dyn_sys.delays, inputInFeatures=self.dyn_sys.inputInFeatures)
        self.input_dim = self.dyn_sys.m
        self.state_dim = self.dyn_sys.state_dim
        self.u_lb = u_lb
        self.u_ub = u_ub

        if u0 is not None:
            self.u0 = u0
        else:
            self.u0 = np.zeros(self.input_dim)
        self.u = self.u0

        self.validate_problem()

        # Traj Tracking LQR policy
        self.policy = TrajTrackingLQR(dt=self.dyn_sys.Ts, model=dyn_sys, cost_params=self.cost_params)
        self.x_bar = None
        self.u_bar = None
        self.K = None

        # Compute policy (this can be done offline). Computes feedforward term from target.u
        self.x_bar, self.u_bar, self.K = self.policy.compute_policy(self.target)

    def validate_problem(self):
        # Validate target instance variables & their shape
        assert self.target.x is not None and self.target.u is not None and self.target.t is not None
        assert self.target.x.ndim == 2
        assert self.target.u.shape[-1] == self.input_dim
        # assert self.target.x.shape[-1] == self.state_dim

        # Validate cost parameters & their shape
        # assert (self.dyn_sys.H.T @ self.cost_params.Q @ self.dyn_sys.H).shape == (self.state_dim, self.state_dim)
        assert self.cost_params.R.shape == (self.input_dim, self.input_dim)

    def compute_policy(self, t_step, x_belief):
        # Controller policy is computed offline, hence done when initializing the controller
        pass

    def compute_input(self, t_step, x_belief):
        if t_step > self.final_time:
            self.u = self.u0
        else:
            step = int(t_step / self.dt)
            zeta_belief = np.dot(self.dyn_sys.W, np.asarray(self.dyn_sys.lift_data(*x_belief)))
            zeta_ref = np.dot(self.dyn_sys.W, np.asarray(self.dyn_sys.lift_data(*self.x_bar[step])))

            # feedforward + feedback
            self.u = np.clip(np.atleast_1d(self.u_bar[step] + self.K[step] @ (zeta_belief - zeta_ref)),
                             self.u_lb, self.u_ub)
            # feedforward only
            # self.u = np.clip(np.atleast_1d(self.u_bar[step]), self.u_lb, self.u_ub)

            # feedback only
            # self.u = np.clip(np.atleast_1d(self.K[step] @ (zeta_belief - zeta_ref)), self.u_lb, self.u_ub)

            self.u = self.data.scaling.scale_up(u=self.u)
        return self.u

    def evaluate(self, sim_time, y, x, u_prev):
        """
        Update observer at each step of simulation (each function call). Updates controller at controller frequency
        :param time: Time in the simulation [s]
        :param y: Measurement at time time
        :param x: Full order state at time time. Only used at start of control if using an observer, for initialization
        """
        # print(sim_time)
        sim_time = round(sim_time, 4)
        # Startup portion of controller, before OCP controller is activated
        self.observer.update(None, y, None)
            
        self.data.add_measurement(y, u_prev)
        if round(sim_time, 4) < round(self.t_delay, 4):
            self.u = self.u0
        # Optimal controller is active
        else:
            # Updating controller (self.u) and/or policy (if first step or receding horizon)
            if round(sim_time - self.t_delay, 4) >= round(self.t_compute, 4):  # self.t_compute set to
                # Gets lifted Koopman state based on current/past measurements
                zeta_belief = self.data.get_zeta()

                self.u = self.compute_input(self.t_compute, zeta_belief)

                self.t_compute += self.dt  # Increment t_compute
                self.t_compute = round(self.t_compute, 4)
        if self.u.ndim > 1:
            self.u = self.u.flatten()
        else:
            self.u = np.atleast_1d(self.u)
        return self.u.copy()  # Returns copy of self.u


class KoopmanObserver:
    def __init__(self):
        self.z = None

    def update(self, u, y, dt, x=None):
        self.z = y
