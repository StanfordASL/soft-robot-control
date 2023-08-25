import numpy as np
from scipy.interpolate import interp1d

from sofacontrol.tpwl.observer import FullStateObserver
from sofacontrol import closed_loop_controller
from sofacontrol import open_loop_controller
from sofacontrol.lqr.ilqr import iLQR
from sofacontrol.lqr.traj_tracking_lqr import TrajTrackingLQR
from sofacontrol.lqr.lqr import CLQR, DLQR
from sofacontrol.scp.ros import GuSTOClientNode
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
    def __init__(self, dyn_sys, cost_params, dt=0.01, observer=None, delay=2, u0=None):
        super(TemplateController, self).__init__()
        self.dyn_sys = dyn_sys
        self.dt = dt
        self.input_dim = self.dyn_sys.get_input_dim()
        self.state_dim = self.dyn_sys.get_state_dim()

        self.cost_params = cost_params

        # Define observer
        if observer is not None:
            self.observer = observer
        else:
            self.observer = FullStateObserver(self.state_dim, self.dyn_sys.H)

        self.t_delay = delay
        if u0 is not None:
            self.u0 = u0
        else:
            self.u0 = np.zeros(self.input_dim)

        self.t_compute = 0.
        self.u = self.u0

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
        :param y: Measurement at time time
        :param x: Full order state at time time. Only used at start of control if using an observer, for initialization
        """

        # Update observer using previous control and current measurement, note that on the first
        # step self.u is set to self.u0 automatically in __init__
        sim_time = round(sim_time, 4)
        x_actual = self.dyn_sys.rom.compute_RO_state(xf=x)

        self.observer.update(u_prev, y, self.sim_dt, x=x_actual)

        # Startup portion of controller, before OCP controller is activated
        if round(sim_time, 4) < round(self.t_delay, 4):
            self.u = self.u0

        # Optimal controller is active
        else:
            # Updating controller (self.u) and/or policy (if first step or receding horizon)
            if round(sim_time - self.t_delay, 4) >= round(self.t_compute, 4):  # self.t_compute set to
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
        if self.observer is not None:
            info['observer_params'] = self.observer.get_observer_params()
        if self.dyn_sys is not None:
            info['dyn_sys_params'] = self.dyn_sys.get_sim_params()
            info['state_dim'] = self.dyn_sys.get_state_dim()
            info['input_dim'] = self.dyn_sys.get_input_dim()
        return info


class ilqr(TemplateController):

    def __init__(self, dyn_sys, cost_params, target, dt=0.01, observer=None, delay=2., u0=None, **kwargs):
        super().__init__(dyn_sys=dyn_sys, cost_params=cost_params, dt=dt, observer=observer, delay=delay, u0=u0)

        self.target = target
        self.setpoint_reaching = True  # Updated in self.validate_problem() if target is moving horizon

        self.validate_problem()

        if self.setpoint_reaching:
            self.final_time, self.planning_horizon = self.get_problem_horizon(kwargs.get('tf'))
        else:
            self.final_time, self.planning_horizon = self.get_problem_horizon(self.target.t[-1])

        # ilqr policy
        self.policy = iLQR(dt=self.dt, model=self.dyn_sys, cost_params=self.cost_params, planning_horizon=self.planning_horizon)
        self.x_bar = None
        self.u_bar = None
        self.K = None

        # No offline computation performed

    def get_problem_horizon(self, tf):
        if tf is None:
            raise RuntimeError('Final time not set for single-shooting ilqr')
        final_time = tf
        planning_horizon = int(final_time / self.dt)  # = nbr_steps for single-shooting
        return final_time, planning_horizon

    def validate_problem(self):
        # Validate target instance variables & their shape
        assert self.target.z is not None and self.target.Hf is not None
        assert self.target.Hf.shape[0] == self.target.z.shape[-1]
        assert self.target.z.ndim <= 2

        if self.target.z.ndim == 2:
            self.setpoint_reaching = False  # Trajectory opt problem with moving target, i.e. a trajectory to follow

        # Enforces consistency between target output model and dyn sys output model
        assert ((self.target.Hf @ self.dyn_sys.rom.V) == self.dyn_sys.H).all()

        # Validate cost parameters & their shape
        output_dim = self.dyn_sys.get_output_dim()
        if self.setpoint_reaching:
            assert self.cost_params.Qf.shape == (output_dim, output_dim)
        assert self.cost_params.Q.shape == (output_dim, output_dim)

        assert self.cost_params.R.shape == (self.input_dim, self.input_dim)

    def compute_policy(self, t_step, x_belief):
        """
        Policy computed online based on observer belief state and current step
        """
        # Compute target with correct timestep
        if self.setpoint_reaching:
            self.policy.set_target(np.repeat(self.target.z[np.newaxis, :], self.planning_horizon + 1, axis=0))

        else:  # trajectory tracking
            z_interp = interp1d(self.target.t, self.target.z, axis=0)
            self.policy.set_target(z_interp(np.linspace(0, self.final_time, self.planning_horizon + 1)))

        # Compute policy for given z_target
        self.x_bar, self.u_bar, self.K = self.policy.ilqr_computation(x_belief)

    def compute_input(self, t_step, x_belief):
        if t_step > self.final_time:  # After end of control, return u0
            self.u = self.u0
        else:
            step = int(t_step / self.dt)
            self.u = self.u_bar[step] + self.K[step] @ (x_belief - self.x_bar[step])
        return self.u


class scp(TemplateController):
    def __init__(self, dyn_sys, cost, dt, N_replan=None, observer=None, delay=2, u0=None, wait=True, **kwargs):
        super().__init__(dyn_sys, None, dt=dt, observer=observer, delay=delay, u0=u0)
        
        if N_replan is not None:
            self.N_replan = N_replan
        else:
            self.N_replan = 1

        self.t_opt = None
        self.u_opt = None
        self.x_opt = None
        self.u_bar = None
        self.x_bar = None

        self.wait = wait
        
        self.t_next_solve = 0
        self.initialized = False

        self.solve_times = []

        # Set up the ROS client node
        self.GuSTO = GuSTOClientNode()

        self.z_opt_horizon = []
        self.t_opt_horizon = []
        self.mpc = kwargs.pop('mpc', False)

        # LQR
        from sofacontrol.lqr.lqr import dare
        self.K = []
        for i in range(self.dyn_sys.num_points):
            A_d, B_d, _ = self.dyn_sys.discretize_dynamics(self.dyn_sys.tpwl_dict['A_c'][i],
                                                             self.dyn_sys.tpwl_dict['B_c'][i],
                                                             self.dyn_sys.tpwl_dict['d_c'][i], dt)
            K, _ = dare(A_d, B_d, cost.Q, cost.R)
            self.K.append(K)

    def compute_policy(self, t_step, x_belief):
        """
        Policy computed online based on observer belief state and current time
        """
        print('t_sim = {:.3f}'.format(t_step))

        # If the controller hasn't been initialized yet start with x_belief and solve
        if not self.initialized:
            # x_belief = self.dyn_sys.rom.compute_RO_state(xf=self.dyn_sys.rom.x_ref)
            self.run_GuSTO(t_step, x_belief, wait=True)  # Upon instantiation always wait
            self.update_policy(init=True)
            self.initialized = True
        else:
            self.update_policy()

        # We solve for the next iteration at the start of the new policy (to enable real-time usage)
        # So for policy starting at t=t0, we start computing at t=t0-dt*N (N = rollout / replan horizon)
        self.t_next_solve = round(self.t_opt[-1], 6)

        # Solve for the trajectory on the next interval
        # # In theory, for a finite horizon case (trajectory following) full policy can be computed offline
        if self.mpc:
            x0 = x_belief
        else:
            x0 = self.x_opt[-1, :]

        self.run_GuSTO(self.t_opt[-1], x0, wait=self.wait)

    def run_GuSTO(self, t0, x0, wait):
        # Instantiate the GuSTO problem over the horizon
        self.GuSTO.send_request(t0, x0, wait=wait)

    def recompute_policy(self, t_step):
        """
        """
        if round(t_step, 4) >= round(self.t_next_solve, 4):
            return True
        else:
            return False

    def update_policy(self, init=False):
        # Query whether the solution is ready
        if not self.GuSTO.check_if_done():  # If running with wait=True, this is always False
            print('GuSTO cannot provide real-time compatibility, consider modifying problem')
            self.GuSTO.force_wait()

        t_opt_p, u_opt_p, x_opt_p, t_solve = self.GuSTO.get_solution(self.state_dim, self.input_dim)

        self.solve_times.append(t_solve)

        # Downsample and only take the first N_replan dt steps
        u_opt_intp = interp1d(t_opt_p, np.vstack((u_opt_p, u_opt_p[-1,:])), axis=0)
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
            u_opt_new = u_opt_intp(t_opt_new)
            x_opt_new = x_opt_intp(t_opt_new)
            self.t_opt = np.concatenate((self.t_opt, t_opt_new[1:]))
            self.u_opt = np.concatenate((self.u_opt[:-1,:], u_opt_new))
            self.x_opt = np.concatenate((self.x_opt, x_opt_new[1:,:]))

        # Define short time optimal horizon solutions
        self.z_opt_horizon.append(self.dyn_sys.x_to_zfyf(x_opt_p, zf=True))
        self.t_opt_horizon.append(t_opt_p)

        # Define interpolation functions for new optimal trajectory, note
        # that these include traj from time t = 0 onward
        self.u_bar = interp1d(self.t_opt, self.u_opt, axis=0)
        self.x_bar = interp1d(self.t_opt, self.x_opt, axis=0)

    # TODO: The clipping to input constraint is currently hard-coded (between 0 and 800)
    def compute_input(self, t_step, x_belief):
        self.GuSTO.force_spin()  # Periodic querying of client node

        # LQR
        i_near = self.dyn_sys.calc_nearest_point(self.x_bar(t_step))
        u = np.clip(self.u_bar(t_step) + self.K[i_near] @ (x_belief - self.x_bar(t_step)), 0, 800)
        # u = self.u_bar(t_step) + self.K[i_near] @ (x_belief - self.x_bar(t_step))
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


class TrajTracking(TemplateController):

    """
    Implements trajectory following for non-linear systems, which is an extension to a classic LQR problem.
    Requires a trajectory as input
    """

    def __init__(self, dyn_sys, cost_params, target, dt=0.01, observer=None, delay=2., u0=None, **kwargs):

        super().__init__(dyn_sys=dyn_sys, cost_params=cost_params, dt=dt, observer=observer, delay=delay, u0=u0)

        self.target = target
        self.validate_problem()

        self.final_time = self.target.t[-1]

        # Traj Tracking LQR policy
        self.policy = TrajTrackingLQR(dt=dt, model=dyn_sys, cost_params=self.cost_params)
        self.x_bar = None
        self.u_bar = None
        self.K = None

        # Compute policy (this can be done offline)
        self.x_bar, self.u_bar, self.K = self.policy.compute_policy(self.target)

    def validate_problem(self):
        # Validate target instance variables & their shape
        assert self.target.x is not None and self.target.u is not None and self.target.t is not None
        assert self.target.x.ndim == 2 and self.target.u.ndim == 2
        assert self.target.u.shape[-1] == self.input_dim
        assert self.target.x.shape[-1] == self.state_dim

        # Validate cost parameters & their shape
        assert self.cost_params.Q.shape == (self.state_dim, self.state_dim)
        assert self.cost_params.R.shape == (self.input_dim, self.input_dim)

    def compute_policy(self, t_step, x_belief):
        # Controller policy is computed offline, hence done when initializing the controller
        pass

    def compute_input(self, t_step, x_belief):
        if t_step > self.final_time:
            self.u = self.u0
        else:
            step = int(t_step / self.dt)
            self.u = np.atleast_1d(self.u_bar[step] + self.K[step] @ (x_belief - self.x_bar[step]))
        return self.u


class StateDLQR(TemplateController):
    """
        Infinite horizon discrete LQR framework, which can be used effectively for setpoint reaching tasks
    """
    LQR_type = DLQR

    def __init__(self, dyn_sys, cost_params, target, dt=0.01, observer=None, delay=2, u0=None, **kwargs):

        super().__init__(dyn_sys=dyn_sys, cost_params=cost_params, dt=dt, observer=observer, delay=delay, u0=u0)
        self.target = target
        self.validate_problem()

        # Policy
        self.policy = self.LQR_type(dt=dt, model=dyn_sys, cost_params=self.cost_params)
        self.x_bar = None
        self.u_bar = None
        self.K = None

        # Compute policy, this can be performed offline
        self.x_bar, self.u_bar, self.K = self.policy.compute_policy(target=self.target)

    def validate_problem(self):
        # Validate target instance variables & their shape
        assert self.target.A is not None and self.target.B is not None and self.target.u is not None \
               and self.target.x is not None

        assert self.target.A.shape == (self.state_dim, self.state_dim)
        assert self.target.B.shape == (self.state_dim, self.input_dim)
        assert self.target.x.shape[-1] == self.state_dim
        assert self.target.u.shape[-1] == self.input_dim

        assert self.cost_params.Q.shape == (self.state_dim, self.state_dim)
        assert self.cost_params.R.shape == (self.input_dim, self.input_dim)

    def compute_policy(self, t_step, x_belief):
        pass

    def compute_input(self, t_step, x_belief):
        self.u = self.u_bar + self.K @ (x_belief - self.x_bar)
        return self.u


class StateCLQR(StateDLQR):
    """
    This is an infinite horizon continuous LQR framework, which can be used effectively for setpoint reaching
    """
    LQR_type = CLQR


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
        :param x: full order state of the system
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