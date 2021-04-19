import numpy as np
from scipy.interpolate import interp1d

from sofacontrol.baselines.ros import MPCClientNode
from sofacontrol.closed_loop_controller import TemplateController
from sofacontrol.lqr.lqr import dare
from sofacontrol.baselines.rompc.observer import DiscreteLuenbergerObserver


class ROMPC(TemplateController):
    """
    """

    def __init__(self, dyn_sys, cost, costL, dt, N_replan=None, delay=2, u0=None, wait=True):
        self.dyn_sys = dyn_sys
        self.dt = dt
        self.input_dim = self.dyn_sys.get_input_dim()
        self.state_dim = self.dyn_sys.get_state_dim()

        self.t_compute = 0.
        self.t_delay = delay

        if u0 is not None:
            self.u0 = u0
        else:
            self.u0 = np.zeros(self.input_dim)
        self.u = self.u0

        if N_replan is not None:
            self.N_replan = N_replan

        else:
            self.N_replan = 1

        self.t_opt = None
        self.u_opt = None
        self.x_opt = None
        self.ubar = None
        self.xbar = None

        self.solve_times = []

        self.wait = wait

        self.t_next_solve = 0
        self.initialized = False

        # Set up the ROS client node
        self.MPC = MPCClientNode()

        # Gains for controller
        self.K, _ = dare(self.dyn_sys.A_d, self.dyn_sys.B_d, cost.Q, cost.R)

        # Define state estimator
        self.observer = DiscreteLuenbergerObserver(dyn_sys, costL.Q, costL.R)

    def evaluate(self, sim_time, y, x, u_prev):
        """
        :param time: Time in the simulation [s]
        :param y: Measurement at sim_time
        :param x: Full order state at sim_time.
        :param u_prev: previous control input
        """
        if not self.initialized:
            # Initialize estimator
            self.observer.initialize(x)

        # Startup portion of controller, before ROMPC controller is activated
        if round(sim_time, 4) < round(self.t_delay, 4):
            self.u = self.u0
        else:
            if round(sim_time - self.t_delay, 4) >= round(self.t_compute, 4):
                # Re-solve the reduced order OCP
                if round(self.t_compute, 4) >= round(self.t_next_solve, 4):
                    self.ubar, self.xbar = self.solve_OCP()
                    print('t_sim = {:.3f}'.format(self.t_compute))

                # Compute control for the current time step
                self.u = self.ubar(self.t_compute) + self.K @ (self.observer.x - self.xbar(self.t_compute))

                self.t_compute += self.dt  # Increment t_compute
                self.MPC.force_spin()  # Periodic querying of ROS client node

        self.u = np.atleast_1d(self.u)

        # Update state estimator
        self.observer.update(self.u, y)

        return self.u.copy()  # Returns copy of self.u

    def solve_OCP(self):
        """
        Solve the reduced order OCP problem and get solutions ubar and xbar. At the first iteration
        use x_belief as initial condition, otherwise use end of previous solve
        """
        if not self.initialized:
            self.MPC.send_request(self.t_compute, self.observer.x, wait=True)
            ubar, xbar = self.get_OCP_solution(init=True)
            self.initialized = True
        else:
            ubar, xbar = self.get_OCP_solution()

        # Start the MPC running for the next interval
        self.MPC.send_request(self.t_opt[-1], self.x_opt[-1, :], wait=self.wait)
        self.t_next_solve = round(self.t_opt[-1], 6)

        return ubar, xbar

    def get_OCP_solution(self, init=False):
        """
        Takes the OCP solution and appends to the nominal trajectory
        """
        # Query whether the solution is ready
        if not self.MPC.check_if_done():  # If running with wait=True, this is always False
            print('MPC cannot provide real-time compatibility, consider modifying problem')
            self.MPC.force_wait()

        # Get solution from solver
        t_opt_p, u_opt_p, x_opt_p, t_solve = self.MPC.get_solution(self.state_dim, self.input_dim)
        self.solve_times.append(t_solve)

        # Downsample and only take the first N_replan dt steps
        u_opt_intp = interp1d(t_opt_p, np.vstack((u_opt_p, u_opt_p[-1, :])), axis=0)
        x_opt_intp = interp1d(t_opt_p, x_opt_p, axis=0)
        if init:
            self.t_opt = self.dt * np.arange(self.N_replan + 1)
            self.u_opt = u_opt_intp(self.t_opt)
            self.x_opt = x_opt_intp(self.t_opt)
        else:
            t_opt_new = self.t_opt[-1] + self.dt * np.arange(self.N_replan + 1)
            u_opt_new = u_opt_intp(t_opt_new)
            x_opt_new = x_opt_intp(t_opt_new)
            self.t_opt = np.concatenate((self.t_opt, t_opt_new[1:]))
            self.u_opt = np.concatenate((self.u_opt[:-1, :], u_opt_new))
            self.x_opt = np.concatenate((self.x_opt, x_opt_new[1:, :]))

        # Define interpolation functions for new optimal trajectory, note
        # that these include traj from time t = 0 onward
        ubar = interp1d(self.t_opt, self.u_opt, axis=0)
        xbar = interp1d(self.t_opt, self.x_opt, axis=0)
        return ubar, xbar

    def save_controller_info(self):
        info = dict()
        info['t_opt'] = self.t_opt
        info['u_opt'] = self.u_opt
        info['z_opt'] = self.dyn_sys.x_to_zfyf(self.x_opt, zf=True)
        info['solve_times'] = self.solve_times
        info['rollout_time'] = self.N_replan * self.dt
        return info
