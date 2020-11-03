import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

from sofacontrol.baselines.ros import MPCClientNode
from sofacontrol.closed_loop_controller import TemplateController
from sofacontrol.utils import load_data


# NOTE: Only tested on a single output node, position only (3 dim), and a delay of 1. To test with larger delays and
# compare with MATLAB code


class KoopmanData:
    def __init__(self, scale, delay):
        self.delay = delay
        self.scaling = KoopmanScaling(scale)

        self.y_norm = None  # Down-scaled
        self.u_norm = None  # Down-scaled

    def add_measurement(self, y, u):
        """
        Adds data point (called in online simulation): The reduced order state incorporates delays hence this is req.
        :param y: Measurement of low-dimensional state (e.g. position of EE node)
        :param u: Control input
        """
        if self.y_norm is None:
            self.y_norm = self.scaling.scale_down(y=y)
            self.u_norm = self.scaling.scale_down(u=u)

        else:
            self.y_norm = np.append(self.y_norm, self.scaling.scale_down(y=y), axis=0)
            self.u_norm = np.append(self.u_norm, self.scaling.scale_down(u=u), axis=0)

    def get_zeta(self, step=-1):
        if len(self.y_norm) < self.delay + 1:
            return None
        else:
            y = self.y_norm[step]
            u = self.u_norm[step]

            ydel = np.zeros((self.delay * self.y_norm.shape[1]))
            udel = np.zeros((self.delay * self.u_norm.shape[1]))

            for j in range(self.delay):
                fillrange_y = range(self.y_norm.shape[1] * j, self.y_norm.shape[1] * (j + 1))
                fillrange_u = range(self.u_norm.shape[1] * j, self.u_norm.shape[1] * (j + 1))
                ydel[fillrange_y] = self.y_norm[step - (j + 1), :]
                udel[fillrange_u] = self.u_norm[step - (j + 1), :]

            zetak = np.hstack([y, ydel, udel])
            return zetak


class KoopmanOfflineData(KoopmanData):
    """
    Defines data required for Koopman based modeling.
    """

    def __init__(self, scale, delay):
        super().__init__(scale, delay)
        self.y = None
        self.u = None
        self.t = None

        self.zeta = None

    def load_offline_data(self, file):
        """
        Load a file containing (output node information, time step and control input) from a simulation or hardware exp.
        :param file: Data file path
        """
        data = load_data(file)
        self.y = data['z']
        self.t = data['t']
        self.u = data['u']
        self.y_norm = self.scaling.scale_down(y=self.y)
        self.u_norm = self.scaling.scale_down(u=self.u)

    def add_zeta_offline(self):
        """
        Bulk computation of zeta, low dimensional state considered for Koopman operator
        """
        self.zeta = []

        for i in range(self.delay, self.y_norm.shape[0]):
            self.zeta.append(self.get_zeta(step=i))
        self.zeta = np.asarray(self.zeta)


class KoopmanScaling:
    """
    Provides functions for going back and forth between scaled and normalized data given the scale as parameter
    """

    def __init__(self, scale):
        self.y_offset = scale['y_offset'][0, 0]
        self.y_factor = scale['y_factor'][0, 0]
        self.u_offset = scale['u_offset'][0, 0]
        self.u_factor = scale['u_factor'][0, 0]

    def scale_up(self, u=None, y=None):
        if y is not None:
            return y * self.y_factor + self.y_offset
        elif u is not None:
            return u * self.u_factor + self.u_offset

    def scale_down(self, u=None, y=None):
        if y is not None:
            return (y - self.y_offset) / self.y_factor
        elif u is not None:
            return (u - self.u_offset) / self.u_factor


class KoopmanModel:
    """
    Builds a Koopman model, and has lift_f, which defines how data is lifted from zeta --> z (lifted state)
    """

    def __init__(self, model_in, params_in):
        self.A_d = model_in['A'][0, 0]
        self.B_d = model_in['B'][0, 0]
        self.C = model_in['C'][0, 0]
        self.H = self.C.copy()
        self.M = model_in['M'][0, 0]
        self.K = model_in['K'][0, 0]

        self.n = int(params_in['n'])
        self.m = int(params_in['m'])
        self.N = int(params_in['N'])
        self.state_dim = int(params_in['nzeta'])
        self.delays = int(params_in['delays'])
        self.obs_degree = int(params_in['obs_degree'])
        self.obs_type = str(params_in['obs_type'][0, 0][0, 0][0])
        self.Ts = float(params_in['Ts'])
        self.scale = params_in['scale'][0, 0]

        self.assert_dimensions()
        self.lift_data = self.get_lifting_function()

    def assert_dimensions(self):
        """
        Assert dimensions of Model are correct
        """
        assert self.A_d.shape == (self.N, self.N)
        assert self.B_d.shape == (self.N, self.m)
        assert self.C.shape == (self.n, self.N)

    def get_lifting_function(self):
        """
        Lift data from the basis functions to the lifted state
        :return: Lambdified expression in numpy style using sympy
        """
        if self.obs_type == 'poly':
            zeta = sp.Matrix(sp.symbols('zeta1:{}'.format(self.state_dim + 1)))
            polynoms = sorted(itermonomials(list(zeta), self.obs_degree),
                              key=monomial_key('grlex', list(reversed(zeta))))
            polynoms.append(polynoms[0])
            polynoms = polynoms[1:]
            assert len(polynoms) == self.N
            return sp.lambdify(zeta, polynoms, 'numpy')
        else:
            print('{} is not implemented / not a valid selection. Please select a different obs type'
                  .format(self.obs_type))


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

        self.data = KoopmanData(self.dyn_sys.scale, self.dyn_sys.delays)

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

        self.t_delay = delay

    def set_sim_timestep(self, dt):
        """
        Required for interfacing with Sofa
        """
        self.sim_dt = dt

    def compute_policy(self, t_step, zeta_belief):
        xlift_curr = np.asarray(self.dyn_sys.lift_data(*zeta_belief))  # x_belief is directly y_belief in this case
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
        # if self.Y is not None:
        if self.Y is not None and not self.Y.contains(y):
            y = self.Y.project_to_polyhedron(y)
        self.data.add_measurement(y, u_prev)
        if round(sim_time, 4) < round(self.t_delay, 4):
            self.u = self.u0

        # Optimal controller is active
        else:
            # Updating controller (self.u) and/or policy (if first step or receding horizon)
            if round(sim_time - self.t_delay, 4) >= round(self.t_compute, 4):  # self.t_compute set to
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
        info['solve_times'] = self.solve_times
        info['rollout_time'] = self.rollout_horizon * self.dt
        return info


class KoopmanObserver:
    def __init__(self):
        self.z = None

    def update(self, u, y, dt, x=None):
        self.z = y
