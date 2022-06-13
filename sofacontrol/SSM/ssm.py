import numpy as np
from scipy.interpolate import interp1d
import sofacontrol.utils as scutils
import sofacontrol.tpwl.tpwl_utils as tutils
from sofacontrol.mor import pod

###  DEFAULT VALUES
DISCR_METHOD = 'zoh'  # other options: be, zoh, bil. Forward Euler, Backward Euler, Bilinear transf. or Zero-Order Hold
TPWL_METHOD = 'nn'  # other options: weighting. Nearest neighbor or exponential weighting

DISCR_DICT = {'fe': 'forward Euler', 'be': 'implicit Euler', 'bil': 'bilinear transform', 'zoh': 'zero-order hold'}


class SSM:
    """
    Provided a data dictionary or file location this class can give Jacobians w.r.t state, input and affine term or
    forward simulate the dynamics (i.e. perform rollout)
    """

    def __init__(self, eq_point, maps, n, m, o, discrete=False, discr_method='fe'):

        # TODO: Using code gen from matlab for manifold maps, temporarily
        self.maps = maps
        # Map from reduced to observable space
        self.C_map = self.maps['C']
        # Map from observable to reduced coordinates
        self.W_map = self.maps['W']

        self.discrete = discrete
        self.discr_method = discr_method

        # Get state and input dimensions
        self.state_dim = n
        self.input_dim = m
        self.output_dim = o

        # Set reference equilibrium point
        self.z_ref = eq_point

        # Variables for precomputing discrete time matrices if desired
        self.A_d = None
        self.B_d = None
        self.d_d = None

        # Set performance to identity because observation and performance states the same
        self.H = np.eye(n)

    def update_state(self, x, u, dt):
        raise NotImplementedError("update_state must be overriden by a child class")

    def get_jacobians(self, x, dt=None):
        raise NotImplementedError("get_jacobians must be overriden by a child class")

    # Notation: zf is unshifted equilibrium point
    def zfyf_to_zy(self, zf=None):
        """
        :zf: (N, n_z) or (n_z,) array
        :yf: (N, n_y) or (n_y,) array
        """
        if zf is not None and self.z_ref is not None:
            return zf - self.z_ref
        else:
            raise RuntimeError('Need to specify equilibrium point')

    def zy_to_zfyf(self, z=None):
        """
        :z: (N, n_z) or (n_z,) array
        :y: (N, n_y) or (n_y,) array
        """
        if z is not None and self.z_ref is not None:
            return z + self.z_ref
        else:
            raise RuntimeError('Need to specify equilibrium point')

    # x is reduced state => This function goes from reduced state to (shifted) observation
    # TODO: This may also present some issues since C_map expects (n, N) where n is the ROM state
    def x_to_zfyf(self, x, zf=True):
        """
        :x: (N, n_x) or (n_x,) array
        :zf: boolean
        :yf: boolean
        """
        return self.C_map(x.T).T + self.z_ref

    def x_to_zy(self, x):
        """
        :x: (N, n_x) or (n_x,) array
        :z: boolean
        :y: boolean
        """
        return self.C_map(x)

    def get_sim_params(self):
        return {'beta_weighting': self.beta_weighting, 'discr_method': self.discr_method,
                'dist_weights': self.dist_weights}

    def get_state_dim(self):
        return self.state_dim

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    def rollout(self, x0, u, dt):
        """
        :x0: initial condition
        :u: array of control (N, n_u)
        :dt: time step

        Returns reduced order state x (N + 1, n_x) and performance variable z (N + 1, n_z)
        """
        N = u.shape[0]
        x = np.zeros((N + 1, self.state_dim))

        # Set initial condition
        x[0,:] = x0

        # Simulate
        for i in range(N):
            x[i+1,:] = self.update_state(x[i,:], u[i,:], dt)

        z = self.x_to_zfyf(x)

        return x, z


class SSMDynamics(SSM):
    """
    """
    def __init__(self, eq_point, maps, n, m, o, discrete=False, discr_method='fe'):
        super(SSMDynamics, self).__init__(eq_point, maps, n, m, o, discrete=discrete, discr_method=discr_method)

    def update_state(self, x, u, dt):
        """
        Compute x+ based on a discretization time of dt.
        :x: current state
        :u: current input
        :dt: time step
        """
        A_d, B_d, d_d = self.get_jacobians(x, dt=dt, u=u)
        return self.update_dynamics(x, u, A_d, B_d, d_d)

    def get_jacobians(self, x, dt=None, u=None):
        """
        Extract the Jacobians A, B, d (or A_d, B_d, d_d) at the state x
        :x: reduced state
        :discrete (optional): Specify whether to extract continuous or discrete dynamics
        """
        assert u is not None, 'Need to supply current input'
        x = x.reshape(self.state_dim, 1)
        u = u.reshape(self.input_dim, 1)
        if self.discrete:
            A = self.maps['A_d'](x)
            B = self.maps['B_d'](x)
            f_nl = self.maps['f_nl_d'](x, u)
            d = f_nl - A@x - B@u
        else:
            assert dt is not None, "Need to specify dt since dynamics are continuous time"

            A_c = self.maps['A'](x)
            B_c = self.maps['B'](x)
            f_nl_c = self.maps['f_nl'](x, u)
            d_c = f_nl_c - A_c @ x - B_c @ u
            A, B, d = self.discretize_dynamics(A_c, B_c, d_c, dt)

        return A, B, np.squeeze(d)

    def discretize_dynamics(self, A_c, B_c, d_c, dt):
        if self.discr_method == 'fe':
            A_d = np.eye(A_c.shape[0]) + dt * A_c
            B_d = dt * B_c
            d_d = dt * d_c

        elif self.discr_method == 'be':
            A_d = np.linalg.inv(np.eye(A_c.shape[0]) - dt * A_c)
            sep_term = np.linalg.inv(A_c) @ (A_d - np.eye(A_c.shape[0]))
            B_d = sep_term @ B_c
            d_d = sep_term @ d_c

        elif self.discr_method == 'bil':
            A_d = (np.eye(A_c.shape[0]) + 0.5 * dt * A_c) @ np.linalg.inv(np.eye(A_c.shape[0])
                                                                               - 0.5 * dt * A_c)
            sep_term = np.linalg.inv(A_c) @ (A_d - np.eye(A_c.shape[0]))
            B_d = sep_term @ B_c
            d_d = sep_term @ d_c

        elif self.discr_method == 'zoh':
            A_d, B_d, d_d = scutils.zoh_affine(A_c, B_c, d_c, dt)

        else:
            raise RuntimeError('self.discr_method must be in [fe, be, bil, zoh]')

        return A_d, B_d, d_d

    #TODO: This might be a problem later due to expectation of shape
    @staticmethod
    def update_dynamics(x, u, A_d, B_d, d_d):
        x_next = np.squeeze(A_d @ x) + np.squeeze(B_d @ u) + np.squeeze(d_d)
        return x_next

    def get_ref_point(self):
        return self.z_ref

    def compute_RO_state(self, z):
        """
        Compute reduced order projection of vector
        :param zf: position and velocity of observed state
        :return: Reduced order vector
        """
        return self.W_map(z - self.z_ref)