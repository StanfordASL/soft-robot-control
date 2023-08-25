import numpy as np
import sofacontrol.utils as scutils
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
import sympy as sp
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
from sofacontrol.utils import norm2Diff

###  DEFAULT VALUES
DISCR_METHOD = 'zoh'  # other options: be, zoh, bil. Forward Euler, Backward Euler, Bilinear transf. or Zero-Order Hold
TPWL_METHOD = 'nn'  # other options: weighting. Nearest neighbor or exponential weighting

DISCR_DICT = {'fe': 'forward Euler', 'be': 'implicit Euler', 'bil': 'bilinear transform', 'zoh': 'zero-order hold'}


class SSM:
    """
    Provided a data dictionary or file location this class can give Jacobians w.r.t state, input and affine term or
    forward simulate the dynamics (i.e. perform rollout)
    """

    def __init__(self, eq_point, discrete=False, discr_method='fe', C=None, **kwargs):
        self.maps = {}

        self.discrete = discrete
        self.discr_method = discr_method

        self.model = kwargs.pop('model', None)
        self.params = kwargs.pop('params', None)
        self.isLinear = kwargs.pop('isLinear', False)

        self.state_dim = self.params['state_dim'] # [0, 0][0, 0]
        self.input_dim = self.params['input_dim'] # [0, 0][0, 0]
        self.output_dim = int(self.params['output_dim']) # [0, 0][0, 0]) #This is also performance dimension
        self.SSM_order = self.params['SSM_order'] # [0, 0][0, 0]
        self.ROM_order = self.params['ROM_order'] # [0, 0][0, 0]
        # TODO: This is new
        self.delays = self.params['delays'] # [0, 0][0, 0]
        self.obs_dim = self.params['obs_dim'] # [0, 0][0, 0]

        self.rom_phi = self.get_poly_basis(self.state_dim, self.ROM_order)
        self.ssm_phi = self.get_poly_basis(self.state_dim, self.SSM_order)
        self.chart_phi = self.get_poly_basis(self.obs_dim, self.SSM_order)
        self.control_phi = self.get_poly_basis(self.input_dim + self.state_dim, 2)

        # Observation model
        if C is not None:
            self.C = C
            assert np.shape(self.C) == (self.output_dim, self.obs_dim), 'Shape of C: ' + str(np.shape(self.C)) + '\n' + 'model shape: ' + str((self.output_dim, self.obs_dim))
        else:
            # When we learn mappings to output variables directly (no time-delays)
            self.C = np.eye(self.obs_dim, self.obs_dim)

        # Continuous-time model
        self.V = self.model['V'] # [0, 0] # Tangent space TODO: this is new
        self.w_coeff = self.model['w_coeff'] # [0, 0] # reduced to observed

        self.v_coeff = self.model['v_coeff'] # [0, 0]  # observed to reduced
        # if len(self.v_coeff) == 0:
        #     self.v_coeff = None

        self.r_coeff = self.model['r_coeff'] # [0, 0] # reduced coefficients
        self.B_r = self.model['B'] # [0, 0] #reduced control matrix

        # Discrete-time model
        # TODO: There seems to be a bug in the discrete dynamics - by some factor of scaling
        if discrete:
            self.Ts = self.model['Ts'] # [0, 0][0, 0]
            self.rd_coeff = self.model['rd_coeff'] # [0, 0]  # reduced coefficients
            self.Bd_r = self.model['Bd'] # [0, 0]  # reduced control matrix

        # Manifold parametrization
        self.W_map = self.reduced_to_output
        self.V_map = self.observed_to_reduced

        # Continuous reduced dynamics
        self.maps['f_nl'] = self.reduced_dynamics

        # Discrete reduced dynamics
        if self.discrete:
            self.maps['f_nl_d'] = self.reduced_dynamics_discrete

        # Set reference equilibrium point
        self.y_eq = eq_point
        self.y_ref = self.y_eq
        # self.y_ref = np.tile(self.y_eq, self.delays + 1)

        # Variables for precomputing discrete time matrices if desired
        self.A_d = None
        self.B_d = None
        self.d_d = None

        # Set performance to zero matrix with appropriate dimension (n_z, n_x)
        self.H = np.zeros((self.output_dim, self.state_dim))
        self.nonlinear_observer = True

    def update_state(self, x, u, dt):
        raise NotImplementedError("update_state must be overriden by a child class")

    def get_jacobians(self, x, u, dt):
        raise NotImplementedError("get_jacobians must be overriden by a child class")

    # Notation: zf is unshifted equilibrium point
    def zfyf_to_zy(self, zf=None):
        """
        :zf: (N, n_z) or (n_z,) array
        :yf: (N, n_y) or (n_y,) array
        """
        if zf is not None and self.y_ref is not None:
            return zf - self.y_ref
        else:
            raise RuntimeError('Need to specify equilibrium point')

    def zy_to_zfyf(self, y=None):
        """
        :z: (N, n_z) or (n_z,) array
        :y: (N, n_y) or (n_y,) array
        """
        if y is not None and self.y_ref is not None:
            return y + self.y_ref
        else:
            raise RuntimeError('Need to specify equilibrium point')

    # x is reduced state => This function goes from reduced state to (shifted) observation
    # W_map expects (n, N) where n is the ROM state. W_map takes reduced to performance vars
    def x_to_zfyf(self, x, zf=True):
        """
        :x: (N, n_x) or (n_x,) array
        :zf: boolean
        :yf: boolean
        """
        return self.W_map(x.T).T + self.y_ref


    def x_to_zy(self, x):
        """
        :x: (N, n_x) or (n_x,) array
        :z: boolean
        :y: boolean
        """
        return self.W_map(x)

    def get_sim_params(self):
        return {'beta_weighting': self.beta_weighting, 'discr_method': self.discr_method,
                'dist_weights': self.dist_weights}

    def get_state_dim(self):
        return self.state_dim

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.obs_dim

    def rollout(self, x0, u, dt):
        """
        :x0: initial condition
        :u: array of control (N, n_u)
        :dt: time step

        Returns reduced order state x (N + 1, n_x) and performance variable z (N + 1, n_z)
        """
        N = u.shape[0]
        x = np.zeros((N + 1, self.state_dim))
        z_lin = np.zeros((N + 1, self.output_dim))

        # Set initial condition
        x[0,:] = x0

        # Simulate
        for i in range(N):
            x[i+1,:] = self.update_state(x[i,:], u[i,:], dt)
            z_lin[i,:] = self.update_observer_state(x[i, :])

        z = self.x_to_zfyf(x)

        return x, z

    def get_poly_basis(self, dim, order):
        zeta = sp.Matrix(sp.symbols('x1:{}'.format(dim + 1)))
        polynoms = sorted(itermonomials(list(zeta), order),
                          key=monomial_key('grevlex', list(reversed(zeta))))

        polynoms = polynoms[1:]
        return sp.lambdify(zeta, polynoms, modules=[jnp, jsp.special])

    # Continuous maps
    def reduced_dynamics(self, x, u):
        return jnp.dot(self.r_coeff, jnp.asarray(self.rom_phi(*x))) + jnp.dot(self.B_r, u) # jnp.dot(self.B_r, jnp.asarray(self.control_phi(*jnp.hstack([u, x]))))

    def reduced_to_output(self, x):
        return jnp.dot(jnp.asarray(self.C), jnp.dot(jnp.asarray(self.w_coeff), jnp.asarray(self.ssm_phi(*x))))

    @partial(jax.jit, static_argnums=(0,))
    def observed_to_reduced(self, y):
        if self.v_coeff is not None:
            return jnp.dot(jnp.asarray(self.v_coeff), jnp.asarray(self.chart_phi(*y)))
        else:
            return jnp.dot(np.transpose(self.V), y)

    def observed_to_reduced_nojit(self, y):
        if self.v_coeff is not None:
            return np.dot(self.v_coeff, np.asarray(self.chart_phi(*y)))
        else:
            return np.dot(np.transpose(self.V), y)

    # Discrete Map
    def reduced_dynamics_discrete(self, x, u):
        return jnp.dot(self.rd_coeff, jnp.asarray(self.rom_phi(*x))) + jnp.dot(self.Bd_r, u)


class SSMDynamics(SSM):
    """
    """
    def __init__(self, eq_point, discrete=False, discr_method='fe', C=None, **kwargs):
        super(SSMDynamics, self).__init__(eq_point, discrete=discrete, discr_method=discr_method, C=C, **kwargs)

    def update_state(self, x, u, dt):
        """
        Compute x+ based on a discretization time of dt.
        :x: current state
        :u: current input
        :dt: time step
        """
        A_d, B_d, d_d = self.get_jacobians(x, dt=dt, u=u)
        return self.update_dynamics(x, u, A_d, B_d, d_d)

    # Set self and f as non-traced elements that python handles each time they are changed
    @partial(jax.jit, static_argnums=(0,))
    def get_continuous_jacobians(self,
              x: jnp.ndarray,
              u: jnp.ndarray):
        A, B = jax.jacobian(self.maps['f_nl'], (0,1))(x, u)
        d = self.maps['f_nl'](x, u) - jnp.dot(A, x) - jnp.dot(B, u)
        return A, B, d

    @partial(jax.jit, static_argnums=(0,))
    def get_discrete_jacobians(self,
                               x: jnp.ndarray,
                               u: jnp.ndarray):
        A, B = jax.jacobian(self.maps['f_nl_d'], (0, 1))(x, u)
        d = self.maps['f_nl_d'](x, u) - jnp.dot(A, x) - jnp.dot(B, u)
        return A, B, d

    # TODO: Testing jax components
    @partial(jax.jit, static_argnums=(0,))
    def get_jacobians(self, x, u, dt):
        # x = x.reshape(self.state_dim, 1)
        # u = u.reshape(self.input_dim, 1)
        if not self.discrete:
            Ac, Bc, dc = self.get_continuous_jacobians(jnp.asarray(x), jnp.asarray(u))
            A, B, d = self.discretize_dynamics(Ac, Bc, dc, dt)
        else:
            A, B, d = self.get_discrete_jacobians(jnp.asarray(x), jnp.asarray(u))

        return A, B, d

    def get_jacobians_nojit(self, x, u, dt):
        if not self.discrete:
            Ac, Bc, dc = self.get_continuous_jacobians(jnp.asarray(x), jnp.asarray(u))
            A, B, d = self.discretize_dynamics(Ac, Bc, dc, dt)
        else:
            A, B, d = self.get_discrete_jacobians(jnp.asarray(x), jnp.asarray(u))

        return A, B, d

    # TODO: Testing jax capes
    @partial(jax.jit, static_argnums=(0,))
    def get_observer_jacobians(self,
                               x: jnp.ndarray):
        H = jax.jacobian(self.W_map, 0)(x)
        c_res = self.W_map(x) - jnp.dot(H, x)
        return H, c_res

    def get_observer_jacobians_nojit(self,
                               x: jnp.ndarray):
        # x = x.reshape(self.state_dim, 1)

        H = jax.jacobian(self.W_map, 0)(x)
        c_res = self.W_map(x) - jnp.dot(H, x)
        return H, c_res
    
    def get_obstacleConstraint_jacobians(self,
                                      x: jnp.ndarray, obs_center: jnp.ndarray):
        normFunc = partial(norm2Diff, y=obs_center)
        g = lambda x: normFunc(self.W_map(x))
        G = jax.jacobian(g)(x)
        b = g(x) - G @ x
        return G, b
        

    # def get_jacobians(self, x, dt=None, u=None):
    #     """
    #     Extract the Jacobians A, B, d (or A_d, B_d, d_d) at the state x
    #     :x: reduced state
    #     """
    #     assert u is not None, 'Need to supply current input'
    #     x = x.reshape(self.state_dim, 1)
    #     u = u.reshape(self.input_dim, 1)
    #     if self.discrete:
    #         A = self.maps['A_d'](x)
    #         B = self.maps['B_d'](x)
    #         f_nl = self.maps['f_nl_d'](x, u)
    #         d = f_nl - A@x - B@u
    #     else:
    #         A = self.maps['A'](x)
    #         B = self.maps['B'](x)
    #         f_nl = self.maps['f_nl'](x, u)
    #         d = f_nl - A @ x - B @ u
    #         if dt is not None:
    #             A, B, d = self.discretize_dynamics(A, B, d, dt)
    #
    #     return A, B, np.squeeze(d)

    # def get_observer_jacobians(self, x, dt=None, u=None):
    #     """
    #     Extract the Jacobians H, c at the state x
    #     :x: reduced state
    #     """
    #     x = x.reshape(self.state_dim, 1)
    #     H = self.maps['H'](x)
    #     c_nl = self.W_map(x)
    #     c_res = c_nl - H @ x
    #     return H, c_res

    def update_observer_state(self, x, dt=None, u=None):
        # TODO: Testing jax capes
        # H, c = self.get_observer_jacobians(x, dt=dt, u=u)
        # return np.squeeze(H @ x) + np.squeeze(c)

        H, c = self.get_observer_jacobians(x)
        return np.squeeze(jnp.dot(H, x)) + np.squeeze(c)

    def discretize_dynamics(self, A_c, B_c, d_c, dt):
        if self.discr_method == 'fe':
            A_d = jnp.eye(A_c.shape[0]) + dt * A_c
            B_d = dt * B_c
            d_d = dt * d_c

        elif self.discr_method == 'be':
            A_d = jnp.linalg.inv(np.eye(A_c.shape[0]) - dt * A_c)
            sep_term = jnp.dot(jnp.linalg.inv(A_c), (A_d - jnp.eye(A_c.shape[0])))
            B_d = jnp.dot(sep_term, B_c)
            d_d = jnp.dot(sep_term, d_c)

        elif self.discr_method == 'bil':
            A_d = jnp.dot((jnp.eye(A_c.shape[0]) + 0.5 * dt * A_c), jnp.linalg.inv(jnp.eye(A_c.shape[0])
                                                                               - 0.5 * dt * A_c))
            sep_term = jnp.dot(jnp.linalg.inv(A_c), (A_d - jnp.eye(A_c.shape[0])))
            B_d = jnp.dot(sep_term, B_c)
            d_d = jnp.dot(sep_term, d_c)

        else:
            raise RuntimeError('self.discr_method must be in [fe, be, bil, zoh]')

        return A_d, B_d, d_d

    # def discretize_dynamics(self, A_c, B_c, d_c, dt):
    #     if self.discr_method == 'fe':
    #         A_d = np.eye(A_c.shape[0]) + dt * A_c
    #         B_d = dt * B_c
    #         d_d = dt * d_c
    #
    #     elif self.discr_method == 'be':
    #         A_d = np.linalg.inv(np.eye(A_c.shape[0]) - dt * A_c)
    #         sep_term = np.linalg.inv(A_c) @ (A_d - np.eye(A_c.shape[0]))
    #         B_d = sep_term @ B_c
    #         d_d = sep_term @ d_c
    #
    #     elif self.discr_method == 'bil':
    #         A_d = (np.eye(A_c.shape[0]) + 0.5 * dt * A_c) @ np.linalg.inv(np.eye(A_c.shape[0])
    #                                                                            - 0.5 * dt * A_c)
    #         sep_term = np.linalg.inv(A_c) @ (A_d - np.eye(A_c.shape[0]))
    #         B_d = sep_term @ B_c
    #         d_d = sep_term @ d_c
    #
    #     elif self.discr_method == 'zoh':
    #         A_d, B_d, d_d = scutils.zoh_affine(A_c, B_c, d_c, dt)
    #
    #     else:
    #         raise RuntimeError('self.discr_method must be in [fe, be, bil, zoh]')
    #
    #     return A_d, B_d, d_d

    @staticmethod
    def update_dynamics(x, u, A_d, B_d, d_d):
        x_next = np.squeeze(A_d @ x) + np.squeeze(B_d @ u) + np.squeeze(d_d)
        return x_next

    def get_ref_point(self):
        return self.y_ref

    def compute_RO_state(self, y):
        """
        Compute reduced order projection of vector
        :param zf: position and velocity of observed state
        :return: Reduced order vector
        """
        #TODO: This will introduce bugs. Fix this
        if self.delays == 0:
            return self.V_map(y - self.y_ref)
        else:
            return np.transpose(self.V) @ (y - self.y_ref)