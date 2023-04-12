import numpy as np
import sofacontrol.utils as scutils
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
import sympy as sp
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
from sofacontrol.utils import norm2, blockDiagonalize
from scipy.linalg import solve_continuous_lyapunov


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
        self.robustParams = kwargs.pop('robustParams', None) # Expect dictionary {lambda_n, lambda_r, L_n, L_r, Bnorm}

        self.state_dim = self.params['state_dim'][0, 0][0, 0]
        self.n_x = self.state_dim  # This is for SSM reduced dynamics only
        self.input_dim = self.params['input_dim'][0, 0][0, 0]
        self.output_dim = int(self.params['output_dim'][0, 0][0, 0]) #This is also performance dimension
        self.n_z = self.output_dim  # This is pure output only
        self.SSM_order = self.params['SSM_order'][0, 0][0, 0]
        self.ROM_order = self.params['ROM_order'][0, 0][0, 0]
        self.Ts = self.model['Ts'][0, 0][0, 0]
        # TODO: This is new
        self.delays = self.params['delays'][0, 0][0, 0]
        self.obs_dim = self.params['obs_dim'][0, 0][0, 0]

        self.rom_phi = self.get_poly_basis(self.state_dim, self.ROM_order)
        self.ssm_phi = self.get_poly_basis(self.state_dim, self.SSM_order)

        # Continuous-time model
        self.V = self.model['V'][0, 0] # Tangent space TODO: this is new
        self.w_coeff = self.model['w_coeff'][0, 0] # reduced to observed

        self.v_coeff = self.model['v_coeff'][0, 0]  # observed to reduced
        if len(self.v_coeff) == 0:
            self.v_coeff = None

        self.r_coeff = self.model['r_coeff'][0, 0] # reduced coefficients
        self.B_r = self.model['B'][0, 0] #reduced control matrix

        # self.Bn = self.model['Bn'][0, 0]

        # Discrete-time model
        if discrete:
            self.rd_coeff = self.model['rd_coeff'][0, 0]  # reduced coefficients
            self.Bd_r = self.model['Bd'][0, 0]  # reduced control matrix

        # Manifold parametrization
        self.W_map = self.reduced_to_output
        self.W_map_traj = self.reduced_to_output_traj
        self.V_map = self.observed_to_reduced

        # Continuous reduced dynamics
        if self.robustParams is not None:
            self.lambda_n = self.robustParams['lambda_n']
            self.lambda_r = self.robustParams['lambda_r']
            self.L_n = self.robustParams['L_n']
            self.L_r = self.robustParams['L_r']
            self.L_b = self.robustParams['L_b']
            self.Bnorm = self.robustParams['Bnorm']
            self.d = self.robustParams['d']
            self.robust = True
            # Set performance to zero matrix with appropriate dimension (n_z, n_x)
            self.maps['f_nl'] = self.robust_reduced_dynamics
            self.state_dim = self.state_dim + 2 # Add dimension due to tube dynamics
            self.output_dim = self.output_dim + 2
            # Define positive definite P
            Ar = self.r_coeff[:self.n_x, :self.n_x]
            self.P = solve_continuous_lyapunov(np.transpose(Ar), -np.eye(self.n_x))

            # TODO: Add this later
            self.T, _ = blockDiagonalize(self.r_coeff[:self.n_x, :self.n_x]) # Change of basis for diagonalization

            # TODO: Add retrieval of Bn. Comment it out when using the learned dynamics
            # self.Lwnl = self.robustParams['Lwnl']
        else:
            self.robust = False
            self.maps['f_nl'] = self.reduced_dynamics
            self.T = np.eye(self.n_x)

        # Observation model
        if C is not None:
            self.C = C
            assert np.shape(self.C) == (self.output_dim, self.obs_dim)
        else:
            # When we learn mappings to output variables directly (no time-delays)
            self.C = np.eye(self.obs_dim, self.obs_dim)

        # Set performance to zero matrix with appropriate dimension (n_z, n_x)
        self.H = np.zeros((self.output_dim, self.state_dim))

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
        x = x[:, :self.n_x]
        return self.W_map_traj(x.T).T + self.y_ref


    def x_to_zy(self, x):
        """
        :x: (N, n_x) or (n_x,) array
        :z: boolean
        :y: boolean
        """
        x = x[:, :self.n_x]
        return self.W_map_traj(x.T).T

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
        x = np.zeros((N + 1, np.size(x0)))
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
        return jnp.dot(self.r_coeff, jnp.asarray(self.rom_phi(*x))) + jnp.dot(self.B_r, u)

    # TODO: Lump tube dynamics with normal dynamics. Might be breaking other functions because now we have augmented states
    # TODO: Make it standard to just evaluate the reduced states up to the reduced dimension
    # TODO: Need to handle special case where norm(u) = 0
    def robust_reduced_dynamics(self, x, u):
        # TODO: Learned B matrix
        T = jnp.asarray(self.T)
        xr = jnp.linalg.inv(T) @ x[:self.n_x]
        xdot = jnp.hstack((jnp.dot(T, jnp.dot(self.r_coeff, jnp.asarray(self.rom_phi(*xr))) + jnp.dot(self.B_r, u)),
                          -(self.lambda_n - self.L_b) * x[self.n_x] + self.Bnorm * norm2(u) + self.d,
                           -(self.lambda_r - self.L_r) * x[self.n_x + 1] + self.L_n * x[self.n_x] + self.d))

        # TODO: Known B matrix
        # xdot = jnp.hstack((jnp.dot(self.r_coeff, jnp.asarray(self.rom_phi(*x[0:self.n_x]))) + jnp.dot(self.B_r, u),
        #                    -(self.lambda_n - self.L_n) * x[self.n_x] + (1 + self.Lwnl) * (self.L_n * x[self.n_x] + self.d) + norm2(self.Bn @ u) + self.Lwnl * norm2(self.B_r @ u),
        #                    -(self.lambda_r - self.L_r) * x[self.n_x + 1] + self.L_n * x[self.n_x] + self.d))

        return xdot

    # TODO: Pad the last two entries with 0 if we are in robust case
    @partial(jax.jit, static_argnums=(0,))
    def reduced_to_output(self, x):
        T = jnp.asarray(self.T)
        xr = jnp.linalg.inv(T) @ x[:self.n_x]
        if self.robust:
            return jnp.hstack((jnp.dot(jnp.asarray(self.C), jnp.dot(jnp.asarray(self.w_coeff),
                                                               jnp.asarray(self.ssm_phi(*xr)))), 0., 0.))
        else:
            return jnp.dot(jnp.asarray(self.C), jnp.dot(jnp.asarray(self.w_coeff),
                                                        jnp.asarray(self.ssm_phi(*x[0:self.n_x]))))
    def reduced_to_output_traj(self, x):
        T = jnp.asarray(self.T)
        xr = jnp.linalg.inv(T) @ x[:self.n_x]
        if self.robust:
            return jnp.vstack((jnp.dot(jnp.asarray(self.C), jnp.dot(jnp.asarray(self.w_coeff),
                                                               jnp.asarray(self.ssm_phi(*xr)))), jnp.zeros((2, x.shape[1]))))
        else:
            return jnp.dot(jnp.asarray(self.C), jnp.dot(jnp.asarray(self.w_coeff),
                                                                    jnp.asarray(self.ssm_phi(*x))))

    @partial(jax.jit, static_argnums=(0,))
    def observed_to_reduced(self, y):
        if self.v_coeff is not None:
            return jnp.dot(self.v_coeff, jnp.asarray(self.ssm_phi(*y)))
        else:
            return jnp.dot(np.transpose(self.V), y)

    def observed_to_reduced_nojit(self, y):
        if self.v_coeff is not None:
            return np.dot(self.v_coeff, np.asarray(self.ssm_phi(*y)))
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

    def update_observer_state(self, x, dt=None, u=None):
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
        # TODO: This will introduce bugs. Fix this
        # TODO: Make sure we do the same for delays
        if self.delays == 0:
            return self.V_map(y - self.y_ref[:-2]) if self.robust else self.V_map(y - self.y_ref)
        else:
            return np.transpose(self.V) @ (y - self.y_ref)