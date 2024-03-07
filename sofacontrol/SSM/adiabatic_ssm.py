import numpy as np
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
import sympy as sp
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
import pickle

from .interpolators import InterpolatorFactory


INTERPOLATION_METHOD = "modified_idw" # "qp", "modified_idw", "linear", "ct", "nn", "idw"
ORIGIN_IDX = 0


class AdiabaticSSM:
    """
    Provided a data dictionary or file location this class can give Jacobians w.r.t state, input and affine term or
    forward simulate the dynamics (i.e. perform rollout)
    """

    def __init__(self, eq_point, models, params, discrete=False, discr_method='fe', C=None, zf_target=None, **kwargs):
        self.maps = {}

        self.discrete = discrete
        self.discr_method = discr_method

        self.interp_method = INTERPOLATION_METHOD

        self.models = models
        self.params = params

        self.adiabatic = True
        # if self.interp_method in ["idw", "modified_idw"]: # "krg", "rbf", "tps", "nn"]:
        #     self.interp_3d = True
        # else:
        #     self.interp_3d = False
        if self.interp_method in ["idw", "modified_idw", "nn"]:
            self.interp_slice = np.s_[:]
        elif self.interp_method in ["krg", "rbf", "tps"]:
            self.interp_slice = np.s_[:3]
        else:
            self.interp_slice = np.s_[:2]
        
        # Model dimensions
        self.state_dim = self.params['state_dim']
        self.input_dim = self.params['input_dim']
        self.output_dim = int(self.params['output_dim']) # This is also performance dimension
        self.SSM_order = self.params['SSM_order']
        self.ROM_order = self.params['ROM_order']
        self.delays = self.params['delays']
        self.obs_dim = self.params['obs_dim']

        # SSM basis functions (polynomials)
        self.rom_phi = self.get_poly_basis(self.state_dim, self.ROM_order)
        self.ssm_map_phi = self.get_poly_basis(self.state_dim, self.SSM_order)
        self.ssm_chart_phi = self.get_poly_basis(self.obs_dim, self.SSM_order)
        self.control_phi = self.get_poly_basis(self.input_dim, self.params['u_order'])

        # Observation model
        if C is not None:
            self.C = C
            assert np.shape(self.C) == (self.output_dim, self.obs_dim)
        else:
            # When we learn mappings to output variables directly (no time-delays)
            self.C = np.eye(self.obs_dim, self.obs_dim)
        
        # adiabatic SSM interpolation
        self.V, self.w_coeff, self.v_coeff, self.r_coeff, self.B_r, self.q_bar, self.u_bar = [], [], [], [], [], [], []
        for model in self.models:
            # Continuous-time model
            self.V.append(model['V'])               # tangent space
            self.w_coeff.append(model['w_coeff'])   # reduced to observed
            self.v_coeff.append(model['v_coeff'])   # observed to reduced
            self.r_coeff.append(model['r_coeff'])   # reduced coefficients
            self.B_r.append(model['B'])             # reduced control matrix
            self.q_bar.append(model['q_eq'])
            self.u_bar.append(model['u_eq'])
        self.coeff_dict = {
                    'w_coeff': self.w_coeff,
                    'V': self.V,
                    'r_coeff': self.r_coeff,
                    'B_r': self.B_r,
                    'u_bar': self.u_bar,
                    'q_bar': self.q_bar
                }
        if self.v_coeff[0] is not None:
            self.coeff_dict['V'] = self.v_coeff

        self.interpolator = InterpolatorFactory(self.interp_method, [(self.V[0].T @ np.tile(q, 5))[self.interp_slice] for q in self.q_bar], self.coeff_dict).get_interpolator()

        # Manifold parametrization
        self.W_map = self.reduced_to_output
        self.V_map = self.observed_to_reduced

        # Continuous reduced dynamics
        self.maps['f_nl'] = self.reduced_dynamics

        # Set reference equilibrium point
        self.y_eq = eq_point
        self.y_ref = self.y_eq
        # self.y_ref = np.tile(self.y_eq, self.delays + 1)

        # Set performance to zero matrix with appropriate dimension (n_z, n_x)
        self.H = np.zeros((self.output_dim, self.state_dim))
        self.nonlinear_observer = True

    def update_state(self, x, u, dt):
        raise NotImplementedError("update_state must be overriden by a child class")

    def get_jacobians(self, x, u, dt):
        raise NotImplementedError("get_jacobians must be overriden by a child class")

    # Notation: zf is unshifted equilibrium point
    def zfyf_to_zy(self, zf):
        """
        :zf: (N, n_z) or (n_z,) array
        :yf: (N, n_y) or (n_y,) array
        """
        if self.y_ref is not None:
            zy = zf - self.y_ref
            return zy
        else:
            raise RuntimeError('Need to specify equilibrium point')

    def zy_to_zfyf(self, y):
        """
        :z: (N, n_z) or (n_z,) array
        :y: (N, n_y) or (n_y,) array
        """
        if self.y_ref is not None:
            zfyf = y + self.y_ref
            return zfyf
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
        W = self.interpolator.transform(x[0, self.interp_slice], "w_coeff")
        x_bar = self.V[0].T @ np.tile(self.interpolator.transform(x[0, self.interp_slice], "q_bar"), 5)
        y_bar = np.tile(self.interpolator.transform(x[0, self.interp_slice], "q_bar"), 5)
        return self.W_map(W, x.T, x_bar, y_bar).T + self.y_ref


    def x_to_zy(self, x):
        """
        :x: (N, n_x) or (n_x,) array
        :z: boolean
        :y: boolean
        """
        W = self.interpolator.transform(x[0, self.interp_slice], "w_coeff")
        x_bar = self.V[0].T @ np.tile(self.interpolator.transform(x[0, self.interp_slice], "q_bar"), 5)
        y_bar = np.tile(self.interpolator.transform(x[0, self.interp_slice], "q_bar"), 5)
        return self.W_map(W, x.T, x_bar, y_bar)

    def get_sim_params(self):
        return {'discr_method': self.discr_method}

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
    def reduced_dynamics(self, R, B, x, u, x_bar, u_bar):
        xdot = jnp.dot(R, jnp.asarray(self.rom_phi(*x - x_bar))) + jnp.dot(B, jnp.asarray(self.control_phi(*(u - u_bar))))
        return xdot

    def reduced_to_output(self, W, x, x_bar, y_bar):
        output = jnp.dot(jnp.asarray(self.C), (jnp.dot(jnp.asarray(W), jnp.asarray(self.ssm_map_phi(*(x.T - x_bar).T))).T + y_bar).T)
        return output

    # @partial(jax.jit, static_argnums=(0,))
    # def observed_to_reduced(self, y):
    #     # memorize the observation for interpolation (adiabatic framework)
    #     self.last_observation_y = y
    #     if self.v_coeff[0] is not None:
    #         return jnp.dot(self.interpolate_coeffs(y[-3:-1], 'v_coeff'), jnp.asarray(self.ssm_chart_phi(*(y - jnp.tile(self.interpolate_coeffs(y[-3:-1], 'q_bar'), 5)))))
    #     else:
    #         return jnp.dot(jnp.transpose(self.interpolate_coeffs(self.last_observation_y[-3:-1], 'V')), y - jnp.tile(self.interpolate_coeffs(y[-3:-1], 'q_bar'), 5))
    
    # no jit
    def observed_to_reduced(self, V, y):
        # if not jnp.allclose(y, self.last_observation_y):
        #     with open("/home/jonas/Projects/stanford/soft-robot-control/examples/trunk/y_last_obs.pkl", "wb") as f:
        #         pickle.dump(y, f)
        #     # if self.interp_3d:
        #     #     xy_z = y[-3:]
        #     # else:
        #     #     xy_z = y[-3:-1]
        #     xy_z = jnp.dot(jnp.transpose(V), y - y_bar)
        #     self.y_bar_current = np.tile(self.interpolator.transform(xy_z, 'q_bar'), 5) # np.concatenate([self.interpolator.transform(xy_z, 'q_bar'), np.zeros(3)]) # 
        #     self.u_bar_current = self.interpolator.transform(xy_z, 'u_bar')
        #     self.B_r_current = self.interpolator.transform(xy_z, 'B_r')
        #     self.R_current = self.interpolator.transform(xy_z, 'r_coeff')
        #     self.V_current = self.interpolator.transform(xy_z, 'V')
        #     self.W_current = self.interpolator.transform(xy_z, 'w_coeff')
        return jnp.dot(jnp.transpose(V), y) # jnp.dot(V, jnp.asarray(self.ssm_chart_phi(*(y - y_bar)))) # 


class AdiabaticSSMDynamics(AdiabaticSSM):
    def __init__(self, eq_point, models, params, discrete=False, discr_method='fe', C=None, **kwargs):
        super(AdiabaticSSMDynamics, self).__init__(eq_point, models, params, discrete=discrete, discr_method=discr_method, C=C, **kwargs)

    def update_state(self, x, u, dt):
        """
        Compute x+ based on a discretization time of dt.
        :x: current state
        :u: current input
        :dt: time step
        """
        R = self.interpolator.transform(x[self.interp_slice], 'r_coeff')
        B_r = self.interpolator.transform(x[self.interp_slice], 'B_r')
        u_bar = self.interpolator.transform(x[self.interp_slice], 'u_bar')
        x_bar = self.V[0].T @ np.tile(self.interpolator.transform(x[self.interp_slice], 'q_bar'), 5)
        A_d, B_d, d_d = self.get_jacobians(x, u, dt, R, B_r, x_bar, u_bar) # self.R_current, self.B_r_current, self.u_bar_current)
        return self.update_dynamics(x, u, A_d, B_d, d_d)

    # Set self and f as non-traced elements that python handles each time they are changed
    @partial(jax.jit, static_argnums=(0,))
    def get_continuous_jacobians(self,
            x: jnp.ndarray,
            u: jnp.ndarray,
            R: jnp.ndarray,
            B_r: jnp.ndarray,
            x_bar: jnp.ndarray,
            u_bar: jnp.ndarray):
        A, B = jax.jacobian(self.maps['f_nl'], (2, 3))(R, B_r, x, u, x_bar, u_bar)
        d = self.maps['f_nl'](R, B_r, x, u, x_bar, u_bar) - jnp.dot(A, x) - jnp.dot(B, u)
        return A, B, d

    # @partial(jax.jit, static_argnums=(0,))
    # def get_discrete_jacobians(self,
    #                         x: jnp.ndarray,
    #                         u: jnp.ndarray):
    #     A, B = jax.jacobian(self.maps['f_nl_d'], (0, 1))(x, u)
    #     d = self.maps['f_nl_d'](x, u) - jnp.dot(A, x) - jnp.dot(B, u)
    #     return A, B, d

    @partial(jax.jit, static_argnums=(0,))
    def get_jacobians(self, x, u, dt, R, B_r, x_bar, u_bar):
        # if not self.discrete:
        Ac, Bc, dc = self.get_continuous_jacobians(jnp.asarray(x), jnp.asarray(u), R, B_r, x_bar, u_bar)
        A, B, d = self.discretize_dynamics(Ac, Bc, dc, dt)
        # else:
        #     A, B, d = self.get_discrete_jacobians(jnp.asarray(x), jnp.asarray(u))
        return A, B, d

    # def get_jacobians_nojit(self, x, u, dt):
    #     if not self.discrete:
    #         Ac, Bc, dc = self.get_continuous_jacobians(jnp.asarray(x), jnp.asarray(u))
    #         A, B, d = self.discretize_dynamics(Ac, Bc, dc, dt)
    #     else:
    #         A, B, d = self.get_discrete_jacobians(jnp.asarray(x), jnp.asarray(u))
    #     return A, B, d

    @partial(jax.jit, static_argnums=(0,))
    def get_observer_jacobians(self,
                            x: jnp.ndarray,
                            W: jnp.ndarray,
                            x_bar: jnp.ndarray,
                            y_bar: jnp.ndarray):
        H = jax.jacobian(self.W_map, 1)(W, x, x_bar, y_bar)
        c_res = self.W_map(W, x, x_bar, y_bar) - jnp.dot(H, x)
        return H, c_res

    # def get_observer_jacobians_nojit(self,
    #                         x: jnp.ndarray):
    #     # x = x.reshape(self.state_dim, 1)
    #     H = jax.jacobian(self.W_map, 0)(x)
    #     c_res = self.W_map(x) - jnp.dot(H, x)
    #     return H, c_res

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
        W = self.interpolator.transform(x[self.interp_slice], "w_coeff")
        x_bar = self.V[0].T @ np.tile(self.interpolator.transform(x[self.interp_slice], "q_bar"), 5)
        y_bar = np.tile(self.interpolator.transform(x[self.interp_slice], "q_bar"), 5)
        H, c = self.get_observer_jacobians(x, W, x_bar, y_bar) # self.W_current, self.y_bar_current)
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

    # @staticmethod
    def update_dynamics(self, x, u, A_d, B_d, d_d):
        x_next = np.squeeze(A_d @ x) + np.squeeze(B_d @ u) + np.squeeze(d_d)
        return x_next

    def get_ref_point(self):
        return self.y_ref

    # def compute_RO_state(self, y):
    #     """
    #     Compute reduced order projection of vector
    #     :param zf: position and velocity of observed state
    #     :return: Reduced order vector
    #     """
    #     #TODO: This will introduce bugs. Fix this
    #     if self.delays == 0:
    #         return self.V_map(y - self.y_ref)
    #     else:
    #         return np.transpose(self.V) @ (y - self.y_ref)