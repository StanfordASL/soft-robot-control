import numpy as np
from scipy.interpolate import interp1d
import sofacontrol.utils as scutils
import sofacontrol.tpwl.tpwl_utils as tutils
from sofacontrol.mor import pod
import jax.numpy as jnp
from functools import partial
from sofacontrol.utils import norm2Diff
import jax


###  DEFAULT VALUES
DISCR_METHOD = 'zoh'  # other options: be, zoh, bil. Forward Euler, Backward Euler, Bilinear transf. or Zero-Order Hold
TPWL_METHOD = 'nn'  # other options: weighting. Nearest neighbor or exponential weighting

DISCR_DICT = {'fe': 'forward Euler', 'be': 'implicit Euler', 'bil': 'bilinear transform', 'zoh': 'zero-order hold'}


class TPWL:
    """
    Provided a data dictionary or file location this class can give Jacobians w.r.t state, input and affine term or
    forward simulate the dynamics (i.e. perform rollout)
    """

    def __init__(self, data, params=None, Cf=None, Hf=None, **kwargs):
        # Define dictionary of data that includes the TPWL points
        if isinstance(data, dict):
            self.tpwl_dict = data
        else:
            # In case a file location is passed in
            self.tpwl_dict = scutils.load_data(data)
        self.num_points = len(self.tpwl_dict['q'])
        self.discr_method = params.get('discr_method', DISCR_METHOD)

        # Build ROM object in case it is needed
        if self.tpwl_dict['rom_info']['type'] == 'POD':
            self.rom = pod.POD(self.tpwl_dict['rom_info'])
        else:
            raise NotImplementedError("Unknown ROM type")

        # Get state and input dimensions
        try:
            self.state_dim = (self.tpwl_dict['q'][0].shape[-1] * 2) if self.tpwl_dict is not None else None
            self.input_dim = self.tpwl_dict['u'][0].shape[-1] if self.tpwl_dict is not None else None

        except IndexError:
            self.state_dim = (self.tpwl_dict['q'].shape[-1] * 2) if self.tpwl_dict is not None else None
            self.input_dim = self.tpwl_dict['u'].shape[-1] if self.tpwl_dict is not None else None

        if params is None:
            params = dict()
        # Get configuration parameters for the model
        self.tpwl_method = params.get('tpwl_method', TPWL_METHOD)
        self.beta_weighting = params.get('beta_weighting', None)
        self.dist_weights = params.get('dist_weights')

        # Optionally set output and measurement models
        if Cf is not None:
            self.set_measurement_model(Cf)
        else:
            self.C = None
            self.y_ref = None
            self.meas_dim = None

        if Hf is not None:
            self.set_output_model(Hf)
        else:
            self.H = None
            self.z_ref = None
            self.output_dim = None

        self.nonlinear_observer = False
        # Variables for precomputing discrete time matrices if desired
        self.pre_discretized_dt = None
        self.A_d = None
        self.B_d = None
        self.d_d = None

    def update_state(self, x, u, dt):
        raise NotImplementedError("update_state must be overriden by a child class")

    def get_jacobians(self, x, dt=None):
        raise NotImplementedError("get_jacobians must be overriden by a child class")

    def set_measurement_model(self, Cf):
        self.C = Cf @ self.rom.V
        self.y_ref = Cf @ self.rom.x_ref
        self.meas_dim = self.C.shape[0]

    def set_output_model(self, Hf):
        self.H = Hf @ self.rom.V
        self.z_ref = Hf @ self.rom.x_ref
        self.output_dim = self.H.shape[0]

    def zfyf_to_zy(self, zf=None, yf=None):
        """
        :zf: (N, n_z) or (n_z,) array
        :yf: (N, n_y) or (n_y,) array
        """
        if zf is not None and self.z_ref is not None:
            return zf - self.z_ref
        elif yf is not None and self.y_ref is not None:
            return yf - self.y_ref
        else:
            raise RuntimeError('Need to set output or meas. model')

    def zy_to_zfyf(self, z=None, y=None):
        """
        :z: (N, n_z) or (n_z,) array
        :y: (N, n_y) or (n_y,) array
        """
        if z is not None and self.z_ref is not None:
            return z + self.z_ref
        elif y is not None and self.y_ref is not None:
            return y + self.y_ref
        else:
            raise RuntimeError('Need to set output or meas. model')

    def x_to_zfyf(self, x, zf=False, yf=False):
        """
        :x: (N, n_x) or (n_x,) array
        :zf: boolean
        :yf: boolean
        """
        if zf and self.H is not None:
            return np.transpose(self.H @ x.T) + self.z_ref
        elif yf and self.C is not None:
            return np.transpose(self.C @ x.T) + self.y_ref
        else:
            raise RuntimeError('Need to set output or meas. model')

    def x_to_zy(self, x, z=False, y=False):
        """
        :x: (N, n_x) or (n_x,) array
        :z: boolean
        :y: boolean
        """
        if z and self.H is not None:
            return np.transpose(self.H @ x.T)
        elif y and self.C is not None:
            return np.transpose(self.C @ y.T)
        else:
            raise RuntimeError('Need to set output or meas. model')

    def get_state_dim(self):
        return self.state_dim

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    def get_meas_dim(self):
        return self.meas_dim

    def get_rom_info(self):
        return self.tpwl_dict['rom_info']

    def get_sim_params(self):
        return {'beta_weighting': self.beta_weighting, 'discr_method': self.discr_method,
                'tpwl_method': self.tpwl_method, 'dist_weights': self.dist_weights}

    def calc_nearest_point(self, x):
        """
        Calculates distances from point to all ref points and weights them by dist_weights, returns 
        the index of the point with the smallest distance defined by d = qdist + vdist
        """
        q, v = scutils.x2qv(x)
        q_dist = self.dist_weights['q']*np.linalg.norm(self.tpwl_dict['q'] - q, axis=1)
        v_dist = self.dist_weights['v']*np.linalg.norm(self.tpwl_dict['v'] - v, axis=1)
        return np.argmin(q_dist + v_dist)

    def calc_weighting_factors(self, x):
        """
        Calculates the weights for each ref. point in the TPWL model with respect to the current point
        """
        q, v = scutils.x2qv(x)
        q_dist = self.dist_weights['q']*np.linalg.norm(self.tpwl_dict['q'] - q, axis=1)
        v_dist = self.dist_weights['v']*np.linalg.norm(self.tpwl_dict['v'] - v, axis=1)
        dist_comb = q_dist + v_dist
        m_idx = np.argmin(dist_comb)
        m = dist_comb[m_idx] # minimum distance

        # If the minimum is 0 then just take that point
        if m == 0:
            weights_norm = np.zeros(np.shape(dist_comb))
            weights_norm[m_idx] = 1

        # Otherwise compute all weights
        else:
            weights = np.exp(-self.beta_weighting * dist_comb / m)
            weights_norm = weights / np.sum(weights)

        return weights_norm

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

        if self.H is not None:
            z = self.x_to_zfyf(x, zf=True)
        else:
            z = None

        return x, z


class TPWLATV(TPWL):
    """
    """
    def __init__(self, data, params=None, Cf=None, Hf=None, **kwargs):
        super(TPWLATV, self).__init__(data, params, Cf=Cf, Hf=Hf, **kwargs)
        self.ref_point = None

    def update_state(self, x, u, dt):
        """
        Compute x+ based on a discretization time of dt.
        :x: current state
        :u: current input
        :dt: time step (set to -1 if pre_discretize(dt) as already been called and tpwl_method = 'nn')
        """
        A_d, B_d, d_d = self.get_jacobians(x, dt)
        return self.update_dynamics(x, u, A_d, B_d, d_d)

    def get_jacobians(self, x, dt=None, u=None):
        """
        Extract the Jacobians A, B, d (or A_d, B_d, d_d) at the state x
        :x: state
        :dt: (optional) do not specify to extract continuous time Jacobians, specify as
             timestep for discrete time Jacobians, if prediscretized with same timestep, will use
             prediscretized value
        """
        if self.tpwl_method == 'weighting':
            weights = self.calc_weighting_factors(x)
            A = np.einsum("i, ijk -> jk", weights, self.tpwl_dict['A_c'])
            B = np.einsum("i, ijk -> jk", weights, self.tpwl_dict['B_c'])
            d = np.einsum("i, ij -> j", weights, self.tpwl_dict['d_c'])
            if dt is not None:
                A, B, d = self.discretize_dynamics(A, B, d, dt)

        elif self.tpwl_method == 'nn':
            self.ref_point = self.calc_nearest_point(x)

            # If dt = None, this defaults to false
            if self.pre_discretized_dt is not None and dt == self.pre_discretized_dt:
                A = self.A_d[self.ref_point]
                B = self.B_d[self.ref_point]
                d = self.d_d[self.ref_point]
            else:
                A = self.tpwl_dict['A_c'][self.ref_point]
                B = self.tpwl_dict['B_c'][self.ref_point]
                d = self.tpwl_dict['d_c'][self.ref_point]
                if dt is not None:
                    A, B, d = self.discretize_dynamics(A, B, d, dt)

        else:
            raise RuntimeError('tpwl method should be nn or weighting')

        return A, B, d
    
    def get_obstacleConstraint_jacobians(self,
                                      x: jnp.ndarray, obs_center: jnp.ndarray):
        normFunc = partial(norm2Diff, y=obs_center)
        g = lambda x: normFunc((self.H @ x)[3:]) # TODO: Assumes that observables are in format (vel, pos)
        G = jax.jacobian(g)(x)
        b = g(x) - G @ x
        return G, b

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

    def pre_discretize(self, dt):
        """
        Discretizes all points in the TPWL database and stores them. To use
        these precomputed values in either update_state or get_jacobians
        simply pass in dt corresponding to self.prediscretized_dt
        """
        # Only makes sense to do this if tpwl method is 'nn'
        if self.tpwl_method != 'nn':
            raise RuntimeError('tpwl method should be nn to pre-discretize')

        print('Performing pre-discretization using {} of TPWL model with dt = {:.3f}'
              .format(DISCR_DICT[self.discr_method], dt))
        self.A_d = []
        self.B_d = []
        self.d_d = []
        for i in range(self.num_points):
            A_d, B_d, d_d = self.discretize_dynamics(self.tpwl_dict['A_c'][i], 
                                                     self.tpwl_dict['B_c'][i], 
                                                     self.tpwl_dict['d_c'][i], dt)
            self.A_d.append(A_d)
            self.B_d.append(B_d)
            self.d_d.append(d_d)

        self.pre_discretized_dt = dt

    def get_characteristic_dx(self, dt):
        """
        Computed a characteristic quantity for x_{k+1} - x_k based on time step dt
        based on the TPWL points
        """
        x = scutils.qv2x(self.tpwl_dict['q'], self.tpwl_dict['v'])
        dx = np.zeros(x.shape)
        for i in range(x.shape[0]):
            dx[i,:] = self.update_state(x[i,:], self.tpwl_dict['u'][i,:], dt) - x[i,:]
        dx_char = np.abs(dx).max(axis=0)
        return dx_char

    @staticmethod
    def update_dynamics(x, u, A_d, B_d, d_d):
        x_next = A_d @ x + np.squeeze(B_d @ u) + d_d
        return x_next

    def get_ref_point(self):
        return self.ref_point
