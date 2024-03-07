import time
import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np

class FullStateObserver:
    """
    Default observer class, assumes full state perfect measurement
    :n_x: dimension of the state
    :H: (n_z x n_x) output matrix
    """
    def __init__(self, n_x, H=None):
        self.x = None
        self.z = None
        self.meas_dim = n_x
        self.state_dim = n_x
        self.H = H

    def get_meas_dim(self):
        return self.meas_dim

    def get_observer_params(self):
        return {'meas_dim': self.meas_dim, 'state_dim': self.state_dim}

    def update(self, u, y, dt, x=None):
        """
        Full state measurements assumed
        """
        self.x = x
        if self.H is not None:
            self.z = self.H @ x
        else:
            self.z = x

class SSMObserver:
    def __init__(self, dyn_sys):
        self.z = None
        self.x = None
        self.dyn_sys = dyn_sys

    def update(self, u, y, dt, x=None):
        #self.z = vq2qv(y)
        #self.x = self.dyn_sys.V_map(self.dyn_sys.zfyf_to_zy(zf=self.z))

        # Assumes y has been centered
        if hasattr(self.dyn_sys, "adiabatic") and self.dyn_sys.adiabatic:
            V = self.dyn_sys.V[0]
            # y_bar = np.tile(self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "q_bar"), 5)
            self.x = self.dyn_sys.V_map(V, y)
        else:
            self.x = self.dyn_sys.V_map(y)

class DiscreteEKFObserver:
    """
    Updates the belief state mean and covariance using the Extended Kalman Filter framework. We consider C is
    linear with respect to the state. This is a reduced order estimator.

    :dyn_sys: SSMR dynamical system object
    :param Sigma0: Initial state covariance (dim r x r)
    :param W: Process noise covariance (x_k+1 = f(x_k, u_k) + w_k
    :param V: Observation noise covariance (y_k = W(x_k) + v_k)
    """
    def __init__(self, dyn_sys, **kwargs):
        self.dyn_sys = dyn_sys
        self.state_dim = self.dyn_sys.get_state_dim()
        self.meas_dim = self.dyn_sys.get_output_dim()

        # Set initial covariance and noise covariances
        self.Sigma = jnp.asarray(kwargs.get('Sigma0', jnp.eye(self.state_dim)))
        self.W = jnp.asarray(kwargs.get('W', 100*jnp.eye(self.state_dim)))
        self.V = jnp.asarray(kwargs.get('V', jnp.eye(self.meas_dim)))

        # Initialize based on observable equilibrium position
        self.initialize(jnp.zeros(self.meas_dim))

    def get_meas_dim(self):
        return self.meas_dim


    def initialize(self, y):
        """
        Initialize the reduced order state estimate. By default the state is initialized in __init__
        to x_ref, but the user can override if desired
        """
        # Compute x based on current observation (q, v)
        y = jnp.asarray(y)
        self.x = self.dyn_sys.observed_to_reduced(y)
        self.z = jnp.array(self.dyn_sys.x_to_zfyf(self.x, zf=True))

    def update(self, u, y, dt, **kwargs):
        """
        Full EKF, see https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/ekf_lecture_notes.pdf for details
        :param u: input at timestep k
        :param y: measurement at timestep k+1
        :dt: timestep (s)
        """
        u = jnp.asarray(u)
        y = jnp.asarray(y)
        self.predict_state(u, dt)
        self.update_state(y)

    def predict_state(self, u, dt):
        """
        Predictor update step, see https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/ekf_lecture_notes.pdf
        for details
        :param u: input at timestep k
        :dt: timestep (s)
        """
        # Get linearizations of reduced dynamics at current state x
        u = jnp.asarray(u)
        A_d, B_d, d_d = self.dyn_sys.get_jacobians(self.x, u, dt)

        # Get next step based on linearization
        self.x = jnp.asarray(self.dyn_sys.update_dynamics(self.x, u, A_d, B_d, d_d))
        self.Sigma = A_d @ self.Sigma @ A_d.T + self.W

    def update_state(self, y):
        """
        Filter update step, see https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/ekf_lecture_notes.pdf
        for details
        :param y: (centered) measurement at timestep k+1
        :return x: updated state based on measurement (x_{k+1|k+1})
        """

        # Jacobian of manifold mapping (This is predicted x)
        y = jnp.asarray(y)
        H, c = self.dyn_sys.get_observer_jacobians(self.x)

        # Computing inverse explicitly is faster
        S = H @ self.Sigma @ H.T + self.V
        K = self.Sigma @ H.T @ jnp.linalg.inv(S)

        # K = computeRicattiGain(H, self.Sigma, self.V)

        self.x = self.x + K @ (y - self.dyn_sys.reduced_to_output(np.array(self.x)))
        self.Sigma = (jnp.eye(self.state_dim) - K @ H) @ self.Sigma
        # self.z = self.dyn_sys.x_to_zfyf(self.x, zf=True)

        return self.x

# @jax.jit
# def computeRicattiGain(H, Sigma, V):
#     S_T = jnp.transpose(H @ Sigma @ H.T + V)
#     K_T = jsp.linalg.solve(S_T, H @ Sigma.T, sym_pos=True)
#     return jnp.transpose(K_T)