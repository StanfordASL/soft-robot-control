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


class DiscreteEKFObserver:
    """
    Updates the belief state mean and covariance using the Extended Kalman Filter framework. We consider C is
    linear with respect to the state. This is a reduced order estimator, it is assumed that the true
    measurement model is

    y = Cf*xf

    and that with a reduced order approximation xf = V*x + xf_ref this model becomes

    y = C*x + y_ref

    where C = Cf*V and y_ref = Cf*xf_ref.

    :dyn_sys: TPWL dynamical system object of class TPWLATV
    :param Sigma0: Initial state covariance (dim r x r)
    :param W: Process noise covariance (x_k+1 = f(x_k, u_k) + w_k
    :param V: Observation noise covariance (y_k = Cx_k + v_k)
    """
    def __init__(self, dyn_sys, **kwargs):
        self.dyn_sys = dyn_sys
        if self.dyn_sys.C is None:
            raise RuntimeError('Need to set meas. model in dyn_sys')
        self.C = self.dyn_sys.C
        self.state_dim = self.dyn_sys.get_state_dim()
        self.meas_dim = self.C.shape[0]

        # Set initial covariance and noise covariances
        self.Sigma = kwargs.get('Sigma0', np.eye(self.state_dim))
        self.W = kwargs.get('W', 100*np.eye(self.state_dim))
        self.V = kwargs.get('V', np.eye(self.meas_dim))

        # Initialize observer to the TPWL reference state
        self.initialize(self.dyn_sys.rom.x_ref)

    def get_meas_dim(self):
        return self.meas_dim

    def get_observer_params(self):
        return {'W': self.W, 'V': self.V, 'meas_dim': self.meas_dim, 'state_dim': self.state_dim,
                'C': self.C, 'H': self.dyn_sys.H}

    def initialize(self, xf):
        """
        Initialize the reduced order state estimate. By default the state is initialized in __init__
        to x_ref, but the user can override if desired
        """
        self.x = self.dyn_sys.rom.compute_RO_state(xf=xf)

        if self.dyn_sys.H is not None:
            self.z = self.dyn_sys.x_to_zfyf(self.x, zf=True)
        else:
            self.z = self.dyn_sys.x_to_zfyf(self.x, yf=True)

    def update(self, u, y, dt, **kwargs):
        """
        Full EKF, see https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/ekf_lecture_notes.pdf for details
        :param u: input at timestep k
        :param y: measurement at timestep k+1
        :dt: timestep (s)
        """
        self.predict_state(u, dt)
        self.update_state(y)

    def predict_state(self, u, dt):
        """
        Predictor update step, see https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/ekf_lecture_notes.pdf
        for details
        :param u: input at timestep k
        :dt: timestep (s)
        """
        A_d, B_d, d_d = self.dyn_sys.get_jacobians(self.x, dt)
        self.x = self.dyn_sys.update_dynamics(self.x, u, A_d, B_d, d_d)
        self.Sigma = A_d @ self.Sigma @ A_d.T + self.W

    def update_state(self, y):
        """
        Filter update step, see https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/ekf_lecture_notes.pdf
        for details
        :param y: measurement at timestep k+1
        :return x: updated state based on measurement (x_{k+1|k+1})
        """
        y = self.dyn_sys.zfyf_to_zy(yf=y) # convert full order measurement into reduced order measurement

        S = self.C @ self.Sigma @ self.C.T + self.V
        K = self.Sigma @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.C @ self.x)
        self.Sigma = (np.eye(self.state_dim) - K @ self.C) @ self.Sigma
        if self.dyn_sys.H is not None:
            self.z = self.dyn_sys.x_to_zfyf(self.x, zf=True)
        else:
            self.z = self.dyn_sys.x_to_zfyf(self.x, yf=True)

        return self.x

