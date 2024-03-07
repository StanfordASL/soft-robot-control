import numpy as np

import sofacontrol.utils as scutils
from .template import TemplateModel


class SSMGuSTO(TemplateModel):
    """
    Provides an interface between GuSTO's template model and the SSM model class
    defined in sofacontrol/SSM/ssm.py

    :dyn_sys: SSM object as defined in sofacontrol/SSM/ssm.py
    :H: Performance variable matrix. Mapping from reduced to performance is nonlinear
    :   so this is simply the identity matrix for bookkeeping purposes
    """

    def __init__(self, dyn_sys):
        super(SSMGuSTO, self).__init__()
        self.dyn_sys = dyn_sys
        self.dyn_sys.debug = False
        # Define reduced order performance variable matrix.
        if self.dyn_sys.H is not None:
            self.H = self.dyn_sys.H
        else:
            raise RuntimeError('dyn_sys must have output model specified')

        # Model dimensions
        self.n_x = self.dyn_sys.get_state_dim()
        self.n_u = self.dyn_sys.get_input_dim()
        self.n_z = self.H.shape[0]

        # Observer Type
        self.nonlinear_observer = self.dyn_sys.nonlinear_observer

    def get_continuous_dynamics(self, x, u):
        """
        This model is represented as

            xdot = f(x,u) = A(x)x + B(x)u + d(x)

        where A(x), B(x), d(x) are defined by the nearest neighbor of a finite set of points to
        the current point x (think dynamics defined with indicator functions)

        As long as you are not on the boundary between two points you have constant A, B, d such that
        dA(x)/dx = 0, dB(x)/dx = 0, and dd(x)/dx = 0. Therefore

            df/dx = A(x)
            df/du = B(x)

        """
        # TODO: Checking jax capes
        if hasattr(self.dyn_sys, "adiabatic") and self.dyn_sys.adiabatic:
            R = self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "r_coeff")
            B_r = self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "B_r")
            u_bar = self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "u_bar")
            x_bar = self.dyn_sys.V[0].T @ np.tile(self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "q_bar"), 5)
            A, B, d = self.dyn_sys.get_continuous_jacobians(x, u, R, B_r, x_bar, u_bar)
        else:
            A, B, d = self.dyn_sys.get_continuous_jacobians(x, u=u)
        # A, B, d = self.dyn_sys.get_jacobians(x, u=u, dt=None)
        f = A @ x + B @ u + d
        return f, A, B

    def get_discrete_dynamics(self, x, u, dt):
        """
        :x: State x0 (n_x)
        :u: Input u0 (n_u)
        :dt: time step for discretization (seconds)
        """
        if hasattr(self.dyn_sys, "adiabatic") and self.dyn_sys.adiabatic:
            R = self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "r_coeff")
            B_r = self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "B_r")
            u_bar = self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "u_bar")
            x_bar = self.dyn_sys.V[0].T @ np.tile(self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "q_bar"), 5)
            return self.dyn_sys.get_jacobians(x, u, dt, R, B_r, x_bar, u_bar)
        else:
            return self.dyn_sys.get_jacobians(x, dt=dt, u=u)

    def get_observer_jacobians(self, x, u=None, dt=None):
        """
        :x: State x0 (n_x)
        :u: Input u0 (n_u)
        :dt: time step for discretization (seconds)
        """
        if hasattr(self.dyn_sys, "adiabatic") and self.dyn_sys.adiabatic:
            W = self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "w_coeff")
            x_bar = self.dyn_sys.V[0].T @ np.tile(self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "q_bar"), 5)
            y_bar = np.tile(self.dyn_sys.interpolator.transform(x[self.dyn_sys.interp_slice], "q_bar"), 5)
            return self.dyn_sys.get_observer_jacobians(x, W, x_bar, y_bar)
        else:
            return self.dyn_sys.get_observer_jacobians(x)

    def get_characteristic_vals(self):
        """
        An optional function to define a procedure for computing characteristic values
        of the state and dynamics for use with GuSTO scaling, defaults to all ones
        """
        x_char = np.ones(self.n_x)
        f_char = np.ones(self.n_x)
        return x_char, f_char

    def rollout(self, x0, u, dt):
        """
        Simply use the SSM model built in rollout function.
        
        :x0: initial condition (n_x,)
        :u: array of control (N, n_u)
        :dt: time step

        Returns state x (N + 1, n_x), performance variable z (N + 1, n_z),
        and approx performance var z_lin (N + 1, n_z)
        """
        return self.dyn_sys.rollout(x0, u, dt)