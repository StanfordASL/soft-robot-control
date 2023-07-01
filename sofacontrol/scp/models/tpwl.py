import numpy as np

import sofacontrol.utils as scutils
from .template import TemplateModel


class TPWLGuSTO(TemplateModel):
    """
    Provides an interface between GuSTO's template model and the TPWL model class
    defined in sofacontrol/tpwl/tpwl.py

    :dyn_sys: TPWL object as defined in sofacontrol/tpwl/tpwl.py
    :Hf: full order performance variable matrix zf = Hf xf
    """

    def __init__(self, dyn_sys):
        super(TPWLGuSTO, self).__init__()
        self.dyn_sys = dyn_sys

        # Define reduced order performance variable matrix
        if self.dyn_sys.H is not None:
            self.H = self.dyn_sys.H
        else:
            raise RuntimeError('dyn_sys must have output model specified')

        # Model dimensions
        self.n_x = self.dyn_sys.get_state_dim()
        self.n_u = self.dyn_sys.get_input_dim()
        self.n_z = self.H.shape[0]
        self.nonlinear_observer = False

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
        A, B, d = self.dyn_sys.get_jacobians(x)
        f = A @ x + B @ u + d
        return f, A, B

    def get_discrete_dynamics(self, x, u, dt):
        """
        :x: State x0 (n_x)
        :u: Input u0 (n_u)
        :dt: time step for discretization (seconds)
        """
        return self.dyn_sys.get_jacobians(x, dt=dt)

    def pre_discretize(self, dt):
        """ 
        Precompute all discrete time dynamics Jacobians, only possible if tpwl_method = 'nn'
        """
        self.dyn_sys.pre_discretize(dt)

    def get_characteristic_vals(self):
        """
        Uses the TPWL model's saved (q,v,u) points to determine "characteristic"
        quantities for x and f = A(x)x + B(x)u + d(x) (continuous time xdot). The
        characteristic quantities are given by looking at the maximum absolute
        value of each dimension over all saved points in the model.
        """

        # Characteristic quantity of x
        x = scutils.qv2x(self.dyn_sys.tpwl_dict['q'], self.dyn_sys.tpwl_dict['v'])
        x_char = np.abs(x).max(axis=0)

        # Characteristic quantity of the dynamics f
        f = np.zeros(x.shape)
        for i in range(x.shape[0]):
            f[i, :], _, _ = self.get_continuous_dynamics(x[i, :], self.dyn_sys.tpwl_dict['u'][i, :])
        f_char = np.abs(f).max(axis=0)

        return x_char, f_char

    def rollout(self, x0, u, dt):
        """
        Simply use the TPWL model built in rollout function.
        
        :x0: initial condition (n_x,)
        :u: array of control (N, n_u)
        :dt: time step

        Returns state x (N + 1, n_x) and performance variable z (N + 1, n_z)
        """
        return self.dyn_sys.rollout(x0, u, dt)

    def get_obstacleConstraint_jacobians(self, x, obs_center):
        return self.dyn_sys.get_obstacleConstraint_jacobians(x, obs_center)