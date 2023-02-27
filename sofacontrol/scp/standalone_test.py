import numpy as np
from scipy.interpolate import interp1d

from sofacontrol.scp.gusto import GuSTO
from sofacontrol.utils import arr2np, np2arr


def runGuSTOSolverStandAlone(model, N, dt, Qz, R, x0, t=None, z=None, u=None, Qzf=None, zf=None,
                       U=None, X=None, Xf=None, dU=None, verbose=0, warm_start=True, **kwargs):
    """
    Function that builds a ROS node to run GuSTO and runs it continuously. This node
    provides a service that at each query will run GuSTO once.

    :model: GuSTO module TemplateModel object describing dynamics (see scp/models/template.py)
    :N: integer optimization horizon, number of steps of length dt
    :dt: time step (seconds)
    :Qz: positive semi-definite performance variable weighting matrix (n_z, n_z)
    :R: positive definite control weighting matrix (n_u, n_u)
    :x0: initial condition (n_x,)
    :t: (optional) desired trajectory time vector (M,), required if z or u variables are
                   2D arrays, used for interpolation of z and u
    :z: (optional) desired tracking trajectory for objective function. Can either be array
                   of size (M, n_z) to correspond to t, or can be a constant 1D array (n_z,)
    :u: (optional) desired control for objective function. Can either be array of size (M, n_u)
                   to correspond to t, or it can be a constant 1D array (n_u,)
    :Qzf: (optional) positive semi-definite terminal performance variable weighting matrix (n_z, n_z)
    :zf: (optional) terminal target state (n_z,), defaults to 0 if Qzf provided
    :U: (optional) control constraint (Polyhedron object)
    :X: (optional) state constraint (Polyhedron object)
    :Xf: (optional) terminalstate constraint (Polyhedron object)
    :dU: (optional) u_k - u_{k-1} constraint Polyhedron object
    :verbose: (optional) 0,1,2 varying levels of verbosity (default 0)
    :warm_start: (optional) boolean (default True)
    :kwargs: (optional): Keyword args for GuSTO (see gusto.py GuSTO __init__.py and and optionally for the solver
    (https://osqp.org/docs/interfaces/solver_settings.html)
    """
    problem = GuSTOSolverNode(model, N, dt, Qz, R, x0, t=t, z=z, u=u, Qzf=Qzf, zf=zf,
                           U=U, X=X, Xf=Xf, dU=dU, verbose=verbose,
                           warm_start=warm_start, **kwargs)

    return problem.get_solution()

class GuSTOSolverNode():
    """
    Defines a service provider node that will run GuSTO
    """

    def __init__(self, model, N, dt, Qz, R, x0, t=None, z=None, u=None, Qzf=None, zf=None,
                 U=None, X=None, Xf=None, dU=None, verbose=0, warm_start=True, **kwargs):
        self.model = model
        self.N = N
        self.dt = dt

        # Get characteristic values for GuSTO scaling
        x_char, f_char = self.model.get_characteristic_vals()

        # Define cost function matrices
        self.Qzf = Qzf

        # Define target values
        self.t = t
        self.z = z
        self.u = u
        if z is not None and z.ndim == 2:
            self.z_interp = interp1d(t, z, axis=0,
                                     bounds_error=False, fill_value=(z[0, :], z[-1, :]))

        if u is not None and u.ndim == 2:
            self.u_interp = interp1d(t, u, axis=0,
                                     bounds_error=False, fill_value=(u[0, :], u[-1, :]))

        # Set up GuSTO and run first solve with a simple initial guess
        u_init = np.zeros((self.N, self.model.n_u))
        x_init, _ = self.model.rollout(x0, u_init, self.dt)
        z, zf, u = self.get_target(0.0)
        self.gusto = GuSTO(model, N, dt, Qz, R, x0, u_init, x_init, z=z, u=u,
                           Qzf=Qzf, zf=zf, U=U, X=X, Xf=Xf, dU=dU,
                           verbose=verbose, warm_start=warm_start,
                           x_char=x_char, f_char=f_char, **kwargs)

    def get_solution(self):
        self.xopt, self.uopt, self.zopt, _ = self.gusto.get_solution()
        self.topt = self.dt * np.arange(self.N + 1)

        return self.xopt, self.uopt, self.zopt, self.topt

    def get_target(self, t0):
        """
        Returns z, zf, u arrays for GuSTO solve
        """
        t = t0 + self.dt * np.arange(self.N + 1)

        # Get target z terms for cost function
        if self.z is not None:
            if self.z.ndim == 2:
                z = self.z_interp(t)
            else:
                z = self.z.reshape(1, -1).repeat(self.N + 1)
        else:
            z = None

        # Get target zf term for cost function
        if self.Qzf is not None and z is not None:
            zf = z[-1, :]
        else:
            zf = None

        # Get target u terms for cost function
        if self.u is not None:
            if self.u.ndim == 2:
                u = self.u_interp(t)
            else:
                u = self.u.reshape(1, -1).repeat(self.N)
        else:
            u = None

        return z, zf, u