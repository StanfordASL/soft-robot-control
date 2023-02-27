import numpy as np
from scipy.interpolate import interp1d

from sofacontrol.scp.locp import LOCP
from sofacontrol.utils import arr2np, np2arr


def runMPCSolver(model, N, dt, cost_params, x0, target, U=None, X=None, Xf=None, dU=None, verbose=0, warm_start=True,
                     **kwargs):
    problem = MPCSolver(model, N, dt, cost_params, x0, target, U=U, X=X, Xf=Xf, dU=dU, verbose=verbose,
                         warm_start=warm_start, **kwargs)

    return problem.get_solution()


class MPCSolver():
    """
        Defines a service provider node that will run MPC using LOCP
        """

    def __init__(self, model, horizon, dt, cost_params, x0, target, U=None, X=None, Xf=None, dU=None, verbose=0,
                 warm_start=True, **kwargs):
        self.model = model
        self.planning_horizon = horizon
        self.dt = dt

        # Define target values
        self.target = target
        self.cost_params = cost_params
        if self.target.z is not None and self.target.z.ndim == 2:
            self.z_interp = interp1d(self.target.t, self.target.z, axis=0,
                                     bounds_error=False, fill_value=(self.target.z[0, :], self.target.z[-1, :]))

        if self.target.u is not None and self.target.u.ndim == 2:
            self.u_interp = interp1d(self.target.t, self.target.u, axis=0,
                                     bounds_error=False, fill_value=(self.target.u[0, :], self.target.u[-1, :]))

        self.verbose = verbose

        # LOCP problem
        if self.verbose == 2:
            locp_verbose = True
        else:
            locp_verbose = False

        self.locp = LOCP(self.planning_horizon, self.model.H, self.cost_params.Q, self.cost_params.R,
                         Qzf=self.cost_params.Qf, U=U, X=X, Xf=Xf, dU=dU, verbose=locp_verbose, warm_start=warm_start,
                         is_tr_active=False, **kwargs)

        # Get the linear model matrices
        self.A_d = [self.model.A_d for i in range(self.planning_horizon)]
        self.B_d = [self.model.B_d for i in range(self.planning_horizon)]
        if hasattr(self.model, 'd_d'):
            self.d_d = [self.model.d_d for i in range(self.planning_horizon)]
        else:
            self.d_d = [np.zeros(self.model.A_d.shape[0]) for i in range(self.planning_horizon)]

        self.X = X

        self.xopt = None
        self.uopt = None
        self.topt = None

        # Initial condition: if koopman, this is the lifted x0
        x0 = arr2np(x0, self.model.N, squeeze=True)

        z, zf, u = self.get_target(0.0)
        self.locp.update(self.A_d, self.B_d, self.d_d, x0, None, 0, 0, z=z, zf=zf, u=u)
        Jstar, success, solver_stats = self.locp.solve()

        if success:
            if self.verbose:
                try:
                    print('{:.3f} s from LOCP solve'.format(solver_stats.solve_time))
                except:
                    print('Solver failed.')
            self.xopt, self.uopt, _ = self.locp.get_solution()

        else:
            print('No solution found, extending previous solution')
            self.xopt = np.concatenate((self.xopt[1:, :], np.expand_dims(self.xopt[-1, :], axis=0)), axis=0)
            self.uopt = np.concatenate((self.uopt[1:, :], np.expand_dims(self.uopt[-1, :], axis=0)), axis=0)

        self.locp.solve()

    def get_solution(self):
        self.xopt, self.uopt, self.zopt = self.locp.get_solution()
        self.topt = self.dt * np.arange(self.planning_horizon + 1)

        return self.xopt, self.uopt, self.zopt, self.topt

    def get_target(self, t0):
        """
        Returns z, zf, u arrays for GuSTO solve
        """
        t = t0 + self.dt * np.arange(self.planning_horizon + 1)

        # Get target z terms for cost function
        if self.target.z is not None:
            if self.target.z.ndim == 2:
                z = self.z_interp(t)
            else:
                z = self.target.z.reshape(1, -1).repeat(self.planning_horizon + 1)
        else:
            z = None

        # Get target zf term for cost function
        if self.cost_params.Qf is not None and z is not None:
            zf = z[-1, :]
        else:
            zf = None

        # Get target u terms for cost function
        if self.target.u is not None:
            if self.target.u.ndim == 2:
                u = self.u_interp(t)
            else:
                u = self.target.u.reshape(1, -1).repeat(self.planning_horizon)
        else:
            u = None

        return z, zf, u