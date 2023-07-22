import numpy as np
import rclpy
from rclpy.node import Node
from scipy.interpolate import interp1d
from soft_robot_control_ros.srv import GuSTOsrv

from sofacontrol.scp.locp import LOCP
from sofacontrol.utils import arr2np, np2arr
from sofacontrol.utils import CircleObstacle
from functools import partial
import jax.numpy as jnp
import jax


def runMPCSolverNode(model, N, dt, cost_params, target, U=None, X=None, Xf=None, dU=None, verbose=0, warm_start=True,
                     **kwargs):
    """
    See sofacontrol/scp/ros.py for details
    """
    rclpy.init()
    node = MPCSolverNode(model, N, dt, cost_params, target, U=U, X=X, Xf=Xf, dU=dU, verbose=verbose,
                         warm_start=warm_start, **kwargs)
    rclpy.spin(node)
    rclpy.shutdown()

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


class MPCSolverNode(Node):
    """
    Defines a service provider node that will run MPC using LOCP
    """

    def __init__(self, model, horizon, dt, cost_params, target, U=None, X=None, Xf=None, dU=None, verbose=0,
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

        # Initialize the ROS node
        super().__init__('mpc')

        # Define the service, which uses the gusto callback function
        self.srv = self.create_service(GuSTOsrv, 'mpc_solver', self.mpc_callback)

    def mpc_callback(self, request, response):
        """
        Callback function that runs when the service is queried, request message contains:
        t0, x0

        and the response message will contain:

        t, xopt, uopt, zopt
        """
        t0 = request.t0
        x0 = arr2np(request.x0, self.model.N, squeeze=True)

        # Project x0 to state constraint
        # if self.X is not None:
        #     if not self.X.contains(x0):
        #         x0 = self.X.project_to_polyhedron(x0)

        # Update linearization of conic constraint if circle obstacle
        x_init = self.xopt if self.xopt is not None else np.tile(x0, (self.planning_horizon + 1, 1))

        if type(self.X) is CircleObstacle:
            G_d, b_d = self.get_obstacleConstraint_linearization(x_init, self.X.center)
        else:
            G_d, b_d = None, None

        # Get target values at proper times by interpolating
        z, zf, u = self.get_target(t0)
        self.locp.update(self.A_d, self.B_d, self.d_d, x0, None, 0, 0, z=z, zf=zf, u=u, Gd=G_d, bd=b_d)
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
        self.topt = t0 + self.dt * np.arange(self.planning_horizon + 1)
        response.t = np2arr(self.topt)
        response.xopt = np2arr(self.xopt)
        response.uopt = np2arr(self.uopt)
        try:
            response.solve_time = solver_stats.solve_time
        except:
            response.solve_time = 0.0
        return response

    @partial(jax.jit, static_argnums=(0,))
    def get_obstacleConstraint_linearization(self, x, obs_center):
        G_d = []
        b_d = []
        # Only take up to the dimension of the obstacle center
        # n_c = obs_center.shape[0]
        x = jnp.asarray(x)
        for i in range(x.shape[0]):
            G_d_i = []
            b_d_i = []
            # For each constraint in self.X.center, get the linearization of the constraint at point x[i, :]
            for j in range(len(self.X.center)):
                G_d_i_j, b_d_i_j = self.model.get_obstacleConstraint_jacobians(x[i, :], obs_center[j])
                G_d_i.append(G_d_i_j)
                b_d_i.append(b_d_i_j)
            
            G_d.append(G_d_i)
            b_d.append(b_d_i)

        return G_d, b_d

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


class MPCClientNode(Node):
    """
    The client side of the MPC service. This object is used to query
    the ROS node to solve a MPC problem.

    Once a MPCSolverNode is running, instantiate this object and then use
    send_request to send a query the MPC solver.
    """

    def __init__(self):
        rclpy.init()
        super().__init__('mpc_client')
        self.cli = self.create_client(GuSTOsrv, 'mpc_solver')

        # Wait until the solver node is up and running
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('MPC solver not available, waiting...')

        # Request message definition
        self.req = GuSTOsrv.Request()

    def send_request(self, t0, x0, wait=True):
        """
        :param t0:
        :param x0:
        :param wait: Boolean
        :return:
        """
        self.req.t0 = t0
        self.req.x0 = np2arr(x0)

        self.future = self.cli.call_async(self.req)

        if wait:
            # Synchronous call, not compatible for real-time applications
            rclpy.spin_until_future_complete(self, self.future)

    def force_spin(self):
        if not self.check_if_done():
            rclpy.spin_once(self, timeout_sec=0)

    def check_if_done(self):
        """
        """
        return self.future.done()

    def force_wait(self):
        self.get_logger().warning('Overrides realtime compatibility, solve is too slow. Consider modifying problem')
        rclpy.spin_until_future_complete(self, self.future)

    def get_solution(self, n_x, n_u):
        """
        """
        # Get result
        res = self.future.result()

        t = arr2np(res.t, 1, squeeze=True)
        xopt = arr2np(res.xopt, n_x)
        uopt = arr2np(res.uopt, n_u)
        t_solve = res.solve_time

        return t, uopt, xopt, t_solve
