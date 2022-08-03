import numpy as np
import rclpy
from rclpy.node import Node
from scipy.interpolate import interp1d
from soft_robot_control_ros.srv import GuSTOsrv

from sofacontrol.scp.gusto import GuSTO
from sofacontrol.utils import arr2np, np2arr


def runGuSTOSolverNode(model, N, dt, Qz, R, x0, t=None, z=None, u=None, Qzf=None, zf=None,
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
    rclpy.init()
    node = GuSTOSolverNode(model, N, dt, Qz, R, x0, t=t, z=z, u=u, Qzf=Qzf, zf=zf,
                           U=U, X=X, Xf=Xf, dU=dU, verbose=verbose,
                           warm_start=warm_start, **kwargs)
    rclpy.spin(node)
    rclpy.shutdown()


class GuSTOSolverNode(Node):
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
        self.xopt, self.uopt, _, _ = self.gusto.get_solution()
        self.topt = self.dt * np.arange(self.N + 1)

        # Initialize the ROS node
        super().__init__('gusto')

        # Define the service, which uses the gusto callback function
        self.srv = self.create_service(GuSTOsrv, 'gusto_solver', self.gusto_callback)

    def gusto_callback(self, request, response):
        """
        Callback function that runs when the service is queried, request message contains:
        t0, x0

        and the response message will contain:

        t, xopt, uopt, zopt
        """
        t0 = request.t0
        x0 = arr2np(request.x0, self.model.n_x, squeeze=True)

        # Get target values at proper times by interpolating
        z, zf, u = self.get_target(t0)

        # Get initial guess
        idx0 = np.argwhere(self.topt >= t0)[0, 0]
        u_init = self.uopt[-1, :].reshape(1, -1).repeat(self.N, axis=0)
        u_init[0:self.N - idx0] = self.uopt[idx0:, :]
        x_init = self.xopt[-1, :].reshape(1, -1).repeat(self.N + 1, axis=0)
        x_init[0:self.N + 1 - idx0] = self.xopt[idx0:, :]

        # Solve GuSTO and get solution
        self.gusto.solve(x0, u_init, x_init, z=z, zf=zf, u=u)
        self.xopt, self.uopt, zopt, t_solve = self.gusto.get_solution()

        self.topt = t0 + self.dt * np.arange(self.N + 1)
        response.t = np2arr(self.topt)
        response.xopt = np2arr(self.xopt)
        response.uopt = np2arr(self.uopt)
        response.zopt = np2arr(zopt)
        response.solve_time = t_solve

        return response

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


class GuSTOClientNode(Node):
    """
    The client side of the GuSTO service. This object is used to query
    the ROS node to solve a GuSTO problem.

    Once a GuSTOSolverNode is running, instantiate this object and then use
    send_request to send a query the GuSTO solver.
    """

    def __init__(self):
        rclpy.init()
        super().__init__('gusto_client')
        self.cli = self.create_client(GuSTOsrv, 'gusto_solver')

        # Wait until the solver node is up and running
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('GuSTO solver not available, waiting...')

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
