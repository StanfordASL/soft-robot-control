import numpy as np

from sofacontrol.lqr.config import iLQRConfig


class iLQR:
    def __init__(self, dt, model, cost_params, planning_horizon, **kwargs):
        self.params = iLQRConfig()
        self.dt = dt
        self.model = model
        self.planning_horizon = planning_horizon
        self.cost_params = cost_params

        self.state_dim = model.get_state_dim()
        self.input_dim = model.get_input_dim()

        # Parameters that are set through controllers.py
        self.z_target = None
        self.u_last = np.zeros(self.input_dim)  # For receding horizon

    def set_target(self, z_target):
        self.z_target = z_target.copy()

    def set_u_last(self, u_last):
        self.u_last = u_last.copy()

    def ilqr_computation(self, x0, u_warmstart=None):
        """
        Iterative LQR loop
        :param x0: Starting state
        :param u_warmstart: None or np.array(self.time_horizon, self.input_dim): warmstarting can be used for faster iteration
        in receding horizon iLQR
        :return: Optimal policy (x_bar, u_bar, K)
        See https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf for algorithm, notation is largely the same
        """
        self.rho = self.params.rho0
        self.drho = self.params.drho0

        failed_update_counter = 0

        # Initial forward_pass, initialize x
        x_prev = np.zeros((self.planning_horizon + 1, self.state_dim))
        x_prev[0] = x0

        # Initialize u to zeros if no warmstart
        if u_warmstart is None:
            u_warmstart = np.zeros((self.planning_horizon, self.input_dim))

        x, u, cost, A, B, d = self.forward_pass(x_prev, u_warmstart)

        is_converged = False
        nbr_iter = 0

        while not is_converged and nbr_iter <= self.params.max_iter:  # convergence criteria
            print('Iteration {}'.format(nbr_iter))
            # Backward pass (dlqr recursion)
            K, k, Q_u, Q_uu = self.dlqr_recursion(x, u, A, B, d)

            # Calculate optimal trajectory change and do forward pass
            prev_cost = cost

            alpha = self.params.alpha0
            improved = False
            failed = False

            # Line search forward pass logic
            while not improved and not failed:
                improved = True
                x_temp, u_temp, cost_temp, A_temp, B_temp, d_temp = self.forward_pass(x, u, alpha=alpha, K=K, k=k)

                delta_cost = 0
                for t in range(self.planning_horizon):
                    delta_cost += alpha * k[t].T @ Q_u[t] + alpha ** 2 * .5 * k[t].T @ Q_uu[t] @ k[t]

                if self.params.do_linesearch:
                    decrease_ratio = (cost_temp - prev_cost) / delta_cost

                    if decrease_ratio <= self.params.improv_lb or decrease_ratio > self.params.improv_ub:
                        alpha = self.params.alpha_scaling * alpha
                        improved = False

                        if alpha < self.params.alpha_min:  # At the fifth iteration
                            print('No improved cost found, varying regularization parameter')
                            self.update_regularization(
                                increase=True)  # Get closer to GD (as likely far away from local optima)
                            self.rho += self.params.rho_increase_fp  # Also increase additionally
                            failed = True

            if not failed:
                x = x_temp
                u = u_temp
                cost = cost_temp
                A = A_temp
                B = B_temp
                d = d_temp
                is_converged = self.is_converged_calculation(prev_cost, cost)
                failed_update_counter = 0

            else:
                failed_update_counter += 1
                if failed_update_counter >= self.params.counter_limit:
                    print('Found local minima, abandoning search')
                    is_converged = True

            nbr_iter += 1

        return x, u, K  # Returns both optimal sequence and stabilizing controller

    def is_converged_calculation(self, prev_cost, cost):
        """ Convergence criteria: Can be tuned/changed """
        if ((prev_cost - cost) < self.params.epsilon) and ((prev_cost - cost) >= 0):
            print('Cost converged')
            return True
        else:
            return False

    def forward_pass(self, x_prev, u_prev, alpha=1., K=None, k=None):
        """ forward_pass of the system, starting at x0 and applying control sequence u

        x_prev: np.array (config.planning_horizon +1, n): Previous trajectory of the system
        u_prev: np.array (config.planning_horizon, m): Previous control sequence
        K: iLQR optimal feedback gain from previous backward pass
        k: iLQR optimal feedforward term from previous backward pass

        See https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf for algorithm, notation is largely the same
        """
        cost = 0

        x = np.zeros((self.planning_horizon + 1, self.state_dim))
        u = np.zeros((self.planning_horizon, self.input_dim))
        A = np.zeros((self.planning_horizon, self.state_dim, self.state_dim))
        B = np.zeros((self.planning_horizon, self.state_dim, self.input_dim))
        d = np.zeros((self.planning_horizon, self.state_dim))

        x[0] = x_prev[0]
        if K is None:
            K = np.zeros((self.planning_horizon, self.input_dim, self.state_dim))
        if k is None:
            k = np.zeros((self.planning_horizon, self.input_dim))
        # Simulate dynamics forward
        for t in range(self.planning_horizon):
            # Compute control via update to previous iteration control
            u[t] = u_prev[t] + alpha * k[t] + K[t] @ (x[t] - x_prev[t])
            # Compute cost of new trajectory
            if self.params.include_input_var_constraint:
                if t == 0:
                    cost += self.step_cost(x[t], u[t], step=t, u_prev_step=self.u_last)
                else:
                    cost += self.step_cost(x[t], u[t], step=t, u_prev_step=u[t - 1])
            else:
                cost += self.step_cost(x[t], u[t], step=t)

            # Compute Jacobians for current trajectory
            # Including u[t] here may no longer be breaking for TPWL.
            A[t], B[t], d[t] = self.model.get_jacobians(x[t], u=u[t], dt=self.dt)

            # Simulate new trajectory
            x[t + 1] = self.model.update_dynamics(x[t], u[t], A[t], B[t], d[t])
        # Terminal cost of new trajectory
        cost += self.terminal_cost(x[-1])

        return x, u, cost, A, B, d

    def terminal_cost(self, x):
        z = self.model.x_to_zfyf(x, zf=True)
        return .5 * (z - self.z_target[-1, :]).T @ self.cost_params.Qf @ (z - self.z_target[-1, :])

    def step_cost(self, x, u, step, u_prev_step=None):
        z = self.model.x_to_zfyf(x, zf=True)
        if u_prev_step is None:
            return .5 * (z - self.z_target[step, :]).T @ self.cost_params.Q @ (z - self.z_target[step, :]) + \
                   .5 * u.T @ self.cost_params.R @ u
        else:
            return .5 * (z - self.z_target[step, :]).T @ self.cost_params.Q @ (z - self.z_target[step, :]) + \
                   .5 * (u - u_prev_step).T @ self.cost_params.R @ (u - u_prev_step)

    def terminal_cost_vectors(self, x):
        z = self.model.x_to_zfyf(x, zf=True)
        c_xx = self.model.H.T @ self.cost_params.Qf @ self.model.H
        c_x = self.model.H.T @ self.cost_params.Qf @ (z - self.z_target[-1, :])
        c = self.terminal_cost(x)
        return c, c_x, c_xx

    # This only works if z = Hx. For SSM: z = C_map(x) + z_ref. Thus, this implementation
    # of iLQR does not work for general mappings from x to z
    def step_cost_vectors(self, x, u, step, u_prev_step=None):
        z = self.model.x_to_zfyf(x, zf=True)
        c_xx = self.model.H.T @ self.cost_params.Q @ self.model.H
        c_x = self.model.H.T @ self.cost_params.Q @ (z - self.z_target[step, :])
        c = self.step_cost(x, u, step, u_prev_step=u_prev_step)
        if u_prev_step is None:
            c_u = self.cost_params.R @ u
        else:
            c_u = self.cost_params.R @ (u - u_prev_step)
        c_uu = self.cost_params.R
        return c, c_x, c_xx, c_u, c_uu

    def update_regularization(self, increase=True):
        """
        Logic for modifying the regularization term in Q_uu in backward pass. See
        https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf for details
        :param increase: Bool, increase or decrease
        """
        if increase:
            self.drho = np.max((self.drho * self.params.rho_scaling, self.params.rho_scaling))
            self.rho = np.max((self.rho * self.drho, self.params.rho_min))

            if self.rho > self.params.rho_max:
                print('Warning: Max regularization parameter exceeded')
                self.rho = self.params.rho_max

        else:  # elif decrease
            self.dhro = np.min((self.drho / self.params.rho_scaling, 1.0 / self.params.rho_scaling))
            self.rho = self.rho * self.dhro

            if self.rho <= self.params.rho_min:
                self.rho = self.params.rho_min

    def dlqr_recursion(self, x, u, A, B, d):
        """
        Backward pass in iLQR

        :param x: np.array (config.planning_horizon +1, n): Nominal state of the system
        :param u: np.array (config.planning_horizon, m): Nominal control sequence
        :param A: np.array (config.planning_horizon, n, n): Jacobian w.r.t. state at nominal state, control
        :param B: np.array (config.planning_horizon, n, m): Jacobian w.r.t. control at nominal state, control
        :param d: np.array (config.planning_horizon, n): Affine term at nominal state, control
        :return: K: iLQR optimal feedback gain, k: iLQR optimal feedforward term, Q_u: iLQR cost Jacobian w.r.t u,
                 Q_uu: iLQR cost Hessian w.r.t. u, u

         See https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf for algorithm, notation is largely the same
        """

        while True:
            Q_x = np.zeros((self.planning_horizon, self.state_dim))
            Q_u = np.zeros((self.planning_horizon, self.input_dim))
            Q_xx = np.zeros((self.planning_horizon, self.state_dim, self.state_dim))
            Q_uu = np.zeros((self.planning_horizon, self.input_dim, self.input_dim))
            Q_ux = np.zeros((self.planning_horizon, self.input_dim, self.state_dim))

            p = np.zeros((self.planning_horizon + 1, self.state_dim))
            P = np.zeros((self.planning_horizon + 1, self.state_dim, self.state_dim))

            K = np.zeros((self.planning_horizon, self.input_dim, self.state_dim))
            k = np.zeros((self.planning_horizon, self.input_dim))

            _, p[-1], P[-1] = self.terminal_cost_vectors(x[-1])

            for t in reversed(range(self.planning_horizon)):
                if self.params.include_input_var_constraint:
                    if t == 0:
                        c, c_x, c_xx, c_u, c_uu = self.step_cost_vectors(x[t], u[t], t, u_prev_step=self.u_last)
                    else:
                        c, c_x, c_xx, c_u, c_uu = self.step_cost_vectors(x[t], u[t], t, u_prev_step=u[t - 1])
                else:
                    c, c_x, c_xx, c_u, c_uu = self.step_cost_vectors(x[t], u[t], t)

                Q_x[t] = c_x + A[t].T @ p[t + 1]
                Q_u[t] = c_u + B[t].T @ p[t + 1]
                Q_xx[t] = c_xx + A[t].T @ P[t + 1] @ A[t]
                Q_uu[t] = c_uu + B[t].T @ P[t + 1] @ B[t]
                Q_ux[t] = B[t].T @ P[t + 1] @ A[t]  # c_ux = 0

                if self.params.regularize:
                    if self.params.state_regularization:
                        Q_uu_tilde = c_uu + B[t].T @ (P[t + 1] + self.rho * np.eye(self.state_dim)) @ B[t]
                        Q_ux_tilde = B[t].T @ (P[t + 1] + self.rho * np.eye(self.state_dim)) @ A[t]
                    else:
                        Q_uu_tilde = Q_uu[t] + self.rho * np.eye(Q_uu[t].shape[0])
                        Q_ux_tilde = Q_ux[t]

                else:
                    Q_uu_tilde = Q_uu[t]
                    Q_ux_tilde = Q_ux[t]

                try:  # Fastest method to check for PD
                    np.linalg.cholesky(Q_uu_tilde)
                    pos_def = True
                except np.linalg.linalg.LinAlgError:
                    pos_def = False

                if not pos_def:
                    print('Q_uu not PD. If regularized reruns closer to GD')

                    if self.params.regularize:
                        self.update_regularization(increase=True)
                        break  # breaks out of for loop and restarts from top

                Q_uu_tilde_inv = np.linalg.inv(Q_uu_tilde)

                K[t] = - Q_uu_tilde_inv @ Q_ux_tilde
                k[t] = - Q_uu_tilde_inv @ Q_u[t]

                p[t] = Q_x[t] + K[t].T @ Q_uu[t] @ k[t] + K[t].T @ Q_u[t] + Q_ux[t].T @ k[t]
                P[t] = Q_xx[t] + K[t].T @ Q_uu[t] @ K[t] + K[t].T @ Q_ux[t] + Q_ux[t].T @ K[t]

            print('Succesfully performed dlqr recursion')
            self.update_regularization(increase=False)  # decrease parameter
            break  # breaks out of while loop
        return K, k, Q_u, Q_uu
