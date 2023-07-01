import control
import numpy as np
import scipy.linalg


def solve_riccati(A, B, Q, R):
    """
    Solves discrete ARE, returns gain matrix K s.t. u = +K*x
    Faster implementation than control.dlqr for systems with large n (state_dim)
    """
    n = A.shape[0]
    m = B.shape[1]
    P = np.zeros((n, n))
    L = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    Lold = np.infty * np.ones((m, n))
    while (np.linalg.norm(L - Lold)) > 1e-4:
        Lold = L
        P = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
        L = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    return L, P


def dare(Ad, Bd, Q, R):
    """
    Solves discrete ARE, returns gain matrix K s.t. u = +K*x,
    for testing against solve_riccati
    """
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = -scipy.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return K, P


class DLQR:
    """
    Infinite horizon discrete LQR framework, which can be used effectively for setpoint regulation
    """

    def __init__(self, dt, model, cost_params):
        self.dt = dt
        self.model = model
        self.cost_params = cost_params

    def compute_policy(self, target):
        u_nom = np.atleast_1d(target.u)
        x_nom = target.x

        K = self.compute_gain_matrix(target.A, target.B, self.cost_params.Q, self.cost_params.R)
        return x_nom, u_nom, K

    def compute_gain_matrix(self, A, B, Q, R):
        Ad, Bd, _ = self.model.discretize_dynamics(A_c=A, B_c=B, d_c=np.zeros(self.model.get_state_dim()), dt=self.dt)
        K, _ = solve_riccati(Ad, Bd, Q, R)
        return K


class CLQR(DLQR):
    """
    This is an infinite horizon continuous LQR framework, which can be used effectively for setpoint regulation
    """

    def compute_gain_matrix(self, A, B, Q, R):
        K, _, _ = control.lqr(A, B, Q, R)  # in addition to control pkg, slycot needs to be installed
        return np.asarray(K)
