import numpy as np
from scipy.interpolate import interp1d


class TrajTrackingLQR:
    def __init__(self, dt, model, cost_params):
        self.dt = dt
        self.model = model
        self.cost_params = cost_params

        self.x_bar = None
        self.u_bar = None

    def compute_policy(self, target):
        K, _ = self.perform_dlqr_recursion(target)
        return self.x_bar, self.u_bar, K

    def perform_dlqr_recursion(self, target):
        P = [self.model.H.T @ self.cost_params.Q @ self.model.H]
        K = []
        x_nom = []
        u_nom = []

        x_nom_interp = interp1d(target.t, target.x, axis=0)
        u_nom_interp = interp1d(target.t, target.u, axis=0)

        final_time = target.t[-1]
        nbr_steps = int(round(final_time / self.dt))

        for i in reversed(range(nbr_steps)):
            t_step = i * self.dt
            # Determine linear system at point
            x_nom_i = x_nom_interp(t_step)
            u_nom_i = u_nom_interp(t_step)
            A, B, d = self.model.get_jacobians(x_nom_i, u_nom_i, dt=self.dt)
            x_nom.append(x_nom_i)
            u_nom.append(u_nom_i)
            # As we are working with deviations from nom traj no need to extend state with affine terms
            K.append(-1. * np.linalg.solve(self.cost_params.R + B.T @ P[-1] @ B, B.T @ P[-1] @ A))
            P.append(self.model.H.T @ self.cost_params.Q @ self.model.H + K[-1].T @ self.cost_params.R @ K[-1] +
                     (A + B @ K[-1]).T @ P[-1] @ (A + B @ K[-1]))

        K = np.flip(np.asarray(K), axis=0)
        P = np.flip(np.asarray(P), axis=0)
        self.x_bar = np.flip(np.asarray(x_nom), axis=0)
        self.u_bar = np.flip(np.asarray(u_nom), axis=0)

        return K, P
