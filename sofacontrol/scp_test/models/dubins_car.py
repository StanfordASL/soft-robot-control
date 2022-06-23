import numpy as np

from .template import TemplateModel


class DubinsCar(TemplateModel):

    def __init__(self):
        super(DubinsCar, self).__init__()
        self.n_x = 3
        self.n_u = 2
        self.n_z = 3
        self.H = np.eye(3)
        self.n_z = 3

    def get_continuous_dynamics(self, x, u):
        """
        """
        f = np.array([u[0] * np.cos(x[2]),
                      u[0] * np.sin(x[2]),
                      u[1]])

        A = np.zeros((3, 3))
        A[0, 2] = -u[0] * np.sin(x[2])
        A[1, 2] = u[0] * np.cos(x[2])

        B = np.zeros((3, 2))
        B[0, 0] = np.cos(x[2])
        B[1, 0] = np.sin(x[2])
        B[2, 1] = 1.
        return f, A, B

    def get_discrete_dynamics(self, x, u, dt):
        f, A, B = self.get_continuous_dynamics(x, u)
        d = f - A @ x - B @ u

        # Forward Euler discretization
        A_d = np.eye(A.shape[0]) + dt * A
        B_d = dt * B
        d_d = dt * d

        return A_d, B_d, d_d

    def get_next_state(self, x, u, dt):
        f = np.array([u[0] * np.cos(x[2]),
                      u[0] * np.sin(x[2]),
                      u[1]])

        return x + dt * f

    def rollout(self, x0, u, dt):
        """
        Rollouts state input trajectory. Typically
        :param x0: Initial state (n_x)
        :param u: Initial input sequence (N x n_u)
        :return: State, input tuple (x, u)
        :dt: time step (seconds)
        """
        N = u.shape[0]
        x = np.zeros((N + 1, self.n_x))
        x[0, :] = x0
        for i in range(N):
            x[i + 1, :] = np.reshape(self.get_next_state(x[i, :], u[i, :], dt), -1)

        return x
