import sympy as sp
import numpy as np
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

from sofacontrol.utils import load_data

class SSMData:
    def __init__(self, delays, y_eq):
        self.delays = delays
        self.y_eq = np.atleast_2d(y_eq)

        self.y = None
        self.u = None

    def add_measurement(self, y, u):
        """
        Adds data point (called in online simulation): Time delay embeddings are required to embed the SSM in
        an appropriate-dimensional space. The observed states are automatically shifted to the origin
        :param y: Measurement of observable states (e.g. position of EE node)
        :param u: Control input
        """
        if self.y is None:
            self.y = y - self.y_eq
            self.u = u

        else:
            self.y = np.append(self.y, y - self.y_eq, axis=0)
            self.u = np.append(self.u, u, axis=0)

    def get_y_delay(self, step=-1):
        if len(self.y) < self.delays + 1:
            return None
        else:
            # Defaults to the previous measurement
            y = self.y[step]
            # u = self.u[step]

            ydel = np.zeros((self.delays * self.y.shape[1]))
            # udel = np.zeros((self.delay * self.u.shape[1]))

            for j in range(self.delays):
                fillrange_y = range(-self.y.shape[1] * (j + 1), -self.y.shape[1] * j)
                ydel[fillrange_y] = self.y[step - (j + 1), :]

                # fillrange_u = range(self.u.shape[1] * j, self.u.shape[1] * (j + 1))
                # udel[fillrange_u] = self.u[step - (j + 1), :]

            # TODO: For now, neglect inputs
            #zetak = np.hstack([y, ydel, udel])
            yk = np.hstack([ydel, y]) # This is based on assumption we made on measurements vector in OutputModel class
            return yk