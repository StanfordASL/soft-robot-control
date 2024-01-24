import sympy as sp
import numpy as np
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

from sofacontrol.utils import load_data, norm2Diff
from functools import partial
import jax.numpy as jnp
import jax

class KoopmanData:
    def __init__(self, scale, delay, inputInFeatures=True):
        self.delay = delay
        self.scaling = KoopmanScaling(scale)
        self.inputInFeatures = inputInFeatures # Include input in features

        self.y_norm = None  # Down-scaled
        self.u_norm = None  # Down-scaled

    def add_measurement(self, y, u):
        """
        Adds data point (called in online simulation): The reduced order state incorporates delays hence this is req.
        :param y: Measurement of low-dimensional state (e.g. position of EE node)
        :param u: Control input
        """
        if self.y_norm is None:
            self.y_norm = self.scaling.scale_down(y=y)
            self.u_norm = self.scaling.scale_down(u=u)

        else:
            self.y_norm = np.append(self.y_norm, self.scaling.scale_down(y=y), axis=0)
            self.u_norm = np.append(self.u_norm, self.scaling.scale_down(u=u), axis=0)

    def get_zeta(self, step=-1):
        if len(self.y_norm) < self.delay + 1:
            return None
        else:
            y = self.y_norm[step]
            u = self.u_norm[step]

            ydel = np.zeros((self.delay * self.y_norm.shape[1]))
            udel = np.zeros((self.delay * self.u_norm.shape[1]))

            for j in range(self.delay):
                fillrange_y = range(self.y_norm.shape[1] * j, self.y_norm.shape[1] * (j + 1))
                fillrange_u = range(self.u_norm.shape[1] * j, self.u_norm.shape[1] * (j + 1))
                ydel[fillrange_y] = self.y_norm[step - (j + 1), :]
                udel[fillrange_u] = self.u_norm[step - (j + 1), :]

            if self.inputInFeatures:
                zetak = np.hstack([y, ydel, udel])
            else:
                zetak = np.hstack([y, ydel])
            return zetak


class KoopmanOfflineData(KoopmanData):
    """
    Defines data required for Koopman based modeling.
    """

    def __init__(self, scale, delay):
        super().__init__(scale, delay)
        self.y = None
        self.u = None
        self.t = None

        self.zeta = None

    def load_offline_data(self, file):
        """
        Load a file containing (output node information, time step and control input) from a simulation or hardware exp.
        :param file: Data file path
        """
        data = load_data(file)
        self.y = data['z']
        self.t = data['t']
        self.u = data['u']
        self.y_norm = self.scaling.scale_down(y=self.y)
        self.u_norm = self.scaling.scale_down(u=self.u)

    def add_zeta_offline(self):
        """
        Bulk computation of zeta, low dimensional state considered for Koopman operator
        """
        self.zeta = []

        for i in range(self.delay, self.y_norm.shape[0]):
            self.zeta.append(self.get_zeta(step=i))
        self.zeta = np.asarray(self.zeta)


class KoopmanScaling:
    """
    Provides functions for going back and forth between scaled and normalized data given the scale as parameter
    """

    def __init__(self, scale):
        self.y_offset = scale['y_offset'][0, 0]
        self.y_factor = scale['y_factor'][0, 0]
        self.u_offset = scale['u_offset'][0, 0]
        self.u_factor = scale['u_factor'][0, 0]

    def scale_up(self, u=None, y=None):
        if y is not None:
            return y * self.y_factor + self.y_offset
        elif u is not None:
            return u * self.u_factor + self.u_offset

    def scale_down(self, u=None, y=None):
        if y is not None:
            return (y - self.y_offset) / self.y_factor
        elif u is not None:
            return (u - self.u_offset) / self.u_factor


class KoopmanModel:
    """
    Builds a Koopman model, and has lift_f, which defines how data is lifted from zeta --> z (lifted state)
    """

    def __init__(self, model_in, params_in, DMD=False):
        # If truncation used, system matrices are truncated in soft-robot-koopman matlab code
        self.A_d = model_in['A'][0, 0]
        self.B_d = model_in['B'][0, 0]
        self.C = model_in['C'][0, 0]
        self.H = self.C.copy()
        self.M = model_in['M'][0, 0]
        self.K = model_in['K'][0, 0]
        # V is right matrix, W is inverse of V
        if 'V' in model_in.dtype.names:
            self.V = model_in['V'][0, 0]
        else:
            self.V = np.eye(self.A_d.shape[0])

        if 'W' in model_in.dtype.names:
            self.W = model_in['W'][0, 0]
        else:
            self.W = np.eye(self.A_d.shape[0])

        self.n = int(params_in['n'])
        self.m = int(params_in['m'])
        self.N = int(params_in['N'])
        self.state_dim = int(params_in['nzeta'])
        self.delays = int(params_in['delays'])
        self.obs_degree = int(params_in['obs_degree'])
        self.obs_type = str(params_in['obs_type'][0, 0][0, 0][0])
        self.Ts = float(params_in['Ts'])
        self.scale = params_in['scale'][0, 0]
        if 'DMD' in params_in.dtype.names:
            self.DMD = bool(params_in['DMD'])
        else:
            self.DMD = DMD

        if 'G' in model_in.dtype.names:
            self.G = model_in['G'][0, 0]
        else:
            self.G = np.zeros((self.m, 3))

        if 'inputInFeatures' in params_in.dtype.names:
            self.inputInFeatures = bool(params_in['inputInFeatures'])
        else:
            self.inputInFeatures = True

        if not self.inputInFeatures:
            self.state_dim -= self.delays * self.m

        self.assert_dimensions()
        self.lift_data = self.get_lifting_function()

    def assert_dimensions(self):
        """
        Assert dimensions of Model are correct
        """
        assert self.A_d.shape == (self.N, self.N)
        assert self.B_d.shape == (self.N, self.m)
        assert self.C.shape == (self.n, self.N)

    def get_lifting_function(self):
        """
        Lift data from the basis functions to the lifted state
        :return: Lambdified expression in numpy style using sympy
        """
        if self.obs_type == 'poly':
            zeta = sp.Matrix(sp.symbols('zeta1:{}'.format(self.state_dim + 1)))
            polynoms = sorted(itermonomials(list(zeta), self.obs_degree),
                              key=monomial_key('grlex', list(reversed(zeta))))
            if self.DMD:
                polynoms = polynoms[1:]
            else:
                polynoms.append(polynoms[0])
                polynoms = polynoms[1:]

            # TODO: Only assert if not truncated
            # assert len(polynoms) == self.N
            return sp.lambdify(zeta, polynoms, 'numpy')
        else:
            print('{} is not implemented / not a valid selection. Please select a different obs type'
                  .format(self.obs_type))
    
    def get_obstacleConstraint_jacobians(self,
                                      x: jnp.ndarray, obs_center: jnp.ndarray):
        normFunc = partial(norm2Diff, y=obs_center)
        g = lambda x: normFunc(self.H @ x)
        G = jax.jacobian(g)(x)
        b = g(x) - G @ x
        return G, b
    
    def get_jacobians(self, x, u, dt=None):
        return self.A_d, self.B_d, None