import numpy as np

from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
from tps import ThinPlateSpline
from smt.surrogate_models import RBF, IDW, KRG, QP, LS, RMTB

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import naturalneighbor

import time
import matplotlib.pyplot as plt

from jax import jit
from functools import partial

DISPLAY_NAMES = {
    "origin_only": "origin only",
    "idw": "Inverse Distance Weighting",
    "linear": "Barycentric linear",
    "nn": "Nearest Neighbor",
    "ct": "Clough-Tocher",
    "tps": "Thin Plate Spline",
    "rbf": "Radial Basis Function",
    "krg": "Kriging",
    "qp": "Quadratic Polynomial Regression",
    "ls": "Linear Regression",
    "rmts": "RMTS",
    "natural_neighbor": "Natural Neighbor"
}

class InterpolatorFactory():
    def __init__(self, interpolation_method, q_eq, coeff_dict):
        if interpolation_method == 'origin_only':
            interpolator = OriginOnlyInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'idw':
            interpolator = IDWInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'linear':
            interpolator = LinearInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'nn':
            interpolator = NearestNeighborInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'ct':
            interpolator = CloughTocherInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'tps':
            interpolator = ThinPlateSplineInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'rbf':
            interpolator = RBFInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'krg':
            interpolator = KrigingInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'rmts':
            interpolator = RMTSInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'qp':
            interpolator = SquaredRegressionInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'ls':
            interpolator = LinearRegressionInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'ridge':
            interpolator = PolynomialRidgeRegressionInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'natural_neighbor':
            interpolator = NaturalNeighborInterpolator(q_eq, coeff_dict)
        else:
            raise RuntimeError(f"Interpolation method not recognized: {interpolation_method}")
        self.interpolator = interpolator
    
    def get_interpolator(self):
        return self.interpolator

class Interpolator():
    def __init__(self, q_eq, coeff_dict):
        self.q_eq = np.array(q_eq)
        self.coeff_dict = coeff_dict
        self.fit()
    
    def fit(self):
        raise NotImplementedError("'fit' must be overriden by a child class")
    
    def transform(self, q, coeff_name):
        raise NotImplementedError("'transform' must be overriden by a child class")
    
    def timed_transform(self, q, coeff_name):
        t0 = time.time()
        y = self.transform(q, coeff_name)
        t1 = time.time()
        print(f"Interpolation time: {t1-t0}")
        return y


class OriginOnlyInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, origin_idx=0):
        self.origin_idx = origin_idx
        super(OriginOnlyInterpolator, self).__init__(q_eq, coeff_dict)
    
    def fit(self):
        pass
    
    def transform(self, q, coeff_name):
        return self.coeff_dict[coeff_name][self.origin_idx]


class InverseDistanceWeightingInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, p=2):
        self.p = p
        super(InverseDistanceWeightingInterpolator, self).__init__(q_eq, coeff_dict)
    
    def fit(self):
        pass
    
    def transform(self, q, coeff_name):
        return self.idw(q, coeff_name)[0]

    def idw(self, q, coeff_name):
        coeffs = self.coeff_dict[coeff_name]
        weights = 1 / np.linalg.norm(self.q_eq - q, axis=1) ** self.p
        weights /= np.sum(weights)
        return np.sum([coeffs[i] * weights[i] for i in range(len(coeffs))], axis=0), weights
        

class LinearInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict):
        super(LinearInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.tri = Delaunay(self.q_eq)
        # create interpolants for the different coefficient matrices
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            self.interpolation[coeff_name] = LinearNDInterpolator(self.tri, self.coeff_dict[coeff_name])

    def transform(self, q, coeff_name):
        return self.interpolation[coeff_name](q).squeeze()
    

class NearestNeighborInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict):
        super(NearestNeighborInterpolator, self).__init__(q_eq, coeff_dict)
    
    def fit(self):
        self.tri = Delaunay(self.q_eq)
        # create interpolants for the different coefficient matrices
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            self.interpolation[coeff_name] = NearestNDInterpolator(self.tri, self.coeff_dict[coeff_name])

    def transform(self, q, coeff_name):
        return self.interpolation[coeff_name](q).squeeze()
    

class CloughTocherInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict):
        if len(q_eq[0]) > 2:
            raise RuntimeError("Clough-Tocher interpolation only works for 2D data")
        super(CloughTocherInterpolator, self).__init__(q_eq, coeff_dict)
    
    def fit(self):
        self.tri = Delaunay(self.q_eq)
        # create interpolants for the different coefficient matrices
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            self.interpolation[coeff_name] = CloughTocher2DInterpolator(self.tri, self.coeff_dict[coeff_name])

    def transform(self, q, coeff_name):
        return self.interpolation[coeff_name](q).squeeze()
    

class ThinPlateSplineInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict):
        super(ThinPlateSplineInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the SMT object
            tps = ThinPlateSpline(alpha=0.0)  # 0 Regularization
            # Fit the control and target points
            tps.fit(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            # Transform new points
            self.interpolation[coeff_name] = tps

    def transform(self, q, coeff_name):
        y = self.interpolation[coeff_name].transform(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)


class RBFInterpolator(Interpolator):

    def __init__(self, q_eq, coeff_dict, h=100.):
        self.h = h
        super(RBFInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the SMT object
            sm = RBF(d0=self.h, print_global=False)  # d0 is kernel bandwidth
            # Fit the control and target points
            sm.set_training_values(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            sm.train()
            # Transform new points
            self.interpolation[coeff_name] = sm

    def transform(self, q, coeff_name):
        y =  self.interpolation[coeff_name].predict_values(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)


class IDWInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, p=2, eps=0.):
        self.p = p
        self.eps = eps
        super(IDWInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        pass

    def transform(self, q, coeff_name):
        weights = self.calc_weights(q)
        if coeff_name in ["q_bar", "u_bar"]:
            return np.einsum("i, ij -> j", weights, self.coeff_dict[coeff_name])
        else:
            return np.einsum("i, ijk -> jk", weights, self.coeff_dict[coeff_name])
    
    def calc_weights(self, q):
        # weigh the different coordintates differently
        weighting = np.eye(3)
        weighting[2, 2] = 100

        q_dist = np.linalg.norm(np.tensordot(weighting, (self.q_eq - q), axis=1), axis=1)

        m_idx = np.argmin(q_dist)
        m = q_dist[m_idx] # minimum distance
        # If the minimum is 0 then just take that point
        if m == 0:
            weights_norm = np.zeros(np.shape(q_dist))
            weights_norm[m_idx] = 1
        # Otherwise compute all weights
        else:
            weights = 1 / (q_dist ** self.p + self.eps) # np.exp(-self.p * (q_dist + self.eps) / m) # 
            weights_norm = weights / np.sum(weights)
        return weights_norm


class KrigingInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, theta=[1.,]):
        self.theta = theta
        super(KrigingInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the SMT object
            sm = KRG(theta0=self.theta, print_global=False)
            # Fit the control and target points
            sm.set_training_values(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            sm.train()
            # Transform new points
            self.interpolation[coeff_name] = sm

    def transform(self, q, coeff_name):
        y =  self.interpolation[coeff_name].predict_values(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)
    

class RMTSInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict):
        super(RMTSInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the SMT object
            sm = RMTB(
                xlimits=np.array([[-60., 60.], [-60., 60.], [-40., 0.]]),
                order=3,
                num_ctrl_pts=5,
                energy_weight=10.,
                regularization_weight=100.,
                print_global=True)
            print(sm)
            # Fit the control and target points
            sm.set_training_values(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            sm.train()
            # Transform new points
            self.interpolation[coeff_name] = sm

    def transform(self, q, coeff_name):
        y = self.interpolation[coeff_name].predict_values(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)


class SquaredRegressionInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict):
        super(SquaredRegressionInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the SMT object
            sm = QP(print_global=False)
            # Fit the control and target points
            sm.set_training_values(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            sm.train()
            # Transform new points
            self.interpolation[coeff_name] = sm

    def transform(self, q, coeff_name):
        y =  self.interpolation[coeff_name].predict_values(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)
    

class LinearRegressionInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, theta=[.01,]):
        super(LinearRegressionInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the SMT object
            sm = LS(print_global=False)
            # Fit the control and target points
            sm.set_training_values(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            sm.train()
            # Transform new points
            self.interpolation[coeff_name] = sm

    def transform(self, q, coeff_name):
        y =  self.interpolation[coeff_name].predict_values(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)
    

class PolynomialRidgeRegressionInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, degree=2, alpha=.1):
        self.degree = degree
        self.alpha = alpha
        super(PolynomialRidgeRegressionInterpolator, self).__init__(q_eq, coeff_dict)
    
    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            polyreg = make_pipeline(PolynomialFeatures(self.degree), Ridge(alpha=self.alpha, fit_intercept=True))
            polyreg.fit(self.q_eq, np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            self.interpolation[coeff_name] = polyreg
    
    def transform(self, q, coeff_name):
        y = self.interpolation[coeff_name].predict(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)
    

class NaturalNeighborInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict):
        super(NaturalNeighborInterpolator, self).__init__(q_eq, coeff_dict)
    
    def fit(self):
        pass
    
    def transform(self, q, coeff_name):
        points = np.array(self.q_eq)
        values = np.array(self.coeff_dict[coeff_name])
        eps = 0.000001
        interp_range = [[q[0] - eps, q[0], 1j], [q[1] - eps, q[1], 1j], [q[2] - eps, q[2], 1j]]
        interpolation = np.zeros(self.coeff_dict[coeff_name][0].shape)
        if interpolation.ndim > 1:
            for i in range(self.coeff_dict[coeff_name][0].shape[0]):
                for j in range(self.coeff_dict[coeff_name][0].shape[1]):
                        interpolation[i, j] = naturalneighbor.griddata(points, values[:, i, j], interp_range)
        else:
            for i in range(len(self.coeff_dict[coeff_name][0])):
                interpolation[i] = naturalneighbor.griddata(points, values[:, i], interp_range)
        return interpolation

def testInterpolators1D():
    interpolation_methods = ["nn", "idw", "krg", "rbf", "qp", "ls"]
    np.random.seed(seed=20)
    x = np.concatenate([[0], np.random.uniform(-1, 1, 10)])
    y = []
    y.append(np.sin(2*np.pi*x))
    y.append([-0.3, 0.2, -0.7, -0.4, 0.5, 0.0, 1.0, 1.5, 0.9, 1.0, 1.2])
    y = np.vstack(y).T
    xq = np.linspace(-1, 1, 100)
    ytrue = []
    ytrue.append(np.sin(2*np.pi*xq))
    ytrue.append(np.full(len(xq), np.nan))
    ytrue = np.vstack(ytrue).T
    coeff_dict = {"q_bar": y}

    fig1, axs1 = plt.subplots(len(interpolation_methods)//2, 2, figsize=(3*len(interpolation_methods), 3))
    fig2, axs2 = plt.subplots(len(interpolation_methods)//2, 2, figsize=(3*len(interpolation_methods), 3))

    for j, interpolation_method in enumerate(interpolation_methods):
        if interpolation_method == "nn":
            pass
        else:
            interpolator = InterpolatorFactory(interpolation_method, np.atleast_2d(x).T, coeff_dict).get_interpolator()
            yq = np.array([interpolator.transform(xi, "q_bar") for xi in xq])
        for i, axs in enumerate([axs1, axs2]):
            axs = axs.flatten()
            axs[j].plot(xq, ytrue[:, i], '--', color="k", alpha=0.5)
            if interpolation_method == "nn":
                sort_idx = np.argsort(x)
                axs[j].plot(x[sort_idx], y[sort_idx, i], '-', color="tab:blue", ds="steps-mid")
            else:
                axs[j].plot(xq, yq[:, i], '-', color="tab:blue")
            axs[j].plot(x, y[:, i], 'o', color="tab:orange", zorder=3)
            axs[j].set_title(DISPLAY_NAMES[interpolation_method])
            axs[j].set_ylim(-3, 3)
    
    plt.show()


if __name__ == '__main__':
    testInterpolators1D()