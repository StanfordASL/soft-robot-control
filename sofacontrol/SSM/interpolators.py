import numpy as np

from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
# from tps import ThinPlateSpline
from smt.surrogate_models import RBF, IDW, KRG, QP, LS, RMTB

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import naturalneighbor

import time
import matplotlib.pyplot as plt

from jax import jit
from functools import partial

from scipy.spatial import cKDTree

np.set_printoptions(linewidth=200)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'FreeSerif'})
plt.rcParams.update({'mathtext.fontset': 'cm'})

FONTSCALE = 1.2

plt.rc('font', size=12*FONTSCALE)          # controls default text sizes
plt.rc('axes', titlesize=15*FONTSCALE)     # fontsize of the axes title
plt.rc('axes', labelsize=13*FONTSCALE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('legend', fontsize=11*FONTSCALE)    # legend fontsize
plt.rc('figure', titlesize=15*FONTSCALE)   # fontsize of the figure title
suptitlesize = 20*FONTSCALE

plt.rc('figure', autolayout=True)

DISPLAY_NAMES = {
    "origin_only": "origin only",
    "idw": "Inverse distance weighting",
    "modified_idw": "Modified IDW",
    "linear": "Barycentric linear",
    "nn": "Nearest neighbor",
    "ct": "Clough-Tocher",
    "tps": "Thin plate spline",
    "rbf": "Radial basis function",
    "krg": "Kriging",
    "qp": "Quadratic polynomial regression",
    "ls": "Linear regression",
    "rmts": "RMTS",
    "natural_neighbor": "Natural neighbor"
}

class InterpolatorFactory():
    def __init__(self, interpolation_method, q_eq, coeff_dict):
        if interpolation_method == 'origin_only':
            interpolator = OriginOnlyInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'idw':
            interpolator = IDWInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'modified_idw':
            interpolator = ModifiedIDWInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'linear':
            interpolator = LinearInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'nn':
            # interpolator = NearestNeighborInterpolator(q_eq, coeff_dict)
            interpolator = IDWInterpolator(q_eq, coeff_dict, p=np.inf)
        elif interpolation_method == 'ct':
            interpolator = CloughTocherInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'tps':
            interpolator = ThinPlateSplineInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'rbf':
            interpolator = RBFInterpolator(q_eq, coeff_dict)
        elif interpolation_method == 'krg':
            interpolator = KrigingInterpolator(q_eq, coeff_dict)
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

    def __init__(self, q_eq, coeff_dict, h=10.):
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
        if coeff_name in ["q_bar", "x_bar", "u_bar"]:
            return np.einsum("i, ij -> j", weights, self.coeff_dict[coeff_name])
        else:
            return np.einsum("i, ijk -> jk", weights, self.coeff_dict[coeff_name])
    
    def calc_weights(self, q):
        # # weigh the different coordintates differently
        # weighting = np.eye(3)
        # for i in range(len(q)):
        #     weighting[i, i] = 1/ (np.max([q_eq[i] for q_eq in self.q_eq]) - np.min([q_eq[i] for q_eq in self.q_eq]))
        # # weighting[2, 2] = 50
        # q_dist = np.linalg.norm([weighting @ (self.q_eq[i] - q) for i in range(len(self.q_eq))], axis=1)
        # print(q)
        # var = np.var(np.array(self.q_eq), axis=0)
        # var = np.ones(np.shape(q))
        q_dist = np.linalg.norm((self.q_eq - q), axis=1)
        # q_dist = np.array([np.sqrt((q - self.q_eq[i]).T @ cov_inv @ (q - self.q_eq[i])) for i in range(len(self.q_eq))])
        # print(q_dist)
        m_idx = np.argmin(q_dist)
        m = q_dist[m_idx] # minimum distance
        # If the minimum is 0 or if only one model is available, then just take that point
        if m == 0 or len(q_dist) == 1 or self.p == np.inf:
            weights_norm = np.zeros(np.shape(q_dist))
            weights_norm[m_idx] = 1
        # Otherwise compute all weights
        else:
            R = np.max(q_dist)
            weights = ((R - q_dist) / (R * q_dist)) ** self.p
            # weights = 1 / (q_dist ** self.p + self.eps) # np.exp(-self.p * (q_dist + self.eps) / m) # 
            weights_norm = weights / np.sum(weights)
        return weights_norm
    

class ModifiedIDWInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, p=2, n_neighbors=8, eps=0.):
        self.p = p
        self.n_neighbors = n_neighbors
        self.eps = eps
        super(ModifiedIDWInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.kdtree = cKDTree(np.array(self.q_eq), leafsize=10, compact_nodes=False, balanced_tree=False)

    def transform(self, q, coeff_name):
        weights = self.calc_weights(q)
        if coeff_name in ["q_bar", "x_bar", "u_bar"]:
            return np.einsum("i, ij -> j", weights, self.coeff_dict[coeff_name])
        else:
            return np.einsum("i, ijk -> jk", weights, self.coeff_dict[coeff_name])

    def calc_weights(self, q):
        distances, idx = self.kdtree.query(q, k=self.n_neighbors)
        weights = np.zeros(len(self.q_eq))
        if self.n_neighbors == 1:
            weights[idx] = 1
            return weights
        elif len(self.q_eq) < self.n_neighbors:
            raise ValueError("Not enough neighbors to compute the weights")
        else:
            R = np.max(distances)
            # w = 1.0 / (distances ** self.p + self.eps)
            w = ((R - distances) / (R * distances)) ** self.p
            # w = np.exp(-self.p * (distances + self.eps) / np.min(distances))
            w /= np.sum(w)
            weights[idx] = w
            return weights


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
    interpolation_methods = ["nn", "idw", "modified_idw", "linear", "ct", "qp"]
    np.random.seed(seed=15) # 8, 10, 2023, 
    x = np.concatenate([[0], np.random.uniform(-1, 1, 10)]) * 2
    y = []
    y.append(np.sin(2*np.pi*x))
    y.append([-0.3, 0.2, -0.7, -0.4, 0.5, 0.0, 1.0, 1.5, 0.9, 1.0, 1.2])
    y = np.vstack(y).T
    xq = np.linspace(-2, 2, 1001)
    ytrue = []
    ytrue.append(np.sin(2*np.pi*xq))
    ytrue.append(np.full(len(xq), np.nan))
    ytrue = np.vstack(ytrue).T
    coeff_dict = {"q_bar": y}

    fig1, axs1 = plt.subplots(len(interpolation_methods)//2, 2, figsize=(8, 7))
    fig2, axs2 = plt.subplots(len(interpolation_methods)//2, 2, figsize=(8, 7), sharex=True, sharey=True)

    for j, interpolation_method in enumerate(interpolation_methods):
        if interpolation_method in ["nn", "linear"]:
            pass
        elif interpolation_method == "ct":
            yq = interp1d(x, y.T, kind="cubic", bounds_error=False)(xq).T
        else:
            interpolator = InterpolatorFactory(interpolation_method, np.atleast_2d(x).T, coeff_dict).get_interpolator()
            yq = np.array([interpolator.transform(xi, "q_bar") for xi in xq])
        for i, axs in enumerate([axs1, axs2]):
            axs = axs.flatten()
            axs[j].plot(xq, ytrue[:, i], '--', color="#a8a8a8")
            if interpolation_method == "nn":
                sort_idx = np.argsort(x)
                axs[j].plot(x[sort_idx], y[sort_idx, i], '-', color="tab:blue", ds="steps-mid")
            elif interpolation_method == "linear":
                sort_idx = np.argsort(x)
                axs[j].plot(x[sort_idx], y[sort_idx, i], '-', color="tab:blue")
            else:
                axs[j].plot(xq, yq[:, i], '-', color="tab:blue")
            axs[j].plot(x, y[:, i], 'o', color="tab:orange", zorder=3)
            axs[j].set_title(DISPLAY_NAMES[interpolation_method])
            axs[j].set_ylim(-2.5, 2.5)
            axs[j].set_xlim(-2, 2)
            axs[j].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    fig1.savefig("interpolation_1d_sin-wave.eps", dpi=300)
    fig2.savefig("interpolation_1d_random.eps", dpi=300)
    plt.show()


if __name__ == '__main__':
    testInterpolators1D()