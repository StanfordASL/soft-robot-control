import numpy as np

from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
from tps import ThinPlateSpline
from smt.surrogate_models import RBF, IDW, KRG, QP

import time

from jax import jit
from functools import partial



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
        elif interpolation_method == 'qp':
            interpolator = SquaredRegressionInterpolator(q_eq, coeff_dict)
        else:
            raise RuntimeError(f"Interpolation method not recognized: {interpolation_method}")
        self.interpolator = interpolator
    
    def get_interpolator(self):
        return self.interpolator

class Interpolator():
    def __init__(self, q_eq, coeff_dict):
        self.q_eq= q_eq
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
    def __init__(self, q_eq, coeff_dict, origin_idx=4):
        self.origin_idx = origin_idx
        super(OriginOnlyInterpolator, self).__init__(q_eq, coeff_dict)
    
    def fit(self):
        pass
    
    def transform(self, q, coeff_name):
        return self.coeff_dict[coeff_name][self.origin_idx].squeeze()


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
            # Create the tps object
            tps = ThinPlateSpline(alpha=0.0)  # 0 Regularization
            # Fit the control and target points
            tps.fit(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            # Transform new points
            self.interpolation[coeff_name] = tps

    def transform(self, q, coeff_name):
        y = self.interpolation[coeff_name].transform(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)


class RBFInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, h=20.):
        self.h = h
        super(RBFInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the tps object
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
    def __init__(self, q_eq, coeff_dict, p=2):
        self.p = p
        super(IDWInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the tps object
            sm = IDW(p=self.p, print_global=False)  # p is power factor
            # Fit the control and target points
            sm.set_training_values(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            sm.train()
            # Transform new points
            self.interpolation[coeff_name] = sm

    def transform(self, q, coeff_name):
        y =  self.interpolation[coeff_name].predict_values(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)


class KrigingInterpolator(Interpolator):
    def __init__(self, q_eq, coeff_dict, theta=[.01,]):
        self.theta = theta
        super(KrigingInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the tps object
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
    def __init__(self, q_eq, coeff_dict, theta=[.01,]):
        super(SquaredRegressionInterpolator, self).__init__(q_eq, coeff_dict)

    def fit(self):
        self.interpolation = {}
        for coeff_name in self.coeff_dict:
            # Create the tps object
            sm = QP(print_global=False)
            # Fit the control and target points
            sm.set_training_values(np.array(self.q_eq), np.array(self.coeff_dict[coeff_name]).reshape(len(self.coeff_dict[coeff_name]), -1))
            sm.train()
            # Transform new points
            self.interpolation[coeff_name] = sm

    def transform(self, q, coeff_name):
        y =  self.interpolation[coeff_name].predict_values(np.atleast_2d(q))
        return y.reshape(self.coeff_dict[coeff_name][0].shape)