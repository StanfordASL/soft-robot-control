import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import lil_matrix, vstack


class linearModel:
    """ Assumes state vector x = [v; q] and q = [n1.x; n1.y; n1.z; ...]. 
    This class builds a measurement model y = Cx, callable as y = class.evaluate(x) 

    Inputs:
        nodes: list of integer node numbers starting from 0
        num_nodes: total number of nodes (158 for finger)
        pos: True (default) to output position q of each node in nodes
        vel: True (default) to output velocity v of each node in nodes

    y = class.evaluate(x) where x = [v; q] outputs the measurement with 
    [q1;v1;q2;v2;...] format.

    """

    def __init__(self, nodes, num_nodes, pos=True, vel=True):
        self.pos = pos
        self.vel = vel
        self.build_C_matrix(nodes, num_nodes)
        self.num_nodes = num_nodes

    def build_C_matrix(self, nodes, num_nodes):
        if self.vel and not self.pos:
            self.C = buildCv(nodes, num_nodes)
        elif self.pos and not self.vel:
            self.C = buildCq(nodes, num_nodes)
        else:
            Cv = buildCv(nodes, num_nodes)
            Cq = buildCq(nodes, num_nodes)
            self.C = vstack((Cv, Cq))

    def evaluate(self, x):
        return self.C @ x


class MeasurementModel(linearModel):
    def __init__(self, nodes, num_nodes, pos=True, vel=True, mu_q=None, S_q=None, mu_v=None, S_v=None):
        super().__init__(nodes, num_nodes, pos=pos, vel=vel)

        if pos and vel:
            pos_dim = self.C.shape[0] // 2
            vel_dim = self.C.shape[0] // 2
        elif pos and not vel:
            pos_dim = self.C.shape[0]
            vel_dim = 0
        elif vel and not pos:
            pos_dim = 0
            vel_dim = self.C.shape[0]

        if mu_q is None:
            mu_q = np.zeros(pos_dim)
        if mu_v is None:
            mu_v = np.zeros(vel_dim)
        if S_q is None:
            S_q = np.zeros((pos_dim, pos_dim))
        if S_v is None:
            S_v = np.zeros((vel_dim, vel_dim))

        self.mean = np.concatenate((mu_v, mu_q))
        self.covariance = block_diag(S_v, S_q)

        assert self.mean.shape[0] == self.C.shape[0]
        assert self.covariance.shape[0] == self.C.shape[0] and self.covariance.shape[1] == self.C.shape[0]

    def evaluate(self, x):
        return self.C @ x + np.random.multivariate_normal(mean=self.mean, cov=self.covariance)


def buildCq(nodes, num_nodes):
    Cq = lil_matrix((3 * len(nodes), 6 * num_nodes))
    for (i, node) in enumerate(nodes):
        Cq[3 * i, 3 * num_nodes + 3 * node] = 1.
        Cq[3 * i + 1, 3 * num_nodes + 3 * node + 1] = 1.
        Cq[3 * i + 2, 3 * num_nodes + 3 * node + 2] = 1.
    return Cq


def buildCv(nodes, num_nodes):
    Cv = lil_matrix((3 * len(nodes), 6 * num_nodes))
    for (i, node) in enumerate(nodes):
        Cv[3 * i, 3 * node] = 1.
        Cv[3 * i + 1, 3 * node + 1] = 1.
        Cv[3 * i + 2, 3 * node + 2] = 1.
    return Cv
