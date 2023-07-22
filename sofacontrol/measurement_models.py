import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import lil_matrix, vstack
from sofacontrol.utils import x2qv

class OutputModel:
    """ Assumes state vector y = [y1, ..., yn] representing the vector of measurable outputs
        including, potentially, time delays of the outputs
        This class builds a measurement model z = Cy, callable as z = class.evaluate(y)
        Assumes that the performance variables at the current time-step are located at the end of the vector

        Inputs:
            num_obs: Number of observed quantities (dimension of y)
            num_perf: Number of performance variables (no time-delay)

        z = class.evaluate(y) outputs the performance variable at the current time-step
        from the vector of observed variables (with time-delay)
        C = class.C returns the selection matrix of scalar observables deemed as performance
        variables
        """
    def __init__(self, num_obs, num_perf):
        self.num_obs = num_obs
        self.num_perf = num_perf
        self.build_C_matrix(num_obs, num_perf)

    def build_C_matrix(self, num_obs, num_perf):
        # Generate output matrix (observed to output) - Assumed desired output is in the end of the vector
        # TODO: Probably should refactor this later
        self.C = np.zeros((self.num_perf, self.num_obs))
        for i in range(self.num_perf):
            self.C[self.num_perf - 1 - i, self.num_obs - 1 - i] = 1

    def evaluate(self, y):
        return self.C @ y

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

    def __init__(self, nodes, num_nodes, pos=True, vel=True, qv=False):
        self.pos = pos
        self.vel = vel
        self.build_C_matrix(nodes, num_nodes)
        self.num_nodes = num_nodes
        # self.qv = qv

    def build_C_matrix(self, nodes, num_nodes):
        if self.vel and not self.pos:
            self.C = buildCv(nodes, num_nodes)
        elif self.pos and not self.vel:
            self.C = buildCq(nodes, num_nodes)
        else:
            Cv = buildCv(nodes, num_nodes)
            Cq = buildCq(nodes, num_nodes)
            self.C = vstack((Cv, Cq)) # vstack((Cq, Cv)) # 
    def evaluate(self, x, qv=False):
        z = self.C @ x
        if qv:
            return np.concatenate(x2qv(z))
        else:
            return z


class MeasurementModel(linearModel):
    def __init__(self, nodes, num_nodes, pos=True, vel=True, mu_q=None, S_q=None, mu_v=None, S_v=None, qv=False):
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

        # Set format of output
        self.qv = qv

        assert self.mean.shape[0] == self.C.shape[0]
        assert self.covariance.shape[0] == self.C.shape[0] and self.covariance.shape[1] == self.C.shape[0]

    def evaluate(self, x):
        z = self.C @ x + np.random.multivariate_normal(mean=self.mean, cov=self.covariance)
        if self.qv:
            return np.concatenate(x2qv(z))
        else:
            return z


def buildCq(nodes, num_nodes):
    Cq = lil_matrix((3 * len(nodes), 6 * num_nodes))
    # Format of x: [vx_0, vy_0, vz_0, (up to num_nodes), qx_0, qy_0, qz_0,...]
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
