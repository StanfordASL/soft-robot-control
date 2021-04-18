import os
import pickle
import numpy as np
from scipy.sparse import linalg, coo_matrix
import osqp


class QuadraticCost:
    """
    Method class for setting quadratic cost variables for an Optimal Control Problem (OCP)
    """

    def __init__(self, Q=None, R=None, Qf=None):
        self.Qf = Qf
        self.Q = Q
        self.R = R


class Point:
    """
    Variables can be set to keep track of a previous point
    """

    def __init__(self):
        self.step = None
        self.t = None
        self.q = None
        self.v = None
        self.u = None
        self.H = None
        self.K = None
        self.D = None
        self.M = None
        self.S = None
        self.f = None
        self.b = None
        self.q_next = None
        self.v_next = None
        self.dt = None


class SnapshotData:
    """
    A generic object for storing snapshot data when running Sofa with an open_loop_controller. The add_point() function
    gets called in open_loop_controller to add a point to the collection. save_snapshot() is an optional function that
    can be used to tell open_loop_controller to save a point.
    :param save_dynamics: True to save dynamics matrices for each snapshot, False to only save the point (default=True)
    """

    def __init__(self, save_dynamics=True):
        self.save_dynamics = save_dynamics
        if self.save_dynamics:
            self.dict = {
                't': [],
                'q': [],
                'v': [],
                'u': [],
                'H': [],
                'K': [],
                'D': [],
                'M': [],
                'S': [],
                'b': [],
                'f': [],
                'q+': [],
                'v+': [],
                'dt': -1,
            }
        else:
            self.dict = {
                't': [],
                'q': [],
                'v': [],
                'u': [],
                'q+': [],
                'v+': [],
                'dt': -1,
            }

    def add_point(self, point):
        if self.dict['dt'] == -1:
            self.dict['dt'] = point.dt
        self.dict['t'].append(point.t)
        self.dict['q'].append(point.q)
        self.dict['v'].append(point.v)
        self.dict['u'].append(point.u)
        self.dict['q+'].append(point.q_next)
        self.dict['v+'].append(point.v_next)
        if self.save_dynamics:
            self.dict['K'].append(point.K)
            self.dict['D'].append(point.D)
            self.dict['M'].append(point.M)
            self.dict['b'].append(point.b)
            self.dict['f'].append(point.f)
            self.dict['H'].append(point.H)
            self.dict['S'].append(point.S)

    def save_snapshot(self, *args):
        """
        Called in open_loop_controller.py simulation, provides an optional way to tell
        the simulator to save a point.
        """
        return True

    def save_data(self, filename):
        print('Saving snapshots data to {}...'.format(filename))
        save_data(filename, self.dict)
        print('Done.')

    def simulation_end(self, filename):
        """
        This function is called at the end of the open_loop_sequence
        """
        # Only save if there is something to save
        if self.dict['q']:
            self.save_data(filename)
        else:
            print('No snapshots to save.')


def get_x(robot):
    q = robot.tetras.position.value.flatten()
    v = robot.tetras.velocity.value.flatten()
    return qv2x(q, v)


def qv2x(q, v):
    return np.concatenate((v, q), axis=-1)  # Extends to multiple point case


def x2qv(x):
    # returns (q, v)
    if x.ndim == 1:
        n = x.shape[0] // 2
        return x[n:], x[:n]
    elif x.ndim == 2:  # Extends to multiple point case
        n = x.shape[-1] // 2
        return x[:, n:], x[:, :n]
    else:
        raise IndexError('Unable to process x.ndim > 2')


def save_data(filename, data):
    if not os.path.isdir(os.path.split(filename)[0]):
        os.mkdir(os.path.split(filename)[0])

    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def sparse_list_to_np_array(matrix_list):
    return np.asarray([matrix.todense() for matrix in matrix_list])


def turn_on_LDL_saver(preconditioner, filepath):
    preconditioner.findData('savingMatrixToFile').value = True
    preconditioner.findData('savingFilename').value = filepath
    preconditioner.findData('savingPrecision').value = 10


def turn_off_LDL_saver(preconditioner):
    preconditioner.findData('savingMatrixToFile').value = False


def extract_KDMb(robot, snapshots_dir, step, dt, dv, point):
    alpha = robot.odesolver.rayleighMass.value
    beta = robot.odesolver.rayleighStiffness.value
    node_mass = robot.mass.vertexMass.value
    num_nodes = robot.tetras.size.value
    # Load and parse the LDL matrix into the M, K, and D matrices
    LDL_file = os.path.join(snapshots_dir, 'temp/LDL_%05d.txt' % step)
    LDL = np.zeros((3 * num_nodes, 3 * num_nodes))
    with open(LDL_file, 'r') as file:
        for (i, line) in enumerate(file):
            if i >= 1:  # skip the first row because it contains nothing
                LDL[i - 1, :] = np.fromstring(line.strip().strip('[]'), sep=' ')

    # Delete the LDL matrix text file to save storage space
    os.remove(LDL_file)
    # Extract K, D, M matices using M = diag(m), LDL = M + hD + h^2K,

    M = node_mass * np.eye(3 * num_nodes)
    K = (LDL - (1 + dt * alpha) * M) / (dt ** 2 + dt * beta)
    D = alpha * M + beta * K

    b = LDL @ dv - dt * point.H @ np.atleast_1d(point.u)
    f = b / dt + ((dt + beta) * K + alpha * M) @ point.v

    return coo_matrix(K), coo_matrix(D), coo_matrix(M), b, f, coo_matrix(LDL)


def extract_H_matrix(robot):
    """
    Extracts H matrix (i.e. input matrix) at a specific time instance from Sofa, by accessing findData('constraint')
    :param robot:
    :return: H.T (Input matrix at specific time instance)
    """
    num_nodes = robot.tetras.size.value
    H_strings = robot.tetras.constraint.value.split('\n')[:-1]
    H = np.zeros((len(H_strings), num_nodes * 3))
    for (i, H_string) in enumerate(H_strings):
        # constraint_data has form [input_nbr nbr_nodes node1 dim1 dim2 dim3 node2 dim1 dim2 dim3 ...]
        constraint_data = np.fromstring(H_string, sep=' ')
        number_nodes = int(constraint_data[1])
        constraint_data = constraint_data[2:]

        # Get node numbers that are actively influenced by the constraint
        constraint_active_nodes = constraint_data[::4].copy().astype(int)
        H_matrix_init_elems = constraint_data[np.mod(np.arange(constraint_data.size), 4) != 0].reshape(number_nodes, 3)

        # Build H matrix in right shape
        constraint = np.zeros((num_nodes, 3))
        constraint[constraint_active_nodes.tolist(), :] = H_matrix_init_elems
        H[i, :] = constraint.flatten()  # flatten to get final constraint row vector

    return coo_matrix(H.T)


def extract_AB(K, D, M, H):
    """
    Returns state and input matrices for linear approximation of system wh
    :param K: Stiffness matrix (n x n) sparse array
    :param D: Damping matrix (n x n) sparse array
    :param M: Mass matrix (n x n) sparse array
    :param H: Input matrix (n x m) sparse array
    :return: A (state matrix), B (input matrix) np.array
    """
    if not isinstance(K, np.ndarray):
        Minv = linalg.inv(M.tocsc())
        K_tilde = Minv * K.tocsc()
        D_tilde = Minv * D.tocsc()
        H_tilde = Minv * H.transpose()

        A11 = -D_tilde.toarray()
        A12 = -K_tilde.toarray()
        A21 = np.eye(np.shape(A11)[0])
        A22 = np.zeros(np.shape(A12))
        A = np.block([[A11, A12], [A21, A22]])
        B = np.block([[H_tilde], [np.zeros(np.shape(H_tilde))]])
        return A, B

    else:
        Minv = np.linalg.inv(M)
        K_tilde = Minv @ K
        D_tilde = Minv @ D
        H_tilde = Minv @ H

        A11 = -D_tilde
        A12 = -K_tilde
        A21 = np.eye(np.shape(A11)[0])
        A22 = np.zeros(np.shape(A12))
        A = np.block([[A11, A12], [A21, A22]])
        B = np.block([[H_tilde], [np.zeros(np.shape(H_tilde))]])
        return A, B


def extract_AB_d(S, K, H, dt):
    """Discrete derivation as defined in ThieffryKruszewskiEtAl2019, Trajectory Tracking Control Design for Large-Scale
    Linear Dynamical Systems With Applications to Soft Robotics"""
    Sinv = np.linalg.inv(S)
    SinvK = Sinv @ K
    SinvH = Sinv @ H
    dim = np.shape(K)[0]
    A = np.block([[np.eye(dim) - dt ** 2 * SinvK, -dt * SinvK],
                  [dt * np.eye(dim) - dt ** 3 * SinvK, np.eye(dim) - dt ** 2 * SinvK]])
    B = np.block([[dt * SinvH], [dt ** 2 * SinvH]])
    return A, B


def zoh_linear(A, B, dt):
    """
    Zero-Order Hold discretization. Exact discretization method for linear systems under zero-order hold assumption
    :param A: State matrix (n x n)
    :param B: Input matrix (n x m)
    :param dt: Discretization timestep
    :return: A_d (n x n), B_d (n x m): Discretized system
    """
    em_upper = np.hstack((A, B))
    em_lower = np.hstack((np.zeros((B.shape[1], A.shape[0])),
                          np.zeros((B.shape[1], B.shape[1]))))
    ZOH = linalg.expm(np.vstack((em_upper, em_lower)) * dt)

    # Dispose of the lower rows
    ZOH = ZOH[:A.shape[0], :]
    Ad = ZOH[:A.shape[0], :A.shape[1]]
    Bd = ZOH[:A.shape[0], A.shape[1]:]
    return Ad, Bd


def zoh_affine(A, B, d, dt):
    """
    Zero-order hold discretization for affine system. Exact disc. method for affine systems under zero-order hold assum.
    :param A: n x n dimensional array (state matrix)
    :param B: n x m dimensional array (input matrix)
    :param d: n x 1 dimensional array (affine term)
    :param dt: timestep duration, fixed
    :return: Discretized matrices with same input size
    """
    B_ext = np.hstack((B, np.expand_dims(d, axis=-1)))
    A_d, B_d_ext = zoh_linear(A, B_ext, dt)
    B_d = B_d_ext[:, :-1]
    d_d = B_d_ext[:, -1]  # last column
    return A_d, B_d, d_d


def dict_lists_to_array(dict):
    """
    Transform all lists in dict to np arrays for more complex mathematical operations
    """
    for key in dict:
        if type(dict[key]) == list:
            dict[key] = np.asarray(dict[key])


def get_snapshot_dir():
    """
    Returns absolute path to snapshots directory, creates
    this directory if it does not exist.
    Used solely for tmp data saving
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    snapshots_dir = os.path.join(path, 'snapshots')
    if not os.path.isdir(snapshots_dir):
        print('Creating directory {}'.format(snapshots_dir))
        os.mkdir(snapshots_dir)
    if not os.path.isdir(os.path.join(snapshots_dir, 'temp')):
        print('Creating directory {}'.format(os.path.join(snapshots_dir, 'temp')))
        os.mkdir(os.path.join(snapshots_dir, 'temp'))
    return snapshots_dir


class Polyhedron:
    def __init__(self, A, b, with_reproject=False):
        self.A = A
        self.b = b

        self.with_reproject = with_reproject

        if self.with_reproject:
            from scipy import sparse
            P = sparse.eye(self.A.shape[1]).tocsc()
            q = np.ones(self.A.shape[1])
            u = self.b
            A = sparse.csc_matrix(self.A)
            l = -np.inf * np.ones_like(self.b)
            self.osqp_prob = osqp.OSQP()
            self.osqp_prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
            self.osqp_prob.solve()

    def contains(self, x):
        """
        Returns true if x is contained in the Polyhedron
        """
        if np.max(self.A @ x - self.b) > 0:
            return False

        else:
            return True

    def get_constraint_violation(self, x):
        """
        Returns distance to constraint, i.e. how large the deviation is
        """
        return np.linalg.norm(np.maximum(self.A @ x - self.b, 0))

    def project_to_polyhedron(self, x):
        if not self.with_reproject:
            raise RuntimeError('Reproject not specified for class instance, set with_reproject=True to enable'
                               'reprojection to the Polyhedron through a QP')

        self.osqp_prob.update(q=-x)
        results = self.osqp_prob.solve()
        x_proj_alt = results.x
        return x_proj_alt


class HyperRectangle(Polyhedron):
    def __init__(self, ub, lb):
        n = len(ub)
        A = np.block(np.kron(np.eye(n), np.array([[1], [-1]])))
        b = np.hstack([np.array([ub[i], -lb[i]]) for i in range(n)])
        super(HyperRectangle, self).__init__(A, b)


def arr2np(x, dim, squeeze=False):
    """
    Converts python list to (-1, dim) shape numpy array
    """
    if squeeze:
        return np.asarray(x, dtype='float64').reshape(-1, dim).squeeze()
    else:
        return np.asarray(x, dtype='float64').reshape(-1, dim)


def np2arr(x):
    """ 
    Converts from numpy array to python list
    """
    return x.flatten().tolist()
