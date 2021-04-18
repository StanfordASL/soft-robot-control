import os

import numpy as np
from scipy.sparse import coo_matrix

import sofacontrol.utils as scutils


class POD:
    """
    POD object
    """

    def __init__(self, POD_info):
        self.q_ref = POD_info['q_ref']
        self.v_ref = POD_info['v_ref']
        self.x_ref = scutils.qv2x(self.q_ref, self.v_ref)
        self.U = POD_info['U']
        self.V = np.kron(np.eye(2), self.U)
        self.rom_dim = self.U.shape[1]

    def compute_FO_state(self, q=None, v=None, x=None):
        """
        Compute full order approximate reconstruction of vector
        :param q: Reduced order position
        :param v: Reduced order velocity
        :param x: Reduced order state
        :return: Approximate full order vector
        """
        if q is not None:
            return self.U @ q + self.q_ref
        elif v is not None:
            return self.U @ v + self.v_ref
        elif x is not None:
            return self.V @ x + self.x_ref
        else:
            raise RuntimeError('Must specify vector type')

    def compute_RO_state(self, qf=None, vf=None, xf=None):
        """
        Compute reduced order projection of vector
        :param qf: Full order position
        :param vf: Full order velocity
        :param xf: Full order state
        :return: Reduced order vector
        """
        if qf is not None:
            return self.U.T @ (qf - self.q_ref)
        elif vf is not None:
            return self.U.T @ (vf - self.v_ref)
        elif xf is not None:
            return self.V.T @ (xf - self.x_ref)
        else:
            raise RuntimeError('Must specify vector type')

    def compute_RO_matrix(self, matrix, left=False, right=False):
        """
        Compute matrix in reduced order space, by projection with U
        :param matrix: matrix in High-dimensional space
        :param left: If True, multiplies matrix on the left by U^T
        :param right: If True, multiplies matrix on the right by U
        :return: matrix in low dimensional space
        """
        if not isinstance(matrix, (np.ndarray, coo_matrix)):
            raise RuntimeError('Matrix is not numpy ndarray or sparse coo_matrix')

        if (left and right) or (not left and not right):
            return self.U.T @ matrix @ self.U
        if left:
            return self.U.T @ matrix
        elif right:
            return matrix @ self.U

    def get_info(self):
        """
        Return dictionary with relevant info to recreate the ROM
        """
        return {'q_ref': self.q_ref, 'v_ref': self.v_ref, 'U': self.U, 'type': 'POD'}


class pod_config():
    """
    Object specifying all POD options and their default values
    """

    def __init__(self):
        self.pod_type = 'v'  # current 'v' or 'q'
        self.pod_tolerance = 0.0001
        self.preprocess = []  # string names of preprocess options to run on data
        self.preprocess_args = {'nbr_clusters': 0}


def load_POD(POD_file):
    """
    Loads data saved from run_POD and returns the POD model
    """
    if not os.path.isfile(POD_file):
        raise RuntimeError('POD file specified is not a valid file')

    # Load data from file
    print('Loading POD data from {}'.format(POD_file))
    POD_data = scutils.load_data(POD_file)

    # Generate POD object
    rom = POD(POD_data['POD_info'])

    return rom


def run_POD(snapshots_file, POD_file, config, rom_dim=None):
    """
    :param snapshots_file: absolute path to snapshot.pkl file that contains the snapshots to be used
    :param POD_file: absolute path to where the POD data file should be saved
    :param config: pod_config object with parameters
    :param rom_dim: (optional) number of dimensions to keep in the ROM, default to keep 99.9% "energy"
    """

    """
    TODO:
    1. Add ability to combine v and q (and maybe acceleration too)
    2. Better way of specifying reference value
    3. Make rom_dim specification possible a priori
    """

    # Load snapshots file
    data = scutils.load_data(snapshots_file)
    snapshots = get_snapshots(data, config.pod_type)

    # Run additional preprocess functions on the snapshots
    snapshots = process_snapshots(snapshots, config.preprocess, config.preprocess_args)

    # Run SVD to compute modes
    U_full, U, rom_dim, Sigma = compute_POD(snapshots.T, config.pod_tolerance)
    print('Computed POD with tolerance {}, resulting in {} dimensional system'.format(config.pod_tolerance, rom_dim))

    # Save this stuff
    POD_info = {'U': U, 'q_ref': data['q'][0], 'v_ref': np.zeros(data['v'][0].shape)}
    results = {'POD_info': POD_info, 'config': vars(config), 'Sigma': Sigma}
    print('Saving POD data to {}'.format(POD_file))
    scutils.save_data(POD_file, results)
    return results


def get_snapshots(data, pod_type):
    """
    Extracts the snapshots from the data that was previously saved
    """
    if pod_type == 'q':
        snapshots = np.asarray(data['q']) - data['q'][0]
    elif pod_type == 'v':
        snapshots = np.asarray(data['v'])
    elif pod_type == 'a':
        snapshots = np.asarray(data['v+']) - np.asarray(data['v'])
    return snapshots


def process_snapshots(snapshots, preprocess, args):
    """
    Process a snapshot vector based on specification of preprocess. Always considers snapshot w.r.t. reference value
    defined as the first vector in the snapshot
    """
    if 'normalize' in preprocess:
        # add small constant to avoid division by zero error
        # usage not advised
        snapshots = (snapshots - snapshots.min(axis=0)) / (snapshots.max(axis=0) + 1e-15 - snapshots.min(axis=0))

    if 'substract_mean' in preprocess:
        snapshots -= snapshots.mean(axis=0, keepdims=True)

    if 'clustering' in preprocess:
        if args['nbr_clusters'] > 0:
            snapshots = compute_kmeans_centroids(snapshots, args['nbr_clusters'])
        else:
            print('Not using kmeans because nbr_clusters not specified in config.preprocess_args dictionary')

    # Optionally add other preprocessing possibilities

    return snapshots


def compute_POD(snapshots, tol, rom_dim=None):
    """
    Computes Proper Orthogonal Decomposition on a snapshot matrix (stores states at different point of simulation)
    :param snapshots: np.array, nf x num_snapshots.
    :param tol: (optional) float in (0,1) of "energy" of singular values to discard, default = 0.0001
    :param rom_dim: (optional) integer number of dimensions to keep, overrides tol
    :return: U_full (all left singular vectors), U (projection matrix), Sigma (all singular values)
    """

    U_full, S, V = np.linalg.svd(snapshots, full_matrices=False)

    # Determining nbr of nodes to satisfy tolerance
    s_square = S ** 2
    i = 0
    while (np.sum(s_square[i:]) / np.sum(s_square)) > tol or i == 0:
        i += 1

    nbModes = i
    U = U_full[:, 0:nbModes]
    return U_full, U, nbModes, S


def decrease_dimension():
    NotImplementedError('Not implemented')


def compute_kmeans_centroids(snapshot, k):
    """
    k means algorithm. Extracts kmeans centroids that will be used for computing POD
    """
    from sklearn.cluster import KMeans
    print('Computing %d centroids for an initial snapshot size of %d using k-means clustering'
          % (k, snapshot.shape[0]))
    kmeans = KMeans(k, n_init=100, max_iter=1000, random_state=0).fit(snapshot)
    snapshot_update = kmeans.cluster_centers_
    return snapshot_update
