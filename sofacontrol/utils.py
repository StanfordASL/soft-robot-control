import os
import pickle
import numpy as np
from scipy.sparse import linalg, coo_matrix
from scipy.linalg import block_diag
import osqp
import jax
import jax.numpy as jnp
import lzma
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import jax.scipy as jsp
from functools import partial
import matplotlib.image as mpimg
import scipy.stats as stats

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

def get_q(robot):
    return robot.tetras.position.value.flatten()

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

def vq2qv(x):
    q, v = x2qv(x)
    return np.hstack((q, v))

def save_data(filename, data):
    if not os.path.isdir(os.path.split(filename)[0]):
        os.mkdir(os.path.split(filename)[0])
    # with lzma.open(filename, 'wb') as file:
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    # try:
    #     # recently collected data should now be compressed using lzma
    #     with lzma.open(filename, 'rb') as file:
    #         data = pickle.load(file)
    # except:
    #     # backwards compatibility with uncompressed data
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def sparse_list_to_np_array(matrix_list):
    return np.asarray([matrix.todense() for matrix in matrix_list])


def turn_on_LDL_saver(matrixExporter, filepath):
    matrixExporter.findData('enable').value = True
    # Export every 10th step e.g., if dt = 0.01, save every 0.1 sec
    matrixExporter.findData('exportEveryNumberOfSteps').value = 10
    matrixExporter.findData('filename').value = filepath
    matrixExporter.findData('format').value = 'txt'
    matrixExporter.findData('precision').value = 10


def turn_off_LDL_saver(matrixExporter):
    matrixExporter.findData('enable').value = False
    matrixExporter.findData('exportEveryNumberOfSteps').value = 0


def extract_KDMb(robot, filesInDir, step, dt, dv, point):
    alpha = robot.odesolver.rayleighMass.value
    beta = robot.odesolver.rayleighStiffness.value
    node_mass = robot.mass.vertexMass.value
    num_nodes = robot.tetras.size.value
    # Load and parse the LDL "global" matrix
    LDL_file = filesInDir[0]
    LDL = np.zeros((3 * num_nodes, 3 * num_nodes))
    with open(LDL_file, 'r') as file:
        for (i, line) in enumerate(file):
            if i >= 1:  # skip the first row because it contains nothing
                LDL[i - 1, :] = np.fromstring(line.strip().strip('[]'), sep=' ')

    # Delete the LDL matrix text file to save storage space
    os.remove(LDL_file)

    # Extract K matrix
    K = extract_K_matrix(robot)

    # Get M and D matrix (D matrix is simply proportional damping)
    M = node_mass * np.eye(3 * num_nodes)
    D = alpha * M + beta * K

    b = LDL @ dv - dt * point.H @ np.atleast_1d(point.u)
    f = b / dt + ((dt + beta) * K + alpha * M) @ point.v

    return coo_matrix(K), coo_matrix(D), coo_matrix(M), b, f, coo_matrix(LDL)

def extract_K_matrix(robot):
    K = -robot.forcefield.assembleKMatrix().toarray()
    num_q = np.shape(K)[0]
    constrain_node = np.zeros((3, num_q))
    for DOF in robot.constraints.points.toList():
        # Set columns and rows of constrained nodes to zero
        DOF = DOF[0]
        K[3*DOF:3*DOF + 3, :] = constrain_node
        K[:, 3 * DOF:3 * DOF + 3] = constrain_node.T

        # Set diagonals of constrained nodes to 1
        for i in range(3):
            K[3*DOF + i, 3*DOF + i] = 1

    return K

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

def delayEmbedding(undelayedData, embed_coords=[0, 1, 2], up_to_delay=4):
    undelayed = undelayedData[embed_coords, :]
    buf = [undelayed]
    for delta in list(range(1, up_to_delay+1)):
        delayed_by_delta = np.roll(undelayed, -delta)
        delayed_by_delta[:, -delta:] = 0
        buf.append(delayed_by_delta)
    delayedData = np.vstack(buf)
    return delayedData

class Polyhedron:
    def __init__(self, A, b, with_reproject=False):
        self.A = A
        self.b = b
        self.Ak = None
        self.bk = None

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

    def get_constraint_violation(self, x, z=None, update=False):
        """
        Returns distance to constraint, i.e. how large the deviation is
        """
        if update:
            return np.linalg.norm(np.maximum(self.Ak @ x - self.bk, 0))
        else:
            return np.linalg.norm(np.maximum(self.A @ x - self.b, 0))

    def project_to_polyhedron(self, x):
        if not self.with_reproject:
            raise RuntimeError('Reproject not specified for class instance, set with_reproject=True to enable'
                               'reprojection to the Polyhedron through a QP')

        self.osqp_prob.update(q=-x)
        results = self.osqp_prob.solve()
        x_proj_alt = results.x
        return x_proj_alt

    def update(self, Hk, ck):
        """
        self.Ak and self.bk is constructed so that constraints are with respect to observable coordinates
        """
        self.Ak = np.dot(self.A, Hk)
        self.bk = self.b - np.dot(self.A, ck)

class CircleObstacle(Polyhedron):
    def __init__(self, A, center, diameter):
        self.diameter = diameter
        self.center = center
        super(CircleObstacle, self).__init__(A, None)
    
    
    def contains(self, x):
        """
        Returns true if x is not in the obstacle
        """
        
        # Check if x is within the diameter of the circle
        for j in range(self.center.shape[0]):
            if np.linalg.norm(x - self.center[j]) < (self.diameter[j] / 2):
                return True
        
        return False
    
    def get_constraint_violation(self, x, z=None, update=False):
        """
        Returns distance to constraint, i.e. how large the deviation is
        """
        if z is None:
            if update:
                z = self.Ak @ x + self.bk
            else:
                z = self.A @ x
        
        constraintVals = []
        
        # For each circle, add the maximum of the distance to the circle and 0 to constraintVals
        # TODO: Pretty inefficient since we're looping over all circles for each point
        for j in range(self.center.shape[0]):
            constraintVals.append(np.maximum((self.diameter[j] / 2 - np.linalg.norm(z - self.center[j])), 0))
        
        # Return maximum of all constraint violations
        return np.max(constraintVals)

    def update(self, Hk, ck):
        """
        self.Ak and self.bk is constructed so that constraints are with respect to observable coordinates
        """
        self.Ak = np.dot(self.A, Hk)
        self.bk = np.dot(self.A, ck)
    
    def project_to_boundary(self, x0):
        """
        Projects the point x0 onto the boundary of the closest circle.
        """
        # Find the closest circle center
        distances = [np.linalg.norm(x0 - c) for c in self.center]
        closest_idx = np.argmin(distances)
        closest_center = self.center[closest_idx]

        # If x0 is the center of the circle, return it as it is (no unique projection)
        if distances[closest_idx] == 0:
            return x0

        # Compute direction from center to x0
        direction = (x0 - closest_center) / distances[closest_idx]

        # Project x0 to the boundary
        projection = closest_center + direction * (self.diameter[closest_idx] / 2)

        return projection



class HyperRectangle(Polyhedron):
    def __init__(self, ub, lb):
        n = len(ub)
        A = np.block(np.kron(np.eye(n), np.array([[1], [-1]])))
        b = np.hstack([np.array([ub[i], -lb[i]]) for i in range(n)])
        super(HyperRectangle, self).__init__(A, b)

def pad_vector(x, y, replicate=False):
    """
    Pads numpy array x with dimension of y (assumes dim(x) < dim(y))
    (optional) replicate (Bool): Replicate vallues in extra dimension of y into x
    """
    pad_size = len(y) - len(x)
    if replicate:
        x_pad = np.concatenate((x, y[-pad_size:]))
    else:
        x_pad = np.pad(x, (0, pad_size))
    
    return x_pad

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

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

"""
Drawing trajectories utilities
"""


def drawWaypoints():
    # create an empty figure
    fig = plt.figure()

    # create an empty list to store your points
    points = []

    def onclick(event):
        # when the figure is clicked, append the point (x, y) to the points list
        points.append([event.xdata, event.ydata])
        # plot the point
        plt.plot(event.xdata, event.ydata, 'ro')
        if len(points) > 1:
            # if there are two or more points, plot a line between the last two points
            plt.plot([points[-2][0], points[-1][0]], [points[-2][1], points[-1][1]], 'b-')
        # refresh the plot
        plt.draw()

    # connect the click event to the figure
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # show the figure
    plt.xlim(-20, 20)  # Set the x-axis limits
    plt.ylim(-20, 20)  # Set the y-axis limits
    plt.grid(True)
    plt.show()

    return np.array(points)


def drawContinuousPath(distance_threshold=0.1, image_path=None, z_plane=False, num_repeat=1):
    # Create an empty figure
    fig, ax = plt.subplots()

    # Load and display the image, if image_path is provided
    if image_path is not None:
        img = mpimg.imread(image_path)
        if z_plane:
            ax.imshow(img, extent=[-12, 12, -12, 12])
        else:
            # ax.imshow(img, extent=[-25, 25, -25, 25])
            ax.imshow(img, extent=[-20, 20, -20, 20])

    # Create an empty list to store your points
    points = []

    # Variable that tracks whether the left mouse button is pressed
    is_pressed = False

    def on_press(event):
        nonlocal is_pressed
        is_pressed = True

    def on_release(event):
        nonlocal is_pressed
        is_pressed = False

    def on_motion(event):
        nonlocal is_pressed, points
        if is_pressed:
            if len(points) == 0 or np.linalg.norm(np.array([event.xdata, event.ydata]) - np.array(points[-1])) > distance_threshold:
                # When the mouse moves, append the point (x, y) to the points list
                points.append([event.xdata, event.ydata])
                # Plot the point
                ax.plot(event.xdata, event.ydata, 'ro')
                if len(points) > 1:
                    # If there are two or more points, plot a line between the last two points
                    ax.plot([points[-2][0], points[-1][0]], [points[-2][1], points[-1][1]], 'b-')
                # Refresh the plot
                fig.canvas.draw()

    # Connect the events to the figure
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # Show the figure
    ax.set_xlim(-30, 30)  # Set the x-axis limits
    ax.set_ylim(-30, 30)  # Set the y-axis limits
    plt.grid(True)
    plt.show()

    # Return the points
    return np.tile(points, (num_repeat, 1))


def resample_waypoints(waypoints, total_time, dt=0.01):
    waypoints = np.array(waypoints)
    x, y = waypoints.T

    # Compute the cumulative arc-length
    dx = np.diff(x)
    dy = np.diff(y)
    distance = np.cumsum(np.sqrt(dx ** 2 + dy ** 2))
    distance = np.insert(distance, 0, 0)

    # Total distance
    total_distance = distance[-1]

    # Interpolate the path
    path = CubicSpline(distance, waypoints, axis=0, bc_type='natural')

    # Sample the path at constant time intervals dt
    num_points = int(total_time / dt)
    new_distance = np.linspace(0, total_distance, num_points)
    new_waypoints = path(new_distance)

    return new_waypoints

"""
Custom JAX routine for jnp.norm. Takes difference between first len(y) components of x and y.
"""

def norm2Diff(x, y, P=None):
    idxTrunc = len(y) # assume dim(y) < dim(x)
    x = x[:idxTrunc]
    is_zero = jnp.allclose(x - y, 0.)
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    if P is not None:
        ans = jnp.linalg.norm(jnp.dot(jsp.linalg.sqrtm(P), x.T - y.T), axis=0)
    else:
        ans = jnp.linalg.norm(x - y)
    ans = jnp.where(is_zero, 0., ans)

    return jnp.max(jnp.array([ans, 0.001]))

def norm2Linearize(x, y, dt, P=None):
    fixed_y_norm2 = partial(norm2Diff, y=y, P=P)
    A = jax.jacobian(fixed_y_norm2)(x)
    c = fixed_y_norm2(x) - A @ x
    return A, c

"""
    Create a new target trajectory. TODO: Assume outdofs are [0, 1, 2]
"""
def createTargetTrajectory(controlTask, robot, z_eq_point, output_dim, amplitude=15, freq=None, pathToImage=None, 
                           outdofs=[0, 1, 2], z_offset=None, repeat_traj=1, Mper=3, Tper=10.):
    if controlTask == 'custom':
        # Check if the trajectory is along z-plane
        if outdofs == [0, 1, 2]:
            z_plane = False
        else:
            z_plane = True
        #############################################
        # Problem 0, Custom Drawn Trajectory
        #############################################
        # Draw the desired trajectory
        M = 1
        T = Tper
        points = drawContinuousPath(distance_threshold=0.5, image_path=pathToImage, z_plane=z_plane, num_repeat=repeat_traj)
        resampled_pts = resample_waypoints(points, M * T) # TODO: 10.1

        # Setup target trajectory
        t = np.linspace(0, M * T, resampled_pts.shape[0]) # TODO: 10.1 and resampled_pts.shape[0] + 1
        x_target, y_target = resampled_pts[:, 0], resampled_pts[:, 1]
        # zf_target = np.zeros((resampled_pts.shape[0] + 1, output_dim))
        zf_target = np.tile(np.hstack((z_eq_point, np.zeros(output_dim - len(z_eq_point)))), (resampled_pts.shape[0], 1))
        zf_target[:, outdofs[0]] += x_target
        zf_target[:, outdofs[1]] += y_target
    elif controlTask == "figure8":
        # === figure8 ===
        M = Mper
        T = Tper
        N = 1000
        radius = amplitude
        t = np.linspace(0, M * T, M * N + 1)
        th = np.linspace(0, M * 2 * np.pi, M * N + 1)
        zf_target = np.tile(np.hstack((z_eq_point, np.zeros(output_dim - len(z_eq_point)))), (M * N + 1, 1))
        # zf_target = np.zeros((M*N+1, 6))
        zf_target[:, outdofs[0]] += -radius * np.sin(th)
        zf_target[:, outdofs[1]] += radius * np.sin(2 * th)
        # zf_target[:, 2] += -np.ones(len(t)) * 20
    elif controlTask == "circle":
        if robot == 'trunk':
            # === circle (with constant z) ===
            M = 1
            T = 9 # 10
            N = 900 # 1000
            radius = amplitude
            t = np.linspace(0, M * T, M * N + 1)
            th = np.linspace(0, M * 2 * np.pi, M * N + 1) + np.pi/2
            zf_target = np.tile(np.hstack((z_eq_point, np.zeros(output_dim - len(z_eq_point)))), (M * N + 1, 1))
            # zf_target = np.zeros((M*N+1, 6))
            zf_target[:, outdofs[0]] += radius * np.cos(th)
            zf_target[:, outdofs[1]] += radius * np.sin(th)
            # zf_target[:, 2] += -np.ones(len(t)) * 20
            print(zf_target[0, :].shape)
            idle = np.repeat(np.atleast_2d(zf_target[0, :]), int(1/0.01), axis=0)
            print(idle.shape)
            zf_target = np.vstack([idle, zf_target])
            print(zf_target.shape)
            t = np.linspace(0, M * 10, M * 1000 + 1)
        elif robot == 'diamond':
            M = Mper
            T = Tper
            N = 1000
            t = np.linspace(0, M * T, M * N)
            th = np.linspace(0, M * 2 * np.pi, M * N)
            x_target = np.zeros(M * N)
            
            if freq is None:
                y_target = amplitude * np.sin(th)
                z_target = amplitude - amplitude * np.cos(th)
            else:
                y_target = amplitude * np.sin(freq * T / (2 * np.pi) * th)
                z_target = amplitude - amplitude * np.cos(freq * T / (2 * np.pi) * th)

            zf_target = np.zeros((M * N, output_dim))
            zf_target[:, outdofs[0]] = x_target
            zf_target[:, outdofs[1]] = y_target
            zf_target[:, outdofs[2]] = z_target
        else:
            raise RuntimeError('Requested robot not implemented. Must be trunk or diamond')
    elif controlTask == "pacman":
        M = 1 # 3
        T = 10
        N = 1000
        radius = 20.
        t = np.linspace(0, M * T, M * N + 1)
        th = np.linspace(0, M * 2 * np.pi, M * N + 1)
        zf_target = np.tile(np.hstack((z_eq_point, np.zeros(output_dim - len(z_eq_point)))), (M * N + 1, 1))
        # zf_target = np.zeros((M * N, model.output_dim))
        zf_target[:, outdofs[0]] += radius * np.cos(th)
        zf_target[:, outdofs[1]] += radius * np.sin(th)
        zf_target[:, outdofs[2]] += -np.ones(len(t)) * 10
        t_in_pacman, t_out_pacman = 1., 1.
        zf_target[t < t_in_pacman, :] = z_eq_point + (zf_target[t < t_in_pacman][-1, :] - z_eq_point) * (t[t < t_in_pacman] / t_in_pacman)[..., None]
        zf_target[t > T - t_out_pacman, :] = z_eq_point + (zf_target[t > T - t_out_pacman][0, :] - z_eq_point) * (1 - (t[t > T - t_out_pacman] - (T - t_out_pacman)) / t_out_pacman)[..., None]
        
    else:
        raise RuntimeError('Requested target not implemented. Must be figure8, circle, or custom')
    
    if z_offset is not None:
        zf_target += np.array([0., 0., z_offset])
    
    return zf_target, t

def load_full_equilibrium(root_path):
    # Load equilibrium point
    rest_file = os.path.join(root_path, 'rest_qv.pkl')
    rest_data = load_data(rest_file)
    q_equilibrium = np.array(rest_data['q'][0])

    # Setup equilibrium point (no time delay and observed position and velocity of tip)
    x_eq = qv2x(q=q_equilibrium, v=np.zeros_like(q_equilibrium))

    return x_eq

"""
    Generate a model. TODO: Only SSM for now
"""
def generateModel(root_path, pathToModel, nodes, num_nodes, modelType=None, isLinear=False):
    from sofacontrol.SSM import ssm
    import sofacontrol.measurement_models as msm

    # Setup equilibrium point (no time delay and observed position and velocity of tip)
    x_eq = load_full_equilibrium(root_path)

    # load SSM model
    if modelType is None:
        with open(os.path.join(pathToModel, 'SSM_model.pkl'), 'rb') as f:
            SSM_data = pickle.load(f)
    else:
        with open(os.path.join(pathToModel, 'SSM_model_' + modelType + '.pkl'), 'rb') as f:
            SSM_data = pickle.load(f)

    raw_model = SSM_data['model']
    raw_params = SSM_data['params']

    if raw_params['delay_embedding']:
        outputModel = msm.linearModel(nodes, num_nodes, vel=False)
        z_eq_point = outputModel.evaluate(x_eq, qv=False)
        outputSSMModel = msm.OutputModel(raw_params['obs_dim'], raw_params['output_dim'])
        Cout = outputSSMModel.C
    else:
        outputModel = msm.linearModel(nodes, num_nodes)
        z_eq_point = outputModel.evaluate(x_eq, qv=True)
        Cout = None

    model = ssm.SSMDynamics(z_eq_point, discrete=False, discr_method='be',
                            model=raw_model, params=raw_params, C=Cout, isLinear=isLinear)

    return model

def createControlConstraint(u_min, u_max, input_dim, du_max=None):
    U = HyperRectangle([u_max] * input_dim, [u_min] * input_dim)
    
    if du_max is not None:
        dU = HyperRectangle([du_max] * input_dim, [-du_max] * input_dim)
    else:
        dU = None
    
    return U, dU

"""
    Obstacle constraint in x-y plane. TODO: Only works for SSMR right now
"""
def createObstacleConstraint(output_dim, y_ref, obstacleDiameter, obstacleLoc):
    Hz = np.zeros((2, output_dim))
    Hz[0, 0] = 1
    Hz[1, 1] = 1
    X = CircleObstacle(A=Hz, center=obstacleLoc - Hz @ y_ref, diameter=obstacleDiameter)

    return X

import numpy as np
import time

def generateObstacles(num_obstacles, d_min, d_max, min_distance_from_origin, min_distance_between_obstacles=5):
    max_attempts_per_obstacle = 1000
    seed = int(time.time())  # Use the current time as a seed

    while True:  # Outer loop to keep trying with different seeds
        np.random.seed(seed)

        # Initialize the list of obstacles
        obstacles = []

        for _ in range(num_obstacles):  # Loop for each obstacle
            attempts = 0
            while attempts < max_attempts_per_obstacle:
                # Generate a random location
                location = np.random.uniform(-30, 30, size=2)

                # Generate a random diameter
                diameter = np.random.uniform(d_min, d_max)

                # Check if the new obstacle is too close to the origin
                if np.linalg.norm(location) < min_distance_from_origin:
                    attempts += 1
                    continue

                # Check if the new obstacle is too close to any existing obstacle
                intersection = False
                for existing_diameter, existing_location in obstacles:
                    distance_between_centers = np.linalg.norm(location - existing_location)
                    distance_between_perimeters = distance_between_centers - (diameter + existing_diameter) / 2
                    if distance_between_perimeters < min_distance_between_obstacles:
                        intersection = True
                        break
                
                # If no intersection, add the obstacle to the list
                if not intersection:
                    obstacles.append((diameter, location))
                    break

                attempts += 1

            # If max attempts reached for this obstacle, break out and try new seed
            if attempts == max_attempts_per_obstacle:
                break
        
        # If successfully generated all obstacles, return
        if len(obstacles) == num_obstacles:
            return obstacles

        # Otherwise, reset and try a different seed
        seed = np.random.randint(0, 2**32 - 1)  # Generate a new random seed

def confidence_interval(data, confidence=0.95, distance=True):
    n = data.shape[-1]  # assuming data is a 1D array
    m = np.mean(data, axis=-1)
    se = np.std(data, axis=-1, ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    if distance:
        return m - (m - h), (m + h) - m
    else:
        return m - h, m + h

def remove_decimal(value):
    return str(value).replace(".", "")

def add_decimal(s):
    if not s:
        return 0.0
    return float(s[0] + "." + s[1:])

#############################################
#
# Linear Disturbance Observer (LDO) Utils
#
#############################################

def create_circ_matrix(nd, Nperiod: int):
    # Create a block diagonal matrix with Nperiod copies of the identity matrix of size nd
    # The matrix should be of size (Nperiod*nd, Nperiod*nd)
    # CHECK nd and Nperiod are integers
    assert isinstance(nd, int)
    assert isinstance(Nperiod, int)

    matrix = block_diag(*[np.eye(nd) for _ in range(Nperiod)])

    # Roll the matrix by nd to the right
    matrix = np.roll(matrix, nd, axis=1)

    return matrix

def get_LDO_disturbance_matrices(Bd, Cd, Nperiod: int):
    # Get dimensions
    nd = Bd.shape[1]

    Ashift = create_circ_matrix(nd, Nperiod)
    Bpick = np.block([np.eye(nd), np.zeros((nd, (Nperiod-1)*nd))])

    Bd = Bd @ Bpick
    Cd = Cd @ Bpick

    return Ashift, Bd, Cd

def get_LDO_LTI(A, B, C, Bd, Cd, Nperiod: int):
    # Get dimensions
    nx = A.shape[0]
    nu = B.shape[1]
    nd = Bd.shape[1]

    # As ooposed to lift_LTI, we now have Nperiod copies of the disturbance vector
    # Ashift should be of size (Nperiod*nd, Nperiod*nd)
    # It should shift the disturbance vector by nd at each period
    Ashift = create_circ_matrix(nd, Nperiod)
    # Bpick should be of size (Nperiod*nd, nd)
    # It should pick out the first nd elements of the state vector
    Bpick = np.block([np.eye(nd), np.zeros((nd, (Nperiod-1)*nd))])

    # Augmented state space model
    Ae = np.block([[A, Bd @ Bpick], [np.zeros((Nperiod*nd, nx)), Ashift]])
    Be = np.block([[B], [np.zeros((Nperiod*nd, nu))]])
    Ce = np.block([[C, Cd @ Bpick]])

    return Ae, Be, Ce
