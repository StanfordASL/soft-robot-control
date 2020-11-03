class TemplateModel:
    """
    Template model object for use with GuSTO class. This object describes continuous time dynamics:

    xdot = f(x,u) = f0(x) + B(x)u
    z = Hx
    """

    def __init__(self):
        #### Objective function variables ####
        self.H = None  # Performance variable matrix (n_z x n_x)

        #### Dimensions of problem ####
        self.n_x = None  # State dimension
        self.n_u = None  # Input dimension
        self.n_z = None  # Number of performance variables

    def get_continuous_dynamics(self, x, u):
        """
        For dynamics xdot = f(x,u) = f0(x) + B(x)u returns:
        f = f(x,u): full dynamics
        A = df/dx(x,u): state Jacobian (note contains f0(x) and B(x) terms)
        B = B(x): control Jacobian
        """
        RuntimeError('Must be subclassed and implemented')
        f = None
        A = None
        B = None
        return f, A, B

    def get_discrete_dynamics(self, x, u, dt):
        """
        For dynamics xdot = f0(x) + B(x)u, the Taylor approximation can be written 

            xdot = f0(x0) + A(x0)(x-x0) + B(x0)u

        alternatively written as 

            xdot =  A(x0)x + B(x0)u + d(x0)

        with d(x0) = f0(x0) - A(x0)x0 = f(x0,u0) - A(x0)x0 - B(x0)u0. This function
        then returns a discrete time version of this equation

            x_k+1 =  Ad x_k + Bd u_k + dd

        :x: State x0 (n_x)
        :u: Input u0 (n_u)
        :dt: time step for discretization (seconds)
        """
        RuntimeError('Must be subclassed and implemented')
        dd = None
        Ad = None
        Bd = None
        return Ad, Bd, dd

    def get_characteristic_vals(self):
        """
        An optional function to define a procedure for computing characteristic values
        of the state and dynamics for use with GuSTO scaling, defaults to all ones
        """
        x_char = np.ones(self.n_x)
        f_char = np.ones(self.n_x)
        return x_char, f_char

    def rollout(self, x0, u, dt):
        """
        :x0: initial condition (n_x,)
        :u: array of control (N, n_u)
        :dt: time step

        Returns state x (N + 1, n_x) and performance variable z (N + 1, n_z)
        """
        N = u.shape[0]
        x = np.zeros((N + 1, self.n_x))

        # Set initial condition
        x[0, :] = x0

        # Simulate using some method

        # Compute output variables
        if self.H is not None:
            z = np.transpose(self.H @ x.T)
        else:
            z = None

        RuntimeError('Must be subclassed and implemented')
        return x, z
