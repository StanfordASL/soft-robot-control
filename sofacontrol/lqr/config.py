class iLQRConfig:
    def __init__(self):
        self.max_iter = 50
        self.epsilon = 0.1

        self.include_input_var_constraint = True

        self.do_linesearch = True
        self.regularize = True

        ######## Line search parameters (forward pass) ##########

        self.alpha0 = 1.  # start value for linesearch
        self.alpha_scaling = 0.5
        self.improv_lb = 1e-4
        self.improv_ub = 100
        self.alpha_min = 5e-2  # If smaller than this value forward pass is not completed, and rho is increased

        self.counter_limit = 5

        ####### Regularization parameters (backward pass) #########
        # Intuition: Increasing the regularization term makes the partial Hessian (Q_uu) more like identity matrix,
        # effectively steering Gauss-Newton step (unregularized iLQR) more like gradient descent (naive approach).
        # However gradient descent is generally more robust and reliable far from a local optimum
        self.rho0 = 0.  # This is generally difficult to tune
        self.drho0 = 0.
        self.rho_scaling = 1.5
        self.rho_increase_fp = 10.
        self.rho_max = 1e5
        self.rho_min = 1e-3
        self.state_regularization = True  # if False, does input regularization, generally less robust but better (Taylor)
