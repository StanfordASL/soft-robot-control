from sofacontrol.tpwl import tpwl

class tpwl_config():
    def __init__(self):
        # Either 'dynamics' or 'distance'
        self.eval_type = None

        # Whether to save continuous or discrete time TPWL models (or both)
        self.save_continuous_TPWL = True
        self.save_discrete_TPWL = True

        # Options for eval_type = 'distance' or 'dynamics'
        self.TPWL_weighting_factors = {'q': None, 'v': None}
        self.TPWL_separate_calculation = None
        self.TPWL_threshold = None

        # Options for if using eval_type = 'dynamics'
        self.sim_sys = None
        self.constants_sim = {'dt': None, 'beta_weighting': None, 'element_weights': {'q': None, 'v': None}}
        self.TPWL_type = None
        self.discr_type = None
        self.fom_based = False # if the tpwl error is considered with respect to FOM or ROM
        self.output_based = False


class tpwl_distance_config(tpwl_config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.eval_type = 'distance'
        self.TPWL_separate_calculation = False  # If true, checks velocity and position separately (two lists), else combined
        self.TPWL_weighting_factors = {'q': 10.0, 'v': 1.0}
        self.TPWL_threshold = 1100.  #1100.  # 1300. #1300.#200. #1000. # 1 used for static simulation


class tpwl_dynamics_config(tpwl_config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.eval_type = 'dynamics'
        dt = 0.01  # Recommended to be same as SOFA

        # Weights to put on q and v when computing the distances from current point to all reference points
        dist_weights = {'q': 1., 'v': 0.}

        # Specify 'nn' for TPWL selecting nearest neighbor, or 'weighting' to use weighted mean of all points
        tpwl_method = 'nn'

        # Method for discetizing the dynamics, either 'fe', 'be', 'bil', 'zoh'
        discr_method = 'be'

        # Value that parameterizes the weight distribution if tpwl_method = 'weighting', a high
        # value gives more weight to the closest point (approaches nearest neighbor behavior)
        beta_weighting = None

        self.constants_sim = {'dt': dt, 'beta_weighting': beta_weighting, 'dist_weights': dist_weights, 
                                'tpwl_method': tpwl_method, 'discr_method': discr_method}
        self.sim_sys = tpwl.TPWLATV

        # Separately consider error for position and velocity when evaluating to save new point
        self.TPWL_separate_calculation = False

        # Weights for multiplying error metrics for q and v when combining
        self.TPWL_weighting_factors = {'q': 0.0, 'v': 1.0}

        # Threshold for when to add a new point based on weighted error metric
        self.TPWL_threshold = 100000
