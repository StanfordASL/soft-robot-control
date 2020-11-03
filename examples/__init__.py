class Problem:
    """
    Standard problem definition
    Required: self.Robot: Type of Robot: Trunk, TrunkLongCablesOnly, Finger, Diamond (more to come)
    Required: self.ControllerClass: OpenLoopController or ClosedLoopController
    Required: self.controller: Specifies how to compute input at each step. OpenLoop or (input, save) sequence for
    OpenLoopController and one of controllers.py classes for closed_loop_controller
    Optional: self.output_model: z = H x. z is saved in sim_data if sim_data not None. Req for ClosedLoopController,
    req when saving simulation data for OpenLoopController
    Optional: self.measurement_model: y = C x. y, the measurement at current time, is used in ClosedLoopController
    Optional: self.simdata_dir: Specify directory where simdata should be saved. Required to save sim data
    Optional: self.snapshots_dir: Specify directory where snapshots should be saved. Required to save snapshot data
    Optional: self.snapshots: What data to save (Snapshot or TPWLSnapshot). If not set will not save snapshot data.
    No impact if specified for ClosedLoopController.
    Optional: opt['sim_duration']: Time duration of simulation. If None, will continue indefinitely for infinite horizon
    problems.
    Optional: opt['save_prefix']: file is called save_prefix_sim.pkl and save_prefix_snapshots.pkl, otherwise datetime
    """

    def __init__(self):
        self.Robot = None
        self.ControllerClass = None

        self.controller = None
        self.measurement_model = None
        self.output_model = None

        # Directories for saving data (specify directory to activate saving this data type: snapshots or sim_data)
        self.snapshots_dir = None
        self.simdata_dir = None

        # Default don't save anything
        self.snapshots = None

        # Can also have other options, default None though
        self.opt = {'save_prefix': None, 'sim_duration': None}

    def checkDefinition(self):
        """ Check for required definitions """
        if self.ControllerClass is None:
            raise RuntimeError('ControllerClass must be defined in problem')
        elif self.Robot is None:
            raise RuntimeError('Robot must be defined in problem')
        elif self.controller is None:
            raise RuntimeError('controller must be defined in problem')
        elif self.measurement_model is None and self.ControllerClass.__name__ == 'ClosedLoopController':
            raise RuntimeError('measurement_model must be defined in problem')

        elif self.snapshots_dir is not None and self.snapshots is None:
            raise RuntimeError('snapshots must be defined in problem')

        elif self.output_model is None:
            if (self.ControllerClass.__name__ == 'ClosedLoopController') \
                    or (self.ControllerClass.__name__ == 'OpenLoopController'
                        and self.simdata_dir is not None):
                raise RuntimeError('output_model must be defined in problem')
