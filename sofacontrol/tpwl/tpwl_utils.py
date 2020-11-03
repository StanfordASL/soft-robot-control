import numpy as np
import sofacontrol.utils as scutils


class Target:
    """
    Instance of this class required as an input for iLQR and TrajTrackingLQR controller methods
    iLQR: Requires (z, Hf), for trajectory tracking also requires t
    TrajTrackingLQR: Requires (t, u). Performs rollout of command u, and stores it in self.x
    Allows for loading a _sim data file, either from open_loop_controller or closed_loop_controller
    (allows any other pkl data file too)
    """
    def __init__(self):
        self.t = None  # np.array of time in simulation
        self.u = None  # np.array of inputs (matching self.t timestep)
        self.z = None
        self.x = None
        self.Hf = None

    def load_target_file(self, file):
        data = scutils.load_data(file)
        self.t = data.get('t')
        self.u = data.get('u')
        self.z = data.get('z')
        self.Hf = data.get('Hf')


class DynamicsTarget(Target):
    def __init__(self):
        """
        Instance of this class required as an input for LQR and dLQR controller methods
        These require u, x, A, B to be set
        Requires manually setting A, B (from e.g. computation of ROM linearized system from _snapshots pkl file)
        """
        super(DynamicsTarget, self).__init__()
        self.A = None
        self.B = None
        self.x = None


class TPWLSnapshotData(scutils.SnapshotData):
    """
    An object that collects and stores points that defines a TPWL model. It includes functions
    that interface with Sofa to collect the data. Data is saved as a dictionary which includes:
    q, v, u, K, D, ... f: reduced order information about each saved point (lists)
    A_c, B_c, d_c, A_d, B_d, d_d: linear system matrices in continuous and discrete time
    rom_info: information on model reduction that can be used to rebuild an object
    info: other information such as config parameters and number of points in the model
    """
    def __init__(self, rom, config, info=dict(), Hf=None):
        super().__init__(save_dynamics=True)

        # Add additional data fields
        self.dict['A_c'] = []
        self.dict['B_c'] = []
        self.dict['d_c'] = []
        self.dict['A_d'] = []
        self.dict['B_d'] = []
        self.dict['d_d'] = []
        self.dict['z'] = []
        self.dict['z_est'] = []

        # Model order reduction object
        self.rom = rom

        # Save ROM object info so it can be recreated
        self.dict['rom_info'] = self.rom.get_info()

        # Configuration parameters
        self.config = config

        if self.config.eval_type == 'dynamics':
            self.sim_sys_class = self.config.sim_sys
            self.sim_sys_params = self.config.constants_sim

        # Info dictionary
        self.info = info

        # Other parameters
        self.save_step = 0
        self.saved_tpwl_steps = []
        self.Hf = Hf

    def add_point(self, point):
        if self.dict['dt'] == -1:
            self.dict['dt'] = point.dt

        self.saved_tpwl_steps.append(point.t)
        print('Time: {}, Number of points saved: {}'.format(point.t, len(self.saved_tpwl_steps)))
        # Add reduced order state and control
        self.dict['q'].append(self.rom.compute_RO_state(qf=point.q))
        self.dict['v'].append(self.rom.compute_RO_state(vf=point.v))
        self.dict['u'].append(point.u)

        # Add reduced order matrices
        self.dict['K'].append(self.rom.compute_RO_matrix(point.K))
        self.dict['D'].append(self.rom.compute_RO_matrix(point.D))
        self.dict['M'].append(self.rom.compute_RO_matrix(point.M))
        self.dict['b'].append(self.rom.compute_RO_matrix(point.b, left=True))
        self.dict['f'].append(self.rom.compute_RO_matrix(point.f, left=True))

        self.dict['H'].append(self.rom.compute_RO_matrix(point.H, left=True))
        self.dict['S'].append(self.rom.compute_RO_matrix(point.S))

        self.dict['q+'].append(self.rom.compute_RO_state(qf=point.q_next))
        self.dict['v+'].append(self.rom.compute_RO_state(vf=point.v_next))

        # Additionally add the new point to the TPWL model
        if self.config.save_continuous_TPWL:
            self.add_continuous_TPWL()

        if self.config.save_discrete_TPWL:
            self.add_discrete_TPWL()

        # If evaluating with dynamics model, regenerate model to include newest points
        if self.config.eval_type == 'dynamics':
            self.sim_sys = self.sim_sys_class(data=self.dict, params=self.sim_sys_params)


    def save_snapshot(self, point, prev_point):
        """
        Function called in open_loop_controller to determine if point should be saved
        """
        save = False
        if prev_point is not None:
            save = self.evaluate_point(point, prev_point)
        return save


    def simulation_end(self, filename):
        """
        Function called in open_loop_controller at the end of the simulation, allows the TPWL
        to save data
        """
        print('Computed TPWL, resulting in %d linearization points' % len(self.saved_tpwl_steps))

        # Add some additional info to the information dictionary
        self.info['state_dim'] = str(self.rom.rom_dim)
        self.info['nbr_lin'] = str(len(self.saved_tpwl_steps))
        self.info['saved_step_nbrs'] = self.saved_tpwl_steps
        self.info['tpwl_method'] = self.config.eval_type
        self.info['tpwl_parameters'] = vars(self.config)
        self.info['tpwl_type'] = self.config.TPWL_type
        self.info['discr_type'] = self.config.discr_type
        if self.config.eval_type == 'dynamics':
            del self.info['tpwl_parameters']['sim_sys']
        self.dict['info'] = self.info

        # Save the data
        print('Saving TPWL data to {}...'.format(filename))
        scutils.dict_lists_to_array(self.dict)
        scutils.save_data(filename, self.dict)
        print('Done.')


    def evaluate_point(self, point, prev_point):
        """
        Function called to evaluate whether a point should be added to the TPWL model.
        Current either uses distance or dynamics based metrics.
        """
        if not self.dict['q']:
            return True  # first point is always added

        if self.config.eval_type == 'distance':
            return self.evaluate_point_dist(point)

        elif self.config.eval_type == 'dynamics':
            return self.evaluate_point_dynamics(point, prev_point)


    def evaluate_point_dist(self, point):
        """
        Evaluate a point based on how far it is from the closest point in the
        set ofcurrent saved points
        """
        add_point = False

        # Compute distances between current reduced order q/v and model points
        q_dists = np.asarray(self.rom.compute_RO_state(qf=point.q) - np.asarray(self.dict['q']))
        v_dists = np.asarray(self.rom.compute_RO_state(vf=point.v) - np.asarray(self.dict['v']))

        # Weight the norms of the distances
        q_dists = self.config.TPWL_weighting_factors['q']*np.linalg.norm(q_dists, axis=1)
        v_dists = self.config.TPWL_weighting_factors['v']*np.linalg.norm(v_dists, axis=1)

        # Check if the weighted distances exceed the threshold
        if self.config.TPWL_separate_calculation:
            if np.min(q_dists) >= self.config.TPWL_threshold:
                add_point = True
            elif np.min(v_dists) >= self.config.TPWL_threshold:
                add_point = True
        else:
            if np.min(q_dists + v_dists) >= self.config.TPWL_threshold:
                add_point = True

        return add_point


    def evaluate_point_dynamics(self, point, prev_point):
        """
        Evaluate a point based on how different a point predicted by
        the current TPWL model is from the actual simulated next point
        """
        add_point = False

        if not (prev_point.u == np.zeros_like(prev_point.u)).all():
            # Get current state
            x = scutils.qv2x(point.q, point.v)

            # Get previous full/reduced order state
            x_prev = scutils.qv2x(prev_point.q, prev_point.v)
            x_prev_r = self.rom.compute_RO_state(xf=x_prev)

            # Predict current reduced order x from previous
            x_r_tpwl = self.sim_sys.update_state(x_prev_r, prev_point.u, prev_point.dt)

            if self.Hf is not None and self.config.output_based:
                # Compute the error in terms of the performance output
                zf_est = self.Hf @ self.rom.compute_FO_state(x=x_r_tpwl)
                zf = self.Hf @ x
                dz = zf_est - zf
                if np.linalg.norm(dz) >= self.config.TPWL_threshold:
                    add_point = True

                # Save some data to see how the model is doing
                self.dict['z_est'].append(zf_est)
                self.dict['z'].append(zf)

            else:
                if not self.config.fom_based:
                    # Project x to reduced order state
                    x_r = self.rom.compute_RO_state(xf=x)
                    delta_q, delta_v = scutils.x2qv(x_r - x_prev_r)
                    delta_q_est, delta_v_est = scutils.x2qv(x_r_tpwl - x_prev_r)

                else:
                    # Convert estimated state to full order
                    x_tpwl = self.rom.compute_FO_state(x=x_r_tpwl)
                    delta_q, delta_v = scutils.x2qv(x - x_prev)
                    delta_q_est, delta_v_est = scutils.x2qv(x_tpwl - x_prev)

                # Compute error metrics
                q_error = np.linalg.norm(delta_q_est - delta_q)
                v_error = np.linalg.norm(delta_v_est - delta_v)

                # Weight the errors
                q_error *= self.config.TPWL_weighting_factors['q']
                v_error *= self.config.TPWL_weighting_factors['v']

                # Check if the weighted error exceeds the threshold
                if self.config.TPWL_separate_calculation:
                    if q_error >= self.config.TPWL_threshold:
                        add_point = True
                    elif v_error >= self.config.TPWL_threshold:
                        add_point = True
                else:
                    if q_error + v_error >= self.config.TPWL_threshold:
                        add_point = True

        return add_point


    def add_continuous_TPWL(self):
        """
        Add a point to the continuous time TPWL model
        """
        A, B = scutils.extract_AB(self.dict['K'][-1], self.dict['D'][-1],
                          self.dict['M'][-1], self.dict['H'][-1])
        b_normalized = np.linalg.solve(self.dict['M'][-1], self.dict['f'][-1] + self.dict['K'][-1] @
                                       self.dict['q'][-1])

        d = np.hstack((b_normalized, np.zeros(np.shape(b_normalized))))

        self.dict['A_c'].append(A)
        self.dict['B_c'].append(B)
        self.dict['d_c'].append(d)


    def add_discrete_TPWL(self):
        """
        Add a point to the discrete time TPWL model
        """
        A_d, B_d = scutils.extract_AB_d(self.dict['S'][-1], self.dict['K'][-1], self.dict['H'][-1], self.dict['dt'])
        x = scutils.qv2x(self.dict['q'][-1], self.dict['v'][-1])
        x_next = scutils.qv2x(self.dict['q+'][-1], self.dict['v+'][-1])
        d_d = x_next - A_d @ x - B_d @ np.atleast_1d(self.dict['u'][-1])

        self.dict['A_d'].append(A_d)
        self.dict['B_d'].append(B_d)
        self.dict['d_d'].append(d_d)
