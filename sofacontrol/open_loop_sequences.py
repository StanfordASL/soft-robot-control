import random

import numpy as np
import pyDOE
from scipy.interpolate import interp1d


class BaseRobotSequences(object):
    """
    Basic control sequences for open loop control
    :param m: number of actuators
    :param u0: default "resting" control np array
    :param umax: max control of each actuator, with respect to u0
    :param umin: min control of each actuator, with respect to u0
    :param dt: time step of simulation
    :parma t0: start time for control sequences, from 0 to t0 a base control self.u0 is computed
    """

    def __init__(self, m, u0=None, umax=None, umin=None, dt=0.01, t0=0):
        self.m = m  # input dof
        self.dt = dt  # seconds

        # Base input level
        if u0 is None:
            u0 = np.zeros(self.m)
        self.u0 = u0

        # Bounds on control
        self.umax = umax
        self.umin = umin

        # Initial control sequence based on start time
        self.t0 = t0  # seconds
        self.u_base, self.save_base, _ = self.constant_input(self.u0, self.t0, add_base=False)
        self.save_base[-5:] = True  # Allows to save last points before sequence starts

        # Other params
        self.active_inputs = None
        self.name = None

    def constant_input(self, u_constant, t, add_base=True, save_data=False):
        """
        Simple constant input sequence, by default saves no snapshots
        :param u_constant: np array of the constant step input value, defaults
        :param t: duration of the sequence (seconds)
        :param save_data: save sequence. Helpful when applying zero inputs
        """
        self.name = 'constant'
        num_steps = int(t / self.dt)
        if u_constant.shape[0] != self.m or u_constant.ndim != 1:
            raise AssertionError('Dimension mismatch for control input')
        u_sequence = np.broadcast_to(np.expand_dims(u_constant, axis=-1), (self.m, num_steps))
        save_sequence = np.array([save_data] * num_steps)

        # Combine with base sequence
        if add_base:
            u_sequence, save_sequence = self.combined_sequence([self.u_base, u_sequence],
                                                               [self.save_base, save_sequence])

        t_sequence = self.dt * np.arange(u_sequence.shape[1])
        return u_sequence, save_sequence, t_sequence

    def augment_input_with_base(self, u_seq, save_data=True):
        num_steps = u_seq.shape[1]
        save_sequence = np.array([save_data] * num_steps)

        u_sequence, save_sequence = self.combined_sequence([self.u_base, u_seq],
                                                               [self.save_base, save_sequence])

        t_sequence = self.dt * np.arange(u_sequence.shape[1])
        return u_sequence, save_sequence, t_sequence

    def sine_input(self, u_max, t, add_base=True):
        """
        Simple sine wave sequence, by default saves no snapshots
        :param u_max: np array of the sine wave with u_max amplitude
        :param t: duration of the sequence (seconds)
        """
        self.name = 'sine'
        num_steps = int(t / self.dt)
        u_sequence = np.broadcast_to(np.expand_dims(u_max, axis=-1), (self.m, num_steps))
        sine = np.broadcast_to(np.sin(np.linspace(0, np.pi, num_steps)), (self.m, num_steps))
        u_sequence = u_sequence*sine
        save_sequence = np.array([False] * num_steps)

        # Combine with base sequence
        if add_base:
            u_sequence, save_sequence = self.combined_sequence([self.u_base, u_sequence],
                                                               [self.save_base, save_sequence])

        t_sequence = self.dt * np.arange(u_sequence.shape[1])
        return u_sequence, save_sequence, t_sequence

    def individual_actuation(self, t_step=None, interp_pts=0, add_base=True, static=False):
        """
        Creates a sequence of inputs by actuating cables individually (one by one), interpolating
        to add additional points, and then building a sequence of step inputs

        :param t_step: duration of time in between each sample (including interpolation points)
        :param interp_pts: number of interpolation points between u0 --> umax --> umin --> u0
        :param add_base: True to add the base sequence of self.u0 applied until time self.t0
        :param static: True to save only one point at the end of each step point, False (default) to save each point

        Note, for step inputs choose t_step > self.dt. If t_step is not set it defaults to self.dt such that
        the control is not a step input (only use if interp_pts is chosen to be pretty large). This
        leads to a smoother control in between individual actuation samples.

        Requires umin to be 0 or negative. This is achieved by setting u0 such that umin = 0 or u0 = (umax + umin) / 2
        """
        self.name = 'individual'
        u0 = self.u0[self.active_inputs]
        if t_step is None:
            t_step = self.dt

        seq = []
        # Each cable actuated independently
        for i in range(sum(self.active_inputs)):
            cable_i_max = u0.copy()
            cable_i_max[i] += self.umax[self.active_inputs][i]  # maximum actuation
            seq.append(cable_i_max)
            if self.umin[self.active_inputs][i] != 0:  # minimum only considered if different from u0
                cable_i_min = u0.copy()
                cable_i_min[i] += self.umin[self.active_inputs][i]  # minimum actuation
                seq.append(cable_i_min)
            seq.append(u0)  # return to u0

        seq = np.asarray(seq)
        seq, save_sequence = self.interpolate_and_repeat_step_sequence(seq, u0, interp_pts=interp_pts,
                                                                       steps_per_seq=int(t_step / self.dt),
                                                                       static=static)

        u_sequence = np.repeat(self.u0.reshape(-1, 1), seq.shape[0], axis=1)
        u_sequence[self.active_inputs, :] = seq.T  # transpose to match open_loop_controller expectation

        # Combine with base sequence (step input of u0)
        if add_base:
            u_sequence, save_sequence = self.combined_sequence([self.u_base, u_sequence],
                                                               [self.save_base, save_sequence])
        t_sequence = self.dt * np.arange(u_sequence.shape[1])
        return u_sequence, save_sequence, t_sequence

    def lhs_sequence(self, nbr_samples=30, t_step=None, interp_pts=0, nbr_zeros=0, add_base=True, static=False,
                     seed=None):
        """
        Creates a sequence of inputs by performing Latin Hypercube Sampling, interpolating to add
        additional points, and then building a sequence of step inputs.

        :param nbr_samples: number of samples for sampling Latin Hypercube
        :param t_step: duration of time in between each sample (including interpolation points)
        :param interp_pts: number of interpolation points to add between Latin Hypercube samples
        :param nbr_zeros: integer number of zero samples to insert into the Latin Hypercube samples (at random point), default=0
        :param add_base: True to add the base sequence of self.u0 applied until time self.t0
        :param static: True to save only one point at the end of each step point, False (default) to save each point
        :param seed: Set to int to specify seed, allowing user to obtain repeatable outputs

        Note, for step inputs choose t_step > self.dt. If t_step is not set it defaults to self.dt such that
        the control is not a step input (only use if interp_pts is chosen to be pretty large). This
        leads to a smoother control in between Latin Hypercube samples.
        """
        self.name = 'lhs'
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        u0 = self.u0[self.active_inputs]
        if t_step is None:
            t_step = self.dt

        # Sample controls based on Latin Hypercube Sampling method with random ordering
        nb_inputs_active = sum(self.active_inputs)
        seq = u0 + self.umin[self.active_inputs] + (self.umax[self.active_inputs] - self.umin[self.active_inputs]) * \
              pyDOE.lhs(nb_inputs_active, samples=nbr_samples, criterion="m")

        # Optionally add in samples at the origin
        if nbr_zeros > 0:
            seq = np.insert(seq, random.sample(range(nbr_samples), nbr_zeros), u0, axis=0)

        # Compute number of timesteps and then extend the sequence to step inputs with that number of timesteps each
        seq, save_sequence = self.interpolate_and_repeat_step_sequence(seq, u0, interp_pts=interp_pts,
                                                                       steps_per_seq=int(t_step / self.dt),
                                                                       static=static)

        # Now rebuild the full sequence to include zeros for inactive inputs
        u_sequence = np.repeat(self.u0.reshape(-1, 1), seq.shape[0], axis=1)
        u_sequence[self.active_inputs, :] = seq.T  # transpose to match open_loop_controller expectation

        # Combine with base sequence (step input of u0)
        if add_base:
            u_sequence, save_sequence = self.combined_sequence([self.u_base, u_sequence],
                                                               [self.save_base, save_sequence])

        t_sequence = self.dt * np.arange(u_sequence.shape[1])
        return u_sequence, save_sequence, t_sequence

    @staticmethod
    def interpolate_and_repeat_step_sequence(seq, u0, interp_pts=0, steps_per_seq=1, static=False):
        """
        Given a sequence, performs interpolation between points in sequence and repeats to provide a staircase input.
        Saves points
        :param seq:
        :param u0:
        :param interp_pts:
        :param steps_per_seq:
        :param static:
        :return:
        """
        # Add a sample of all zeros at the beginning of the sequence
        seq = np.vstack((u0, seq))

        # Now interpolate the get extra points in between each of the samples
        actual_nbr_samples = seq.shape[0]
        seq = interp1d(np.arange(actual_nbr_samples), seq, axis=0)(
            np.linspace(0, actual_nbr_samples - 1, (interp_pts + 1) * (actual_nbr_samples - 1) + 1))

        seq = np.concatenate((seq[0].reshape(1, -1), np.repeat(seq[1:], steps_per_seq, axis=0)), axis=0)

        if static:
            save_seq = np.array([False] * seq.shape[0])
            save_seq[::steps_per_seq] = True
        else:
            save_seq = np.array([True] * seq.shape[0])
        return seq, save_seq

    def combined_sequence(self, u_sequences, save_sequences, t_sequences=None):
        """
        :param u_sequences: List or tuple of u_sequence arrays
        :param save_sequences: List or tuple of save_sequence arrays/lists
        :param t_sequences: List or tuple of t_sequence arrays
        :return: Concatenated u_sequences and save_sequences (and t_sequences)
        """
        u_sequence = np.concatenate(u_sequences, axis=1)
        save_sequence = np.concatenate(save_sequences, axis=0)
        if t_sequences is None:
            return u_sequence, save_sequence
        else:
            for i in range(1, len(t_sequences)):
                t_sequences[i] += -t_sequences[i][0] + t_sequences[i - 1][-1] + self.dt
            t_sequence = np.concatenate(t_sequences, axis=0)
            return u_sequence, save_sequence, t_sequence

    def traj_tracking(self, generation_method):
        raise NotImplementedError('Must be subclassed')


class FingerRobotSequences(BaseRobotSequences):
    def __init__(self, dt=0.01, t0=0.):
        # Definition of the robot
        m = 1
        u0 = np.array([0.])
        umax = 2000 * np.ones(m)
        umin = np.zeros(m)

        # Initialize the object
        super(FingerRobotSequences, self).__init__(m, u0=u0, umax=umax, umin=umin, dt=dt, t0=t0)
        self.active_inputs = [True] * self.m

    def traj_tracking(self, generation_method='periodic_input', add_base=False, **kwargs):
        if generation_method == 'periodic_input':
            input_mean = kwargs.get('input_mean', 10. * 100)
            amplitude = kwargs.get('amplitude', 10. * 100)
            period = kwargs.get('period', 5)
            repetitions = kwargs.get('repetitions', 1)

            sine_wave = input_mean + amplitude * np.sin(np.linspace(0, 2 * repetitions * np.pi,
                                                                    int(period / self.dt * repetitions)))
            u_sequence = sine_wave.reshape(1, -1)

        else:
            raise NotImplementedError('This generation_method of traj tracking is not implemented')

        save_sequence = np.array([True] * u_sequence.shape[1])
        if add_base:
            u_sequence, save_sequence = self.combined_sequence([self.u_base, u_sequence],
                                                               [self.save_base, save_sequence])
        t_sequence = self.dt * np.arange(u_sequence.shape[1])
        return u_sequence, save_sequence, t_sequence


class TrunkRobotSequences(BaseRobotSequences):
    def __init__(self, dt=0.01, t0=0., umax=800):
        # Definition of the robot
        m = 8
        u0 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) * 100
        umax = np.ones(m) * umax
        umin = 0 * np.ones(m) * 100

        # Initialize the object
        super(TrunkRobotSequences, self).__init__(m, u0=u0, umax=umax, umin=umin, dt=dt, t0=t0)
        self.active_inputs = [True] * self.m

    def traj_tracking(self, generation_method='infinity_sign', add_base=False, **kwargs):
        self.name = 'traj_tracking'
        if generation_method == 'infinity_sign':
            amplitude = kwargs.get('amplitude', 5. * 100)
            period = kwargs.get('period', 2.5)
            repetitions = kwargs.get('repetitions', 2)

            high_freq_sine_wave = amplitude * np.sin(
                np.linspace(0., 2 * repetitions * np.pi, int(period * repetitions / self.dt)))
            low_freq_sine_wave = amplitude * np.sin(
                np.linspace(0., repetitions * np.pi, int(period * repetitions / self.dt)))

            infinity_input = np.zeros((int(period * repetitions / self.dt), 8))
            infinity_input[:, 0 + 4] = np.maximum(0, -high_freq_sine_wave)
            infinity_input[:, 2 + 4] = np.maximum(0, high_freq_sine_wave)
            infinity_input[:, 1 + 4] = np.maximum(0, low_freq_sine_wave)
            infinity_input[:, 3 + 4] = np.maximum(0, -low_freq_sine_wave)
            infinity_input[:, 0] = np.maximum(0, -high_freq_sine_wave / 2.)
            infinity_input[:, 2] = high_freq_sine_wave / 2.
            infinity_input[:, 1] = low_freq_sine_wave / 2.
            infinity_input[:, 3] = -low_freq_sine_wave / 2.
            u_sequence = infinity_input.T
            u_sequence += self.u0.reshape(-1, 1)

        else:
            raise NotImplementedError('This generation_method of traj tracking is not implemented')

        save_sequence = np.array([True] * u_sequence.shape[1])
        if add_base:
            u_sequence, save_sequence = self.combined_sequence([self.u_base, u_sequence],
                                                               [self.save_base, save_sequence])

        t_sequence = self.dt * np.arange(u_sequence.shape[1])

        return u_sequence, save_sequence, t_sequence


class TrunkRobotLongCablesOnlySequences(BaseRobotSequences):
    def __init__(self, dt=0.01, t0=0.):
        # Definition of the robot
        m = 4
        u0 = np.array([0, 0, 0, 0]) * 100
        umax = 8 * np.ones(m) * 100
        umin = 0 * np.ones(m) * 100
        super(TrunkRobotLongCablesOnlySequences, self).__init__(m, u0=u0, umax=umax, umin=umin, dt=dt, t0=t0)
        self.active_inputs = [True] * self.m

    def traj_tracking(self, generation_method=None):
        raise NotImplementedError('Not implemented')


class DiamondRobotSequences(BaseRobotSequences):
    def __init__(self, umin=None, umax=None, dt=0.01, t0=0.):
        # Definition of the robot
        m = 4
        u0 = np.array([0., 0., 0., 0.])
        if umax is None:
            umax = np.array([1500., 1500., 1500., 1500.])
        if umin is None:
            umin = np.array([0., 0., 0., 0.])

        # Initialize the object
        super(DiamondRobotSequences, self).__init__(m, u0=u0, umax=umax, umin=umin, dt=dt, t0=t0)
        self.active_inputs = [True] * m

    def traj_tracking(self, generation_method, add_base=False, **kwargs):
        if generation_method == 'periodic_input':
            input_mean = kwargs.get('input_mean', 0)
            amplitude = kwargs.get('amplitude', 1500.)
            period = kwargs.get('period', 5)
            repetitions = kwargs.get('repetitions', 1)

            sine_wave = input_mean + amplitude * np.sin(np.linspace(0, 2 * repetitions * np.pi,
                                                                    int(period / self.dt * repetitions)))
            sine_input = np.zeros((int(period * repetitions / self.dt), self.m))
            sine_input[:, 0] = np.maximum(0, sine_wave)
            sine_input[:, 1] = np.maximum(0, sine_wave)
            sine_input[:, 2] = -np.minimum(0, sine_wave)
            sine_input[:, 3] = -np.minimum(0, sine_wave)
            u_sequence = sine_input.T
            u_sequence += self.u0.reshape(-1, 1)

        else:
            raise NotImplementedError('This generation_method of traj tracking is not implemented')

        save_sequence = np.array([True] * u_sequence.shape[1])
        if add_base:
            u_sequence, save_sequence = self.combined_sequence([self.u_base, u_sequence],
                                                               [self.save_base, save_sequence])
        t_sequence = self.dt * np.arange(u_sequence.shape[1])

        return u_sequence, save_sequence, t_sequence