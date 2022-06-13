import numpy as np

import sofacontrol.utils as scutils
from sofacontrol.mor import pod


class LinearROM():
    """
    Linear ROM, written for compatibility with TPWL class
    """

    def __init__(self, data, dt, Cf=None, Hf=None):
        if not isinstance(data, dict):
            data = scutils.load_data(data)

        # Discretize dynamics via zero-order hold
        self.A_d, self.B_d, self.d_d = scutils.zoh_affine(data['A_c'], data['B_c'], data['d_c'], dt)

        # Build ROM object in case it is needed
        if data['rom_info']['type'] == 'POD':
            self.rom = pod.POD(data['rom_info'])
        else:
            raise NotImplementedError("Unknown ROM type")

        self.state_dim = self.A_d.shape[0]
        self.N = self.state_dim
        self.input_dim = self.B_d.shape[1]

        # Optionally set output and measurement models
        if Cf is not None:
            self.set_measurement_model(Cf)
        else:
            self.C = None
            self.y_ref = None
            self.meas_dim = None

        if Hf is not None:
            self.set_output_model(Hf)
        else:
            self.H = None
            self.z_ref = None
            self.output_dim = None

    def get_jacobians(self, x, dt):
        return self.A_d, self.B_d, self.d_d

    @staticmethod
    def update_dynamics(x, u, A_d, B_d, d_d):
        x_next = A_d @ x + np.squeeze(B_d @ u) + d_d
        return x_next

    def update_state(self, x, u):
        return self.A_d @ x + np.squeeze(self.B_d @ u) + self.d_d

    def set_measurement_model(self, Cf):
        self.C = Cf @ self.rom.V
        self.y_ref = Cf @ self.rom.x_ref
        self.meas_dim = self.C.shape[0]

    def set_output_model(self, Hf):
        self.H = Hf @ self.rom.V
        self.z_ref = Hf @ self.rom.x_ref
        self.output_dim = self.H.shape[0]

    def zfyf_to_zy(self, zf=None, yf=None):
        """
        :zf: (N, n_z) or (n_z,) array
        :yf: (N, n_y) or (n_y,) array
        """
        if zf is not None and self.z_ref is not None:
            return zf - self.z_ref
        elif yf is not None and self.y_ref is not None:
            return yf - self.y_ref
        else:
            raise RuntimeError('Need to set output or meas. model')

    def zy_to_zfyf(self, z=None, y=None):
        """
        :z: (N, n_z) or (n_z,) array
        :y: (N, n_y) or (n_y,) array
        """
        if z is not None and self.z_ref is not None:
            return z + self.z_ref
        elif y is not None and self.y_ref is not None:
            return y + self.y_ref
        else:
            raise RuntimeError('Need to set output or meas. model')

    def x_to_zfyf(self, x, zf=False, yf=False):
        """
        :x: (N, n_x) or (n_x,) array
        :zf: boolean
        :yf: boolean
        """
        if zf and self.H is not None:
            return np.transpose(self.H @ x.T) + self.z_ref
        elif yf and self.C is not None:
            return np.transpose(self.C @ x.T) + self.y_ref
        else:
            raise RuntimeError('Need to set output or meas. model')

    def x_to_zy(self, x, z=False, y=False):
        """
        :x: (N, n_x) or (n_x,) array
        :z: boolean
        :y: boolean
        """
        if z and self.H is not None:
            return np.transpose(self.H @ x.T)
        elif y and self.C is not None:
            return np.transpose(self.C @ y.T)
        else:
            raise RuntimeError('Need to set output or meas. model')

    def get_state_dim(self):
        return self.state_dim

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    def get_meas_dim(self):
        return self.meas_dim

    def get_rom_info(self):
        return self.tpwl_dict['rom_info']


def TPWL2LinearROM(tpwl_loc, save_loc):
    """
    Generates linearized ROM from existing TPWL ROM by taking the first linearization point.

    :param tpwl_loc: absolute path to location of tpwl model pickle file
    :param save_loc: absolutel path to location to save linear rom pickle file
    """

    tpwl_data = scutils.load_data(tpwl_loc)

    linrom_data = dict()
    linrom_data['A_c'] = tpwl_data['A_c'][0]
    linrom_data['B_c'] = tpwl_data['B_c'][0]
    linrom_data['d_c'] = tpwl_data['d_c'][0]
    linrom_data['rom_info'] = tpwl_data['rom_info']

    scutils.save_data(save_loc, linrom_data)
