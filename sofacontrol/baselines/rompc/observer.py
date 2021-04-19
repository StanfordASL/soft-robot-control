from sofacontrol.lqr.lqr import dare

class DiscreteLuenbergerObserver:
    """
    Updates the state estimate for a linear system with a constant observer gain. Assumes measuremnt
    model is linear:

    y = Cf*xf

    and that with a reduced order approximation xf = V*x + xf_ref this model becomes

    y = C*x + y_ref

    where C = Cf*V and y_ref = Cf*xf_ref.

    :dyn_sys: LinearROM dynamical system object
    :param Q, R: Q and R quadratic cost matrices for LQR observer gain
    """

    def __init__(self, dyn_sys, Q, R):
        self.dyn_sys = dyn_sys
        if self.dyn_sys.C is None:
            raise RuntimeError('Need to set meas. model in dyn_sys')
        self.C = self.dyn_sys.C

        # Compute gain
        L, _ = dare(self.dyn_sys.A_d.T, self.dyn_sys.C.T, Q, R)
        self.L = -L.T

    def initialize(self, xf):
        """
        Initialize the reduced order state estimate.
        """
        self.x = self.dyn_sys.rom.compute_RO_state(xf=xf)
        self.update_z()

    def update(self, u, y):
        y = self.dyn_sys.zfyf_to_zy(yf=y)  # convert full order measurement into reduced order measurement
        self.x = self.dyn_sys.update_state(self.x, u) + self.L @ (y - self.dyn_sys.C @ self.x)
        self.update_z()

    def update_z(self):
        if self.dyn_sys.H is not None:
            self.z = self.dyn_sys.x_to_zfyf(self.x, zf=True)
        else:
            self.z = self.dyn_sys.x_to_zfyf(self.x, yf=True)