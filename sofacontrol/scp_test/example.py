import numpy as np
from matplotlib import pyplot as plt

from sofacontrol.scp.gusto import GuSTO
from sofacontrol.scp.models.dubins_car import DubinsCar
from sofacontrol.utils import HyperRectangle

model = DubinsCar()
umax = np.array([1, 1])
umin = np.array([0, -1])
U = HyperRectangle(umax, umin)
xmax = np.array([1, 1, np.pi])
X = HyperRectangle(xmax, -xmax)
x_target = np.array([5, 5, 0])
Xf = HyperRectangle(x_target + 2, x_target - 2)

dumax = np.array([0.1, 0.1])
dumin = np.array([-0.1, -0.1])
dU = HyperRectangle(dumax, dumin)

# Optimization
N = 50
dt = 0.1
Qz = np.zeros((model.n_z, model.n_z))
R = np.eye(model.n_u)
Qzf = 100 * np.eye(model.n_z)
zf_des = np.array([4., 5., 0.])
x0 = np.zeros(3)
u_init = 0.0 * np.ones((N, model.n_u))
x_init = model.rollout(x0, u_init, dt)

x_char = np.array([1., 1., np.pi])

# Build and solve
gusto = GuSTO(model, N, dt, Qz, R, x0, u_init, x_init, u=u_init, zf=zf_des, Qzf=Qzf,
              U=None, dU=dU,
              verbose=1, visual=[], warm_start=False, x_char=x_char)
x, u, z, _ = gusto.get_solution()

# New solve
# zf_des = np.array([2., 2., 0.])
# gusto.solve(x0, u_init, x_init, zf=zf_des)
# x2, u2, z2 = gusto.get_solution()

# Plot
plt.figure()
plt.plot(x[:, 0], 'b')
plt.plot(x[:, 1], 'b')
# plt.plot(x2[:,0],'r')
# plt.plot(x2[:,1],'r')

plt.figure()
plt.plot(u[:, 0], 'b')
plt.plot(u[:, 1], 'b')
# plt.plot(u2[:,0],'r')
# plt.plot(u2[:,1],'r')
plt.show()
