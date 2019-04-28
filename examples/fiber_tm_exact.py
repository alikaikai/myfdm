import numpy as np
from fiber import fiber
from stretchmesh import stretchmesh
from scipy.special import jv, kv
from scipy.optimize import fsolve
from contourmode import contour

nco = 2.5
ncl = 1.5
r = 0.3
wl = 1
side = 0.2

dx = 0.002
dy = 0.002

x, y, eps = fiber([nco, ncl], [r], side, dx, dy)
x, y = stretchmesh(x, y, [96, 0, 96, 0], [4, 1, 4, 1])

V = 2 * np.pi * r / wl * np.sqrt(nco**2 - ncl**2)


def spam(U):
    return nco ** 2 * jv(1, U) / (U * jv(0, U)) + \
        ncl ** 2 * kv(1, np.sqrt(V ** 2 - U ** 2)) / \
        (np.sqrt(V ** 2 - U ** 2) * kv(0, np.sqrt(V ** 2 - U ** 2)))


U = fsolve(spam, 3.2).item()

W = np.sqrt(V**2 - U**2)
neff0 = np.sqrt(nco**2 - (U / (2 * np.pi * r / wl))**2)


x = x.reshape(-1, 1)
y = y.reshape(1, -1)
rho = np.sqrt(np.dot(x**2, np.ones(y.shape)) + np.dot(np.ones(x.shape), y**2))

sinphi = np.divide(np.dot(np.ones(x.shape), y), rho)
cosphi = np.divide(np.dot(x, np.ones(y.shape)), rho)

Hx0 = np.zeros(rho.shape)
Hy0 = np.zeros(rho.shape)

for index, value in np.ndenumerate(rho):
    if value == 0:
        Hx0[index] = 0
        Hy0[index] = 0
    elif value < r:
        Hx0[index] = -sinphi[index] * jv(1, U * value / r) / jv(1, U)
        Hy0[index] = cosphi[index] * jv(1, U * value / r) / jv(1, U)
    elif value > r:
        Hx0[index] = -sinphi[index] * kv(1, W * value / r) / kv(1, W)
        Hy0[index] = cosphi[index] * kv(1, W * value / r) / kv(1, W)

hxmax = np.amax(abs(Hx0))
hymax = np.amax(abs(Hy0))
Hx0 = Hx0 / max(hxmax, hymax)
Hy0 = Hy0 / max(hxmax, hymax)

x = x.flatten()
y = y.flatten()
