from waveguidemesh import waveguidemeshalf
from modesolver import ModeSolver
from contourmode import contour

ns = [3.34, 3.44, 1.0]
hs = [2.0, 1.3, 0.5]
rh = 1.1
rw = 1.0
side = 1.5
dx = 0.0125
dy = 0.0125

wl = 1.55
guess = ns[1]
nmodes = 1

x, y, eps = waveguidemeshalf(ns, hs, rh, rw, side, dx, dy)

solver = ModeSolver(wl, guess, x, y, eps, '000A')

Hxs, Hys, neffs = solver.solve(nmodes)

contour(x, y, Hxs[0], Hys[0])

print(neffs)
