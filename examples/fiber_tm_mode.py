from fiber import fiber
from modesolver import ModeSolver
from contourmode import contour
from stretchmesh import stretchmesh

nco = 2.5
ncl = 1.5
r = 0.3

side = 0.2

dx = 0.002
dy = 0.002

wl = 1
nmodes = 1

boundary = '0A0S'

x, y, eps = fiber([nco, ncl], [r], side, dx, dy)

x, y = stretchmesh(x, y, [96, 0, 96, 0], [4, 1, 4, 1])

solver = ModeSolver(wl, nco, x, y, eps, boundary)

Hxs, Hys, neffs = solver.solve(nmodes)

contour(x, y, Hxs[0], Hys[0])

print(neffs)
