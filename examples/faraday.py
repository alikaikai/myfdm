from waveguidemesh import waveguidemesh
from modesolver import ModeSolver
from contourmode import contour
from stretchmesh import stretchmesh
from numpy import sqrt, dstack

# Calculate the(circularly polarized) modes of a gyrotropic
# ridge waveguide.This example incorporates complex off-diagonal elements in the permittivity tensor, in order
# to model the faraday effect in the presence of an applied DC magnetic field

wl = 1.485  # wavelength (um)

# Refractive indices:
dn = 1.85e-4    # YIG birefringence
n1 = 1.94       # lower cladding(GGG)
n2 = 2.18       # lower cladding(Bi: YIG)
n3 = 2.19       # core(Bi: YIG)
n4 = 1          # upper cladding(air)
delta = 2.4e-4  # Faraday rotation constant

# Vertical dimensions:
h1 = 0.5        # lower cladding
h2 = 3.1        # cladding
h3 = 3.9        # core

h4 = 0.5        # upper cladding
rh = 0.5        # Ridge height

# Horizontal dimensions
rw = 4.0        # Ridge half-width
side = 6.0      # Space on side

# Grid size
dx = 0.100      # grid size(horizontal)
dy = 0.050      # grid size(vertical)

nmodes = 2      # number of modes to compute

x, y, epsxx = waveguidemesh([n1, n2 - dn, n3 - dn, n4], [h1, h2, h3, h4], rh, rw, [side, side], dx, dy)
epszz = epsxx
x, y, epsyy = waveguidemesh([n1, n2, n3, n4], [h1, h2, h3, h4], rh, rw, [side, side], dx, dy)
x, y, epsxy = waveguidemesh([0, sqrt(delta), sqrt(delta), 0], [h1, h2, h3, h4], rh, rw, [side, side], dx, dy)
epsxy = 1j * epsxy
epsyx = -epsxy

# stretch out the mesh at the boundaries:
x, y = stretchmesh(x, y, [10, 10, 60, 60], [2, 2, 5, 5])

solver = ModeSolver(wl, n3, x, y, dstack([epsxx, epsxy, epsyx, epsyy, epszz]), '0000')

Hxs, Hys, neffs = solver.solve(2)

Hp1 = (Hxs[0] + 1j * Hys[0]) / sqrt(2)
Hm1 = (Hxs[0] - 1j * Hys[0]) / sqrt(2)

Hp2 = (Hxs[1] + 1j * Hys[1]) / sqrt(2)
Hm2 = (Hxs[1] - 1j * Hys[1]) / sqrt(2)
