import waveguidemesh as wm
from modesolver import ModeSolver
import stretchmesh as sm
from contourmode import contour

# Refractive indices:
ns = 3.44          # Substrate
n1 = 3.34          # Lower cladding
n2 = 3.44          # Core
n3 = 1.00          # Upper cladding (air)

# Vertical dimensions:
hs = 4.0           # Substrate
h1 = 1.1           # Lower cladding
h2 = 1.3           # Core thickness
h3 = 0.5           # Upper cladding
rh = 1.1           # Ridge height

# Horizontal dimensions:
rw = 1.0           # Ridge half-width
side = 1.5         # Space on side

# Grid size:
dx = 0.0125        # grid size (horizontal)
dy = 0.0125        # grid size (vertical)

wl = 1.55          # vacuum wavelength
nmodes = 1         # number of modes to compute
tol = 1e-8

guess = 3.388688

x, y, eps = wm.waveguidemeshalf([ns, n1, n2, n3], [hs, h1, h2, h3], rh, rw, side, dx, dy)

# Complex coordinate stretching:
xc, yc = sm.stretchmesh(x, y, [0, 160, 20, 0], 1 + 2j)

solver1 = ModeSolver(wl, guess, xc, yc, eps, '000A')
solver2 = ModeSolver(wl, guess, x, y, eps, '000A')

Hxs1, Hys1, neffs1 = solver1.solve(nmodes)
Hxs2, Hys2, neffs2 = solver2.solve(nmodes)

contour(x, y, Hys1[0].real, Hys2[0].real)

print(neffs1)
print(neffs2)
