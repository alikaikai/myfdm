import numpy as np


def waveguidemesh(ns, hs, rh, rw, sides, dx, dy):
    """
    This function creates an index mesh for the finite-difference
    mode solver.  The function will accommodate a generalized three
    layer rib waveguide structure.  (Note: channel waveguides can
    also be treated by selecting the parameters appropriately.)

    INPUT

    ns - indices of refraction for layers in waveguide
    hs - height of each layer in waveguide
    rh - height of waveguide feature
    rw - half-width of waveguide
    sides - excess space to the left and right of waveguide
    dx - horizontal grid spacing
    dy - vertical grid spacing
 
    OUTPUT
 
    x,y - vectors specifying mesh coordinates
    eps - index mesh (n^2)
    """
    # number of dy of each h
    ihs = [round(h / dy) for h in hs]
    # number of dy of ridge height
    irh = round(rh / dy)
    # number of dx of ridge width
    irw = round(2*rw / dx)
    # number of dx of side
    isides = [round(side / dx) for side in sides]

    # number of dx
    nx = irw + sum(isides)
    # number of dy
    ny = sum(ihs)

    x = np.arange(-(isides[0] + irw/2), irw/2 + isides[1] + 1) * dx
    y = np.arange(ny + 1) * dy

    eps = np.zeros((nx, ny))

    hx = 0
    for ih, n in zip(ihs, ns):
        eps[:, hx: hx + ih] = n**2
        hx += ih

    eps[:isides[0], ny - ihs[-1] - irh: ny - ihs[-1]] = ns[-1]**2
    eps[irw + isides[0]:, ny - ihs[-1] - irh: ny - ihs[-1]] = ns[-1]**2

    return x, y, eps


def waveguidemeshalf(ns, hs, rh, rw, side, dx, dy):
    
    ihs = [round(h / dy) for h in hs]
    irh = round(rh / dy)
    irw = round(2 * rw / dx)
    iside = round(side / dx)

    nx = irw + iside
    ny = sum(ihs)

    x = np.arange(nx + 1) * dx
    y = np.arange(ny + 1) * dy

    eps = np.zeros((nx, ny))

    hx = 0
    for ih, n in zip(ihs, ns):
        eps[:, hx: hx + ih] = n ** 2
        hx += ih

    eps[irw:, ny - ihs[-1] - irh: ny - ihs[-1]] = ns[-1] ** 2

    return x, y, eps

