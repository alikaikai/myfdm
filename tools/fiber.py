import numpy as np


def fiber(n, r, side, dx, dy):
    """
    parameters:
        n - indices of refraction for layers in fiber
        r - outer radius of each layer
        side - width of cladding layer to simulate
        dx - horizontal grid spacing
        dy - vertical grid spacing
    returns:
        x, y - vectors specifying mesh coordinates
        eps - index mesh (n^2)
    """

    nx = round((sum(r) + side) / dx)
    ny = round((sum(r) + side) / dy)

    x = dx * np.arange(nx + 1)
    y = dy * np.arange(ny + 1)
    xc = (x[: -1] + x[1 :]) / 2
    yc = (y[: -1] + y[1:]) / 2

    xc = xc.reshape(-1, 1)
    yc = yc.reshape(1, -1)

    rho = np.sqrt( np.dot(xc**2, np.ones(yc.shape)) + np.dot(np.ones(xc.shape), yc**2) )

    eps = n[-1]**2 * np.ones((nx, ny))

    for i in range(len(n)-2, -1, -1):
        eps = np.where(rho < r[i], n[i]**2, eps)

    return x, y, eps
