import numpy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigen

class ModeSolver:

    """
    The ModeSolver class computes the electric and magnetic fields
    for modes of a dielectric waveguide using the "Vector Finite
    Difference (VFD)" method, as described in A. B. Fallahkhair,
    K. S. Li and T. E. Murphy, "Vector Finite Difference Modesolver
    for Anisotropic Dielectric Waveguides", J. Lightwave
    Technol. 26(11), 1423-1431, (2008).

    Parameters
    ----------
    wl : float
        The wavelength of the optical radiation (units are arbitrary,
        but must be self-consistent between all inputs. It is recommended to
        just use microns for everthing)
    x : 1D array of floats
        Array of horizontal coordinates
    y : 1D array of floats
        Array of vertical coordinates
    eps: 2D or 3D array of floats
        Relative permittivity numpy.array of either
        shape( x.shape[0], y.shape[0] ) where each element of the
        array can either be a single float, corresponding the an
        isotropic refractive index, or (x.shape[0], y.shape[0], 5),
        where the last dimension describes the relative permittivity in
        the form (epsxx, epsxy, epsyx, epsyy, epszz).
    boundary : str
        This is a string that identifies the type of boundary
        conditions applied.
        The following options are available:
           'A' - Hx is antisymmetric, Hy is symmetric.
           'S' - Hx is symmetric and, Hy is antisymmetric.
           '0' - Hx and Hy are zero immediately outside of the boundary.
        The string identifies all four boundary conditions, in the
        order: North, south, east, west.  For example, boundary='000A'
    """

    def __init__(self, wl, guess, x, y, eps, boundary):
        self.wl = wl
        self.guess = guess
        self.x = x
        self.y = y
        self.eps = eps
        self.boundary = boundary

    def _get_eps(self):
        eps = self.eps

        def _reshape(eps):
            eps = numpy.c_[eps[:, 0:1], eps, eps[:, -1:]]
            eps = numpy.r_[eps[0:1, :], eps, eps[-1:, :]]
            return eps

        if eps.ndim == 2: # isotropic refractive index
            eps = _reshape(eps)
            epsxx = epsyy = epszz = eps
            epsxy = epsyx = numpy.zeros_like(epsxx)

        elif eps.ndim == 3: # anisotropic refractive index
            epsxx = _reshape(eps[:, :, 0])
            epsxy = _reshape(eps[:, :, 1])
            epsyx = _reshape(eps[:, :, 2])
            epsyy = _reshape(eps[:, :, 3])
            epszz = _reshape(eps[:, :, 4])

        return epsxx, epsxy, epsyx, epsyy, epszz

    def _build_matrix(self):

        dx = numpy.diff(self.x)
        dy = numpy.diff(self.y)

        dx = numpy.r_[dx[0], dx, dx[-1]].reshape(-1, 1)
        dy = numpy.r_[dy[0], dy, dy[-1]].reshape(1, -1)

        epsxx, epsxy, epsyx, epsyy, epszz = self._get_eps()

        nx = len(self.x)
        ny = len(self.y)

        k = 2 * numpy.pi / self.wl

        ones_nx = numpy.ones((nx, 1))
        ones_ny = numpy.ones((1, ny))

        # distance of mesh points to nearest neighbor mesh point:
        n = numpy.dot(ones_nx, dy[:, 1:]).flatten()
        s = numpy.dot(ones_nx, dy[:, :-1]).flatten()
        e = numpy.dot(dx[1:, :], ones_ny).flatten()
        w = numpy.dot(dx[:-1, :], ones_ny).flatten()

        # These define the permittivity (eps) tensor relative to each mesh point
        # using the following geometry:
        #
        #                 NW------N------NE
        #                 |       |       |
        #                 |   1   n   4   |
        #                 |       |       |
        #                 W---w---P---e---E
        #                 |       |       |
        #                 |   2   s   3   |
        #                 |       |       |
        #                 SW------S------SE

        exx1 = epsxx[:-1, 1:].flatten()
        exx2 = epsxx[:-1, :-1].flatten()
        exx3 = epsxx[1:, :-1].flatten()
        exx4 = epsxx[1:, 1:].flatten()
        
        eyy1 = epsyy[:-1, 1:].flatten()
        eyy2 = epsyy[:-1, :-1].flatten()
        eyy3 = epsyy[1:, :-1].flatten()
        eyy4 = epsyy[1:, 1:].flatten()
        
        exy1 = epsxy[:-1, 1:].flatten()
        exy2 = epsxy[:-1, :-1].flatten()
        exy3 = epsxy[1:, :-1].flatten()
        exy4 = epsxy[1:, 1:].flatten()
        
        eyx1 = epsyx[:-1, 1:].flatten()
        eyx2 = epsyx[:-1, :-1].flatten()
        eyx3 = epsyx[1:, :-1].flatten()
        eyx4 = epsyx[1:, 1:].flatten()
        
        ezz1 = epszz[:-1, 1:].flatten()
        ezz2 = epszz[:-1, :-1].flatten()
        ezz3 = epszz[1:, :-1].flatten()
        ezz4 = epszz[1:, 1:].flatten()

        ns21 = n * eyy2 + s * eyy1
        ns34 = n * eyy3 + s * eyy4
        ew14 = e * exx1 + w * exx4
        ew23 = e * exx2 + w * exx3

        # calculate the finite difference coefficients following
        # Fallahkhair and Murphy, Appendix Eqs 21 though 37

        axxn = ((2 * eyy4 * e - eyx4 * n) * (eyy3 / ezz4) / ns34 +
                (2 * eyy1 * w + eyx1 * n) * (eyy2 / ezz1) / ns21) / \
                (n * (e + w))
        axxs = ((2 * eyy3 * e + eyx3 * s) * (eyy4 / ezz3) / ns34 +
                (2 * eyy2 * w - eyx2 * s) * (eyy1 / ezz2) / ns21) / \
                (s * (e + w))
        ayye = (2 * n * exx4 - e * exy4) * exx1 / ezz4 / e / ew14 / \
            (n + s) + (2 * s * exx3 + e * exy3) * \
            exx2 / ezz3 / e / ew23 / (n + s)
        ayyw = (2 * exx1 * n + exy1 * w) * exx4 / ezz1 / w / ew14 / \
            (n + s) + (2 * exx2 * s - exy2 * w) * \
            exx3 / ezz2 / w / ew23 / (n + s)
        axxe = 2 / (e * (e + w)) + \
            (eyy4 * eyx3 / ezz3 - eyy3 * eyx4 / ezz4) / (e + w) / ns34
        axxw = 2 / (w * (e + w)) + \
            (eyy2 * eyx1 / ezz1 - eyy1 * eyx2 / ezz2) / (e + w) / ns21
        ayyn = 2 / (n * (n + s)) + \
            (exx4 * exy1 / ezz1 - exx1 * exy4 / ezz4) / (n + s) / ew14
        ayys = 2 / (s * (n + s)) + \
            (exx2 * exy3 / ezz3 - exx3 * exy2 / ezz2) / (n + s) / ew23

        axxne = +eyx4 * eyy3 / ezz4 / (e + w) / ns34
        axxse = -eyx3 * eyy4 / ezz3 / (e + w) / ns34
        axxnw = -eyx1 * eyy2 / ezz1 / (e + w) / ns21
        axxsw = +eyx2 * eyy1 / ezz2 / (e + w) / ns21

        ayyne = +exy4 * exx1 / ezz4 / (n + s) / ew14
        ayyse = -exy3 * exx2 / ezz3 / (n + s) / ew23
        ayynw = -exy1 * exx4 / ezz1 / (n + s) / ew14
        ayysw = +exy2 * exx3 / ezz2 / (n + s) / ew23

        axxp = -axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw + \
            k ** 2 * (n + s) * \
            (eyy4 * eyy3 * e / ns34 + eyy1 * eyy2 * w / ns21) / (e + w)
        ayyp = -ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw + \
            k ** 2 * (e + w) * \
            (exx1 * exx4 * n / ew14 + exx2 * exx3 * s / ew23) / (n + s)
        axyn = (eyy3 * eyy4 / ezz4 / ns34 - eyy2 * eyy1 / ezz1 /
                ns21 + s * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
        axys = (eyy1 * eyy2 / ezz2 / ns21 - eyy4 * eyy3 / ezz3 /
                ns34 + n * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
        ayxe = (exx1 * exx4 / ezz4 / ew14 - exx2 * exx3 / ezz3 /
                ew23 + w * (exx2 * exx4 - exx1 * exx3) / ew23 / ew14) / (n + s)
        ayxw = (exx3 * exx2 / ezz2 / ew23 - exx4 * exx1 / ezz1 /
                ew14 + e * (exx4 * exx2 - exx1 * exx3) / ew23 / ew14) / (n + s)

        axye = (eyy4 * (1 + eyy3 / ezz4) - eyy3 * (1 + eyy4 / ezz4)) / ns34 / (e + w) - \
               (2 * eyx1 * eyy2 / ezz1 * n * w / ns21 +
                2 * eyx2 * eyy1 / ezz2 * s * w / ns21 +
                2 * eyx4 * eyy3 / ezz4 * n * e / ns34 +
                2 * eyx3 * eyy4 / ezz3 * s * e / ns34 +
                2 * eyy1 * eyy2 * (1. / ezz1 - 1. / ezz2) * w ** 2 / ns21) / e / (e + w) ** 2

        axyw = (eyy2 * (1 + eyy1 / ezz2) - eyy1 * (1 + eyy2 / ezz2)) / ns21 / (e + w) - \
               (2 * eyx1 * eyy2 / ezz1 * n * e / ns21 +
                2 * eyx2 * eyy1 / ezz2 * s * e / ns21 +
                2 * eyx4 * eyy3 / ezz4 * n * w / ns34 +
                2 * eyx3 * eyy4 / ezz3 * s * w / ns34 +
                2 * eyy3 * eyy4 * (1. / ezz3 - 1. / ezz4) * e ** 2 / ns34) / w / (e + w) ** 2

        ayxn = (exx4 * (1 + exx1 / ezz4) - exx1 * (1 + exx4 / ezz4)) / ew14 / (n + s) - \
               (2 * exy3 * exx2 / ezz3 * e * s / ew23 +
                2 * exy2 * exx3 / ezz2 * w * n / ew23 +
                2 * exy4 * exx1 / ezz4 * e * s / ew14 +
                2 * exy1 * exx4 / ezz1 * w * n / ew14 +
                2 * exx3 * exx2 * (1. / ezz3 - 1. / ezz2) * s ** 2 / ew23) / n / (n + s) ** 2

        ayxs = (exx2 * (1 + exx3 / ezz2) - exx3 * (1 + exx2 / ezz2)) / ew23 / (n + s) - \
               (2 * exy3 * exx2 / ezz3 * e * n / ew23 +
                2 * exy2 * exx3 / ezz2 * w * n / ew23 +
                2 * exy4 * exx1 / ezz4 * e * s / ew14 +
                2 * exy1 * exx4 / ezz1 * w * s / ew14 +
                2 * exx1 * exx4 * (1. / ezz1 - 1. / ezz4) * n ** 2 / ew14) / s / (n + s) ** 2

        axyne = +eyy3 * (1 - eyy4 / ezz4) / (e + w) / ns34
        axyse = -eyy4 * (1 - eyy3 / ezz3) / (e + w) / ns34
        axynw = -eyy2 * (1 - eyy1 / ezz1) / (e + w) / ns21
        axysw = +eyy1 * (1 - eyy2 / ezz2) / (e + w) / ns21

        ayxne = +exx1 * (1 - exx4 / ezz4) / (n + s) / ew14
        ayxse = -exx2 * (1 - exx3 / ezz3) / (n + s) / ew23
        ayxnw = -exx4 * (1 - exx1 / ezz1) / (n + s) / ew14
        ayxsw = +exx3 * (1 - exx2 / ezz2) / (n + s) / ew23

        axyp = -(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw) - \
                k ** 2 * (w * (n * eyx1 * eyy2 + s * eyx2 * eyy1) / ns21 + \
                e * (s * eyx3 * eyy4 + n * eyx4 * eyy3) / ns34) / (e + w)
        ayxp = -(ayxn + ayxs + ayxe + ayxw + ayxne + ayxse + ayxnw + ayxsw) - \
                k ** 2 * (n * (w * exy1 * exx4 + e * exy4 * exx1) / ew14 + \
                s * (w * exy2 * exx3 + e * exy3 * exx2) / ew23) / (n + s)

        ii = numpy.arange(nx * ny).reshape((nx, ny))

        # NORTH boundary

        ib = ii[:, -1]

        if self.boundary[0] == 'S':
            sign = 1
        elif self.boundary[0] == 'A':
            sign = -1
        elif self.boundary[0] == '0':
            sign = 0

        axxs[ib]  += sign * axxn[ib]
        axxse[ib] += sign * axxne[ib]
        axxsw[ib] += sign * axxnw[ib]
        ayxs[ib]  += sign * ayxn[ib]
        ayxse[ib] += sign * ayxne[ib]
        ayxsw[ib] += sign * ayxnw[ib]
        ayys[ib]  -= sign * ayyn[ib]
        ayyse[ib] -= sign * ayyne[ib]
        ayysw[ib] -= sign * ayynw[ib]
        axys[ib]  -= sign * axyn[ib]
        axyse[ib] -= sign * axyne[ib]
        axysw[ib] -= sign * axynw[ib]

        # SOUTH boundary

        ib = ii[:, 0]

        if self.boundary[1] == 'S':
            sign = 1
        elif self.boundary[1] == 'A':
            sign = -1
        elif self.boundary[1] == '0':
            sign = 0

        axxn[ib]  += sign * axxs[ib]
        axxne[ib] += sign * axxse[ib]
        axxnw[ib] += sign * axxsw[ib]
        ayxn[ib]  += sign * ayxs[ib]
        ayxne[ib] += sign * ayxse[ib]
        ayxnw[ib] += sign * ayxsw[ib]
        ayyn[ib]  -= sign * ayys[ib]
        ayyne[ib] -= sign * ayyse[ib]
        ayynw[ib] -= sign * ayysw[ib]
        axyn[ib]  -= sign * axys[ib]
        axyne[ib] -= sign * axyse[ib]
        axynw[ib] -= sign * axysw[ib]

        # EAST boundary

        ib = ii[-1, :]

        if self.boundary[2] == 'S':
            sign = 1
        elif self.boundary[2] == 'A':
            sign = -1
        elif self.boundary[2] == '0':
            sign = 0

        axxw[ib]  += sign * axxe[ib]
        axxnw[ib] += sign * axxne[ib]
        axxsw[ib] += sign * axxse[ib]
        ayxw[ib]  += sign * ayxe[ib]
        ayxnw[ib] += sign * ayxne[ib]
        ayxsw[ib] += sign * ayxse[ib]
        ayyw[ib]  -= sign * ayye[ib]
        ayynw[ib] -= sign * ayyne[ib]
        ayysw[ib] -= sign * ayyse[ib]
        axyw[ib]  -= sign * axye[ib]
        axynw[ib] -= sign * axyne[ib]
        axysw[ib] -= sign * axyse[ib]

        # WEST boundary

        ib = ii[0, :]

        if self.boundary[3] == 'S':
            sign = 1
        elif self.boundary[3] == 'A':
            sign = -1
        elif self.boundary[3] == '0':
            sign = 0

        axxe[ib]  += sign * axxw[ib]
        axxne[ib] += sign * axxnw[ib]
        axxse[ib] += sign * axxsw[ib]
        ayxe[ib]  += sign * ayxw[ib]
        ayxne[ib] += sign * ayxnw[ib]
        ayxse[ib] += sign * ayxsw[ib]
        ayye[ib]  -= sign * ayyw[ib]
        ayyne[ib] -= sign * ayynw[ib]
        ayyse[ib] -= sign * ayysw[ib]
        axye[ib]  -= sign * axyw[ib]
        axyne[ib] -= sign * axynw[ib]
        axyse[ib] -= sign * axysw[ib]

        # Assemble sparse matrix

        iall = ii.flatten()
        i_s = ii[:, :-1].flatten()
        i_n = ii[:, 1:].flatten()
        i_e = ii[1:, :].flatten()
        i_w = ii[:-1, :].flatten()
        i_ne = ii[1:, 1:].flatten()
        i_se = ii[1:, :-1].flatten()
        i_sw = ii[:-1, :-1].flatten()
        i_nw = ii[:-1, 1:].flatten()

        Ixx = numpy.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
        Jxx = numpy.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
        Vxx = numpy.r_[axxp[iall], axxe[i_w], axxw[i_e], axxn[i_s], axxs[
            i_n], axxsw[i_ne], axxnw[i_se], axxne[i_sw], axxse[i_nw]]

        Ixy = numpy.r_[iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw]
        Jxy = numpy.r_[
            iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
        Vxy = numpy.r_[axyp[iall], axye[i_w], axyw[i_e], axyn[i_s], axys[
            i_n], axysw[i_ne], axynw[i_se], axyne[i_sw], axyse[i_nw]]

        Iyx = numpy.r_[
            iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw] + nx * ny
        Jyx = numpy.r_[iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se]
        Vyx = numpy.r_[ayxp[iall], ayxe[i_w], ayxw[i_e], ayxn[i_s], ayxs[
            i_n], ayxsw[i_ne], ayxnw[i_se], ayxne[i_sw], ayxse[i_nw]]

        Iyy = numpy.r_[
            iall, i_w, i_e, i_s, i_n, i_ne, i_se, i_sw, i_nw] + nx * ny
        Jyy = numpy.r_[
            iall, i_e, i_w, i_n, i_s, i_sw, i_nw, i_ne, i_se] + nx * ny
        Vyy = numpy.r_[ayyp[iall], ayye[i_w], ayyw[i_e], ayyn[i_s], ayys[
            i_n], ayysw[i_ne], ayynw[i_se], ayyne[i_sw], ayyse[i_nw]]

        I = numpy.r_[Ixx, Ixy, Iyx, Iyy]
        J = numpy.r_[Jxx, Jxy, Jyx, Jyy]
        V = numpy.r_[Vxx, Vxy, Vyx, Vyy]
        A = coo_matrix((V, (I, J))).tocsr()

        return A
        
    def solve(self, nmodes):

        A = self._build_matrix()

        k = 2 * numpy.pi / self.wl
        shift = (self.guess * k) ** 2
        [eigvals, eigvecs] = eigen.eigs(A, sigma=shift, k=nmodes,
                                        tol=1e-8, return_eigenvectors=True)

        neffs = self.wl * numpy.sqrt(eigvals) / (2 * numpy.pi)

        nx = len(self.x)
        ny = len(self.y)

        phixs = []
        phiys = []

        for imode in range(nmodes):
            temp = numpy.reshape(eigvecs[:, imode], (nx*ny, 2), order='F')
            mag_temp = numpy.sqrt(numpy.sum(abs(temp)**2, axis=1))
            ii = numpy.argmax(mag_temp, axis=0)
            mag = numpy.amax(mag_temp, axis=0)

            if abs(temp[ii, 0]) > abs(temp[ii, 1]):
                jj = 0
            else:
                jj = 1

            mag = mag * temp[ii, jj] / abs(temp[ii, jj])
            temp = temp / mag
            phixs.append(temp[:, 0].reshape(nx, ny))
            phiys.append(temp[:, 1].reshape(nx, ny))

        return phixs, phiys, neffs
