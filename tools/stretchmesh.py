import numpy
import scipy.optimize


def stretchmesh(x, y, nlayers, factor, method='PPPP'):
    """
    This function can be used to continuously stretch the grid
    spacing at the edges of the computation window for
    finite-difference calculations. The program implements four 
    different expansion methods: uniform, linear, parabolic (the default) 
    and geometric. The first three methods also allow for complex
    coordinate stretching, which is useful for creating
    perfectly-matched non-reflective boundaries.

    INPUT:

    x,y - vectors that specify the vertices of the original
      grid, which are usually linearly spaced.
    
    nlayers - vector that specifies layers to expand:
    nlayers(1) = # of layers on the north boundary to stretch
    nlayers(2) = # of layers on the south boundary to stretch
    nlayers(3) = # of layers on the east boundary to stretch
    nlayers(4) = # of layers on the west boundary to stretch
    
    factor - cumulative factor by which the layers are to be
      expanded. As with nlayers, this can be a 4-vector.
    
    method - 4-letter string specifying the method of
      stretching for each of the four boundaries. Four different
      methods are supported: uniform, linear, parabolic (default)
      and geometric.

    OUTPUT:

    x,y - the vertices of the new stretched grid
    """

    xx = x.astype(complex)
    yy = y.astype(complex)

    nlayers *= numpy.ones(4)
    nlayers = nlayers.astype(int)
    factor *= numpy.ones(4)

    for idx, (n, f, m) in enumerate(zip(nlayers, factor, method.upper())):

        if n > 0 and f != 1:

            if idx == 0:
                # north boundary
                kv = numpy.arange(len(y) - 1 - n, len(y))
                z = yy
                q1 = z[-1 - n]
                q2 = z[-1]
            elif idx == 1:
                # south boundary
                kv = numpy.arange(0, n)
                z = yy
                q1 = z[n]
                q2 = z[0]
            elif idx == 2:
                # east boundary
                kv = numpy.arange(len(x) - 1 - n, len(x))
                z = xx
                q1 = z[-1 - n]
                q2 = z[-1]
            elif idx == 3:
                # west boundary
                kv = numpy.arange(0, n)
                z = xx
                q1 = z[n]
                q2 = z[0]

            kv = kv.astype(int)

            if m == 'U':
                c = numpy.polyfit([q1, q2], [q1, q1 + f * (q2 - q1)], 1)
                z[kv] = numpy.polyval(c, z[kv])
            elif m == 'L':
                c = (f - 1) / (q2 - q1)
                b = 1 - 2 * c * q1
                a = q1 - b * q1 - c * q1 ** 2
                z[kv] = a + b * z[kv] + c * z[kv] ** 2
            elif m == 'P':
                z[kv] = z[kv] + (f - 1) * (z[kv] - q1) ** 3 / (q2 - q1) ** 2
            elif m == 'G':
                b = scipy.optimize.newton(
                    lambda s: numpy.exp(s) - 1 - f * s, f)
                a = (q2 - q1) / b
                z[kv] = q1 + a * (numpy.exp((z[kv] - q1) / a) - 1)

    xx = xx.real + 1j * numpy.abs(xx.imag)
    yy = yy.real - 1j * numpy.abs(yy.imag)

    return xx, yy
