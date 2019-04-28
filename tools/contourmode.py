import numpy as np
import matplotlib.pyplot as plt


def contour(x, y, Hx1, Hy1):
    """
    Produces a contour plot (in dB) of one field component of the
    mode of an optical waveguide.

    INPUT:
    x,y - vectors describing horizontal and vertical grid points
    mode - the mode or field component to be plotted
"""

    x = x.real
    y = y.real

    dBrange = np.arange(-48, 0, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    CS1 = ax1.contour(x, y, 20 * np.log10(abs(Hx1.T)), dBrange, colors='k', linestyles='solid')
    ax1.clabel(CS1, fmt='%d')
    ax1.set_title("(Hy (TE mode, PML applied)")
    ax1.set_xlabel("x(um)")
    ax1.set_ylabel("y(um)")

    CS2 = ax2.contour(x, y, 20 * np.log10(abs(Hy1.T)), dBrange, colors='k', linestyles='solid')
    ax2.clabel(CS2, fmt='%d')
    ax2.set_title("(Hy (TE mode, no PML)")
    ax2.set_xlabel("x(um)")
    ax2.set_ylabel("y(um)")

    #plt.savefig('pml.png')
    plt.show()

