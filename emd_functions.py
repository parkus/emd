"""
Empirical Mode Decomposition from Huang et al. (1998; RSPA 454:903).

Modificiation History
---------------------
2015-04     Written by Parke Loyd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def emd(t, y, Nmodes=None):
    """
    Decompose function into "intrinsic modes" using empirical mode
    decompisition.

    Parameters
    ----------
    t : 1D array-like
        The independent data, length N.
    y : 1D array-like
        The dependent data, length N.
    Nmodes : int, optional
        The maximum number of modes to return.

    Returns
    -------
    c : 2D array
        An NxM array giving M emprical modes as columns.
    r : 1D array
        The residual, length N.

    References
    ----------
    Huang et al. (1998; RSPA 454:903)
    """

    t, y = map(np.asarray, [t, y])
    if t.ndim > 1:
        raise ValueError("t array must be 1D")
    if y.ndim > 1:
        raise ValueError("y array must be 1D")

    c = np.empty([len(y), 0])
    h, r = map(np.copy, [y, y])
    hold = np.zeros(y.shape)
    while True:
        try:
            while True:
                h = sift(t, h)
                var = np.sum((h-hold)**2 / hold**2)
                if var < 0.25:
                    c = np.append(c, h[:, np.newaxis], axis=1)
                    r = r - h

                    #if the user doesn't want any more modes
                    if len(c) == Nmodes:
                        return c, r

                    h = r
                    hold = np.zeros(y.shape)
                    break
                hold = h
        except FlatFunction: #if the residue has too few extrema
            return c, r

class FlatFunction(Exception):
    pass

def sift(t, y, nref=100, plot=False):
    """
    Identify the dominant "intinsic mode" in a series of data.

    Parameters
    ----------
    t : 1D array-like
        The independent data, length N.
    y : 1D array-like
        The dependent data, length N.
    nref : int, optional
        Number of extema to reflect about each end when fitting splines.
    plot : {True|False}, optional
        If True, create a diagnostic plot of the function and results using
        the matplotlib.pyplot plot functions. If there is alread a plot
        window active, the plotting will be done there. Plot handles are not
        returned.

    Returns
    -------
    h : 1D array
        The intrinsic mode, length N.

    Summary
    -------
    Identifies the relative max and min in the series, fits spline curves
    to these to estimate an envelope, then subtracts the mean of the envelope
    from the series. The difference is then returned. The extrema are refelcted
    about the extrema nearest each end of the series to mitigate end
    effects, where nref controls the maximum total number of extrema (max and
    min) that are reflected.

    References
    ----------
    Huang et al. (1998; RSPA 454:903)

    """

    # identify the relative extrema
    argext = _allrelextrema(y)

    # if there are too few extrema, raise an exception
    if len(argext) < 2:
        raise FlatFunction('Too few max and min in the series to sift')

    # include the left and right endpoints as extrema if they are beyond the
    # limits set by the nearest two extrema
    inclleft = not _inrange(y[[0]], y[argext[0]], y[argext[1]])
    inclright = not _inrange(y[[-1]], y[argext[-2]], y[argext[-1]])
    if inclleft and inclright: argext = np.concatenate([[0],argext,[-1]])
    if inclleft and not inclright: argext = np.insert(argext,0,0)
    if not inclleft and inclright: argext = np.append(argext,-1)
    #if neither, do nothing

    # now reflect the extrema about both sides
    text, yext  = t[argext], y[argext]
    tleft, yleft = text[0] - (text[nref:0:-1] - text[0]) , yext[nref:0:-1]
    tright, yright = text[-1] + (text[-1] - text[-2:-nref-2:-1]), yext[-2:-nref-2:-1]
    tall = np.concatenate([tleft, text, tright])
    yall = np.concatenate([yleft, yext, yright])

    # parse out the min and max. the extrema must alternate, so just figure out
    # whether a min or max comes first
    if yall[0] < yall[1]:
        tmin, tmax, ymin, ymax = tall[::2], tall[1::2], yall[::2], yall[1::2]
    else:
        tmin, tmax, ymin, ymax = tall[1::2], tall[::2], yall[1::2], yall[::2]

    # check again if there are enough extrema, now that the endpoints may have
    # been added
    if len(tmin) < 4 or len(tmax) < 4:
        raise FlatFunction('Too few max and min in the series to sift')

    # compute spline enevlopes and mean
    spline_min, spline_max = map(interp1d, [tmin,tmax], [ymin,ymax], ['cubic']*2)
    m = (spline_min(t) + spline_max(t))/2.0
    h = y - m

    if plot:
        plt.plot(t, y, '-', t, m, '-')
        plt.plot(tmin, ymin, 'g.', tmax, ymax, 'k.')
        tmin = np.linspace(tmin[0], tmin[-1], 1000)
        tmax = np.linspace(tmax[0], tmax[-1], 1000)
        plt.plot(tmin, spline_min(tmin), '-r', tmax, spline_max(tmax), 'r-')

    return h

def _allrelextrema(y):
    """
    Finds all of the relative extrema in order in about half the time
    as using the scipy.signal.argrel{min|max} functions and combining the
    results.
    """
    # compute difference between successive values (like the slope)
    slope = y[1:] - y[:-1]

    # we just want the sign of the slope
    slope_sign = np.zeros(len(slope), np.int8)
    slope_sign[slope > 0] = 1
    slope_sign[slope < 0] = -1

    # so that we can find the sign of the curvature at points with differing
    # slope signs to either side
    curve_sign = slope_sign[1:] - slope_sign[:-1]
    argext = np.nonzero(curve_sign != 0)[0] + 1
    return argext

def _inrange(y, y0, y1):
    """
    Return True if y is within the range (y0, y1).
    """
    if y0 > y1:
        return (y < y0) and (y > y1)
    else:
        return (y > y0) and (y < y1)