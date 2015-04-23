"""
Modificiation History
---------------------
2015-04     Written by Parke Loyd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def sawtooth_sift(t, y, bc='auto'):
    """
    Use the sawtooth transform to find the dominant intrinsic mode in the data.

    Parameters
    ----------
    t : 1D array-like
        The independent data, length N.
    y : 1D array-like
        The dependent data, length N.
    bc : {'auto'|'even'|'odd'|'periodic'|'extend'}, optional
        auto :
            default. Choose even for endpoints that extend beyond the
            nearest extremum, odd otherwise. If the endpoints are identical,
            choose periodic.
        even :
            use the endpoints as extrema and reflect the nearest exrema
            to extend the opposing envelope
        odd :
            reflect and flip the nearest two extrema about each endpoint
            without using the endpoint as an extremum (like an odd function
            with the endpoint as the origin) to extrapolate both envelopes
        periodic :
            treat the function (thus extrema) as periodic to append
            the necessary extra extrema
        extend :
            extrapolate the envelopes from the last two extram as
            necessary

    Returns
    -------
    h : 1D array
        The intrinsic mode, length N.

    References
    ----------
    http://arxiv.org/pdf/0710.3170.pdf
    """
    t, y = map(np.asarray, [t, y])

    # identify the relative extrema
    argext = _allrelextrema(y)
    T = t[argext]
    E = y[argext]

    # if there are too few extrema, raise an exception
    if len(argext) < 2:
        raise FlatFunction('Too few relative max and min to sift the series')

    ## add extra extrema as necessary for boundary conditions
    if bc == 'extend':
        # do nothing, let interpolate extrapolate :)
        pass
    else:
        if bc == 'auto':
            if y[0] == y[-1]:
                return sawtooth_sift(t, y, bc='periodic')
            if _inrange(y[0], *E[:2]):
                t0, tn1 = _reflect(t[0], T[:2])
                E0, En1 = _reflect(y[0], E[:2])
            else:
                t0, E0 = t[0], y[0]
                tn1, En1 = _reflect(t[0], T[1]), E[0]
            if _inrange(y[-1], *E[-2:]):
                tm, tmn1 = _reflect(t[-1], T[-2:])
                Em, Emn1 = _reflect(y[-1], E[-2:])
            else:
                tmn1, Emn1 = t[-1], y[-1]
                tm, Em = _reflect(t[-1], T[-1]), E[-1]
        elif bc == 'even':
                t0, E0 = t[0], y[0]
                tn1, En1 = _reflect(t[0], T[1]), E[0]
                tmn1, Emn1 = t[-1], y[-1]
                tm, Em = _reflect(t[-1], T[-1]), E[-1]
        elif bc == 'odd':
                t0, tn1 = _reflect(t[0], T[:2])
                E0, En1 = _reflect(y[0], E[:2])
                tm, tmn1 = _reflect(t[-1], T[-2:])
                Em, Emn1 = _reflect(y[-1], E[-2:])
        elif bc == 'periodic':
            t0, E0 = t[0], y[0]
            tn1, En1 = t[0] - (t[-1] - T[-1]), E[-1]
            tmn1, Emn1 = t[-1], y[-1]
            tm, Em = t[-1] + (T[0] - t[0]), E[0]
        else:
            raise ValueError('Boundary condition (bc) not understood.')

        N = len(T)
        T = np.insert(T, [0, 0, N, N], [tn1, t0, tmn1, tm])
        E = np.insert(E, [0, 0, N, N], [En1, E0, Emn1, Em])
        argext = np.insert(argext, [0, 0, N, N], [0, 0, len(t), len(t)])

    # linearly interpolate all extrema to form the sawtooth function
    saw = np.interp(t, T, E)

    # linearly interpolate alternating extrema to form upper and lower
    # envelopes (doesn't matter which is which)
    env1 = np.interp(t, T[::2], E[::2])
    env2 = np.interp(t, T[1::2], E[1::2])

    # average the envelopes
    env_mean = (env1 + env2) / 2.0

    # subtract mean from sawtooth
    hsaw = saw - env_mean

    # transform from sawtooth to data space
    u = _saw_transform(t, y, T, E, argext)
    h = np.interp(u, t, hsaw)

    return h

def _saw_transform(t, y, T, E, argext):
    """Return the sawtooth transform of the t coordinate."""
    u = []
    for i in range(1, len(argext) - 2):
        piece = slice(argext[i], argext[i+1])
        upiece = (T[i] + (y[piece] - E[i]) / (E[i+1] - E[i])
                    * (T[i+1] - T[i]))
        u.extend(upiece)
    return np.array(u)

def emd(t, y, Nmodes=None, method='spline', bc='auto'):
    """
    Decompose function into "intrinsic modes" using empirical mode
    decompisition from Huang et al. [1].

    Parameters
    ----------
    t : 1D array-like
        The independent data, length N.
    y : 1D array-like
        The dependent data, length N.
    Nmodes : int, optional
        The maximum number of modes to return.
    method : {'spline'|'sawtooth'}
        With method to use for sifting (identifieng the next dominant
        intrinsic mode). Spline is the originally published method by Huang [1].
        Sawtooth is faster method for identifying modes published on arXiv
        without peer review by Lu [2].
    Returns
    -------
    c : 2D array
        An NxM array giving M emprical modes as columns.
    r : 1D array
        The residual, length N.

    References
    ----------
    [1] Huang et al. (1998; RSPA 454:903)
    [2] http://arxiv.org/pdf/0710.3170.pdf

    Notes
    -----
    The function does not properly handle the special (and presumably rare)
    case where two consecutive, identical points form a relative maximum or
    minimum in the supplied data.
    """

    # groom the input
    t, y = map(np.asarray, [t, y])
    if t.ndim > 1:
        raise ValueError("t array must be 1D")
    if y.ndim > 1:
        raise ValueError("y array must be 1D")
    if method == 'spline':
        sift = spline_sift
    elif method == 'sawtooth':
        sift = sawtooth_sift
    else:
        raise ValueError('method not recognized')

    c = np.empty([len(y), 0])
    h, r = map(np.copy, [y, y])
    hold = np.zeros(y.shape)
    while True:
        try:
            while True:
                h = sift(t, h, bc=bc)
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

def spline_sift(t, y, nref=100, plot=False):
    #TODO: add boundary condition keyword
    """
    Identify the dominant "intinsic mode" in a series of data by fitting
    spline envelopes to the extrema.

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
    T, E  = t[argext], y[argext]
    tleft, yleft = T[0] - (T[nref:0:-1] - T[0]) , E[nref:0:-1]
    tright, yright = T[-1] + (T[-1] - T[-2:-nref-2:-1]), E[-2:-nref-2:-1]
    tall = np.concatenate([tleft, T, tright])
    yall = np.concatenate([yleft, E, yright])

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

def _reflect(x0, x):
    return x0 - (x - x0)