"""
Modificiation History
---------------------
2015-04     Written by Parke Loyd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def saw_sift(t, y, bc='extrap', tol=0.0):
    """
    Use the sawtooth transform to find the dominant intrinsic mode in the data.

    Parameters
    ----------
    t : 1D array-like
        The independent data, length N.
    y : 1D array-like
        The dependent data, length N.
    bc : {'auto'|'even'|'odd'|'periodic'|'extend'}, optional
        extrap :
            default. extrapolate the envelopes from the last two extram as
            necessary
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
        tol : float
            tolerance. changes between points below this level are set to zero.

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
    argext = _allrelextrema(y, tol=tol)

    # if there are too few extrema, raise an exception
    if len(argext) < 2:
        raise FlatFunction('Too few relative max and min to sift the series')

    # parse out the relative extrema
    T = t[argext]
    E = y[argext]

    ## add extra extrema as necessary for boundary conditions
    if bc == 'extrap':
        if len(argext) < 4:
            raise FlatFunction('Too few relative max and min to sift the series')
        t0, tn1 = t[0], t[0]
        E0 = (E[3] - E[1]) / (T[3] - T[1]) * (t0 - T[1]) + E[1]
        En1 = (E[2] - E[0]) / (T[2] - T[0]) * (tn1 - T[0]) + E[0]
        tmn1, tm = t[-1], t[-1]
        Emn1 = (E[-4] - E[-2]) / (T[-4] - T[-2]) * (tmn1 - T[-2]) + E[-2]
        Em = (E[-3] - E[-1]) / (T[-3] - T[-1]) * (tm - T[-1]) + E[-1]
    elif bc == 'even':
        t0, E0 = t[0], y[0]
        tn1, En1 = _reflect(t[0], T[0]), E[0]
        tmn1, Emn1 = t[-1], y[-1]
        tm, Em = _reflect(t[-1], T[-1]), E[-1]
    elif bc == 'odd':
        t0, tn1 = _reflect(t[0], T[:2])
        E0, En1 = _reflect(y[0], E[:2])
        tm, tmn1 = _reflect(t[-1], T[-2:])
        Em, Emn1 = _reflect(y[-1], E[-2:])
    elif bc == 'periodic':
        if _oppsign(y[1] - y[0], y[0] - y[-1], tol):
            # left endpt is a relative extremum
            t0, E0 = t[0], y[0]
            tn1, En1 = t[0] - (t[-1] - T[-1]), E[-1]
        else:
            t0, E0 = t[0] - (t[-1] - T[-1]), E[-1]
            tn1, En1 = t[0] - (t[-1] - T[-2]), E[-2]
        if _oppsign(y[-1] - y[-2], y[0] - y[-1], tol):
            # right endpt is a relative extremum
            tmn1, Emn1 = t[-1], y[-1]
            tm, Em = t[-1] + (T[0] - t[0]), E[0]
        else:
            tmn1, Emn1 = t[-1] + (T[0] - t[0]), E[0]
            tm, Em = t[-1] + (T[1] - t[0]), E[1]
    else:
        raise ValueError('Boundary condition (bc) not understood.')

    # add the boundary points to the extrema
    N = len(T)
    T = np.insert(T, [0, 0, N, N], [tn1, t0, tmn1, tm])
    E = np.insert(E, [0, 0, N, N], [En1, E0, Emn1, Em])
    argext = np.insert(argext, [0, 0, N, N], [0, 0, len(t), len(t)])

    # parse the saw function and envelope points
    Tsaw, Ysaw = T[1:-1], E[1:-1]
    env1 = np.interp(Tsaw, T[::2], E[::2])
    env2 = np.interp(Tsaw, T[1::2], E[1::2])

    # subtract envelope mean from sawtooth (sawtooth is the extrema)
    env_mean = (env1 + env2) / 2.0
    Hsaw = Ysaw - env_mean

    # transform from sawtooth to data space
    u = _saw_transform(t, y, T, E, argext)
    h = np.interp(u, Tsaw, Hsaw)

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

def saw_emd(t, y, Nmodes=None, bc='extrap', tol=1e-10):
    """
    Decompose function into "intrinsic modes" using the decomposition method
    of Lu [1].

    Parameters
    ----------
    t : 1D array-like
        The independent data, length N.
    y : 1D array-like
        The dependent data, length N.
    Nmodes : int, optional
        The maximum number of modes to return.
    bc : {'auto'|'even'|'odd'|'periodic'|'extend'}, optional
        extrap :
            default. extrapolate the envelopes from the last two extram as
            necessary
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
    tol : float
        tolerance relative to the initial range of the function. decomposition
        will stop once wiggles in y are below this level.

    Returns
    -------
    c : 2D array
        An NxM array giving M emprical modes as columns.
    r : 1D array
        The residual, length N.

    References
    ----------
    [1] http://arxiv.org/pdf/0710.3170.pdf

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

    atol = tol * (np.max(y) - np.min(y))

    c = []
    r = np.copy(y)
    while True:
        try:
            h = saw_sift(t, r, bc=bc, tol=atol)
            c.append(h)
            r = r - h
        except FlatFunction: #if the residue has too few extrema
            break
        if len(c) == Nmodes:
            break

    return np.transpose(c), r

def emd(t, y, Nmodes=None):
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

    Returns
    -------
    c : 2D array
        An NxM array giving M emprical modes as columns.
    r : 1D array
        The residual, length N.

    References
    ----------
    [1] Huang et al. (1998; RSPA 454:903)

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

def _allrelextrema(y, tol=0.0):
    """
    Finds all of the relative extrema in order in about half the time
    as using the scipy.signal.argrel{min|max} functions and combining the
    results. The scipy.signal version also misses multi-point max and mins.
    This version returns the midpoint of multi point extrema, or the point
    just left of the middle for multi-point exrtema.
    """

    # compute difference between successive values (like the slope)
    slope = np.diff(y)
    slope[np.abs(slope) < tol] = 0.0

    # remove all zeros while tracking original indices
    nonzero = (slope != 0.0)
    slope = slope[nonzero]
    indices = np.arange(len(y) - 1)
    indices = indices[nonzero]

    # we just want the sign of the slope
    slope_sign = np.zeros(len(slope), 'i1')
    slope_sign[slope > 0] = 1
    slope_sign[slope < 0] = -1

    # so that we can find the sign of the curvature at points with differing
    # slope signs to either side
    curve_sign = np.diff(slope_sign)
    arg_curve_chng = np.nonzero(curve_sign != 0)[0]
    i0 = indices[arg_curve_chng]
    i1 = indices[arg_curve_chng + 1] + 1
    i = np.floor((i0 + i1) / 2.0)

    return i.astype(int)

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

def _oppsign(x, y, tol):
    return (x < -tol and y > tol) or (x > tol and y < -tol)