import numpy as np
from scipy.signal._cacorr import xcorr as _xcorr1

__all__ = ['xcorr1']

def xcorr1(x, y, minlag=None, maxlag=None, axis=-1):
    """Compute the one-dimensional cross correlation between the given vectors
    on the given axis.

    x and y should have the same number of dimensions, and the same dimensions
    on every axis but the one on which the cross correlation is computed.

    Parameters
    ----------
    x: ndarray
        first input
    y: ndarray
        second input
    minlag: int
        minimum lag computed
    maxlag: int
        maximum lag computed
    axis: int
        axis on which to compute the one-dimensional cross correlation.

    Note
    ----
    Use the direct implementation (not FFT), so it is mosly useful if only a
    few lags of the cross-correlation are needed.

    Example
    -------

    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([[1, 3, 5], [2, 4, 6]])
    >>> xcorr(x, y)
    array([[  5.,  13.,  22.,  11.,   3.],
           [ 24.,  46.,  64.,  34.,  12.]])
    >>> xcorr(x, y, 0, 2)
    array([[ 22.,  11.,   3.],
           [ 64.,  34.,  12.]])
    >>> xcorr(x, y, -2, -1)
    array([[  5.,  13.],
           [ 24.,  46.]])
    >>> xcorr(x, y, axis=0)
    array([[  2.,   8.,  18.],
           [  9.,  26.,  51.],
           [  4.,  15.,  30.]])
    """

    # We check those here so that we can give a more precise error message if
    # dimensions do not match
    nd = x.ndim
    if not y.ndim == x.ndim:
        raise ValueError("Both input should have same rank !")
    elif nd < 1:
        raise ValueError("Input should be at least of rank >= 1")

    if axis < 0:
        axis += nd
    if axis < 0 or axis > nd - 1:
        raise ValueError("Axis is out of bounds")

    _to_check = [i for i in range(nd) if not i == axis]
    for i in _to_check:
        if not x.shape[i] == y.shape[i]:
            raise ValueError(
                    "Input should have same dimension for axis %d (%d vs %d)" % \
                    (i, x.shape[i], y.shape[i]))

    if axis == nd - 1:
        return _xcorr1_last_axis(x, y, minlag, maxlag)
    else:
        xtmp = np.swapaxes(x, axis, -1)
        ytmp = np.swapaxes(y, axis, -1)
        out = _xcorr1_last_axis(xtmp, ytmp, minlag, maxlag)
        return np.swapaxes(out, axis, -1)

def _xcorr1_last_axis(x, y, minlag, maxlag):
    nd = x.ndim
    dt = np.common_type(x, y)

    # Make sure we have contiguous, C arrays, and a common type
    x = np.require(x, dt, ['C', 'A'])
    if np.iscomplexobj(y):
        # XXX: we conjugate here because cython complex support is limited as now,
        # and I did not find a way to compute conjugate "on the fly" - that would
        # be better, as no copy would be needed in that case
        y = np.require(np.conj(y), dt, ['C', 'A'])
    else:
        y = np.require(y, dt, ['C', 'A'])

    # Check minlag/maxlag values and set default if not given
    nx = x.shape[nd-1]
    ny = y.shape[nd-1]
    if minlag is None:
        minlag = -ny+1
    else:
        if minlag < -ny + 1:
            raise ValueError("minlag too small (cannot be smaller than %d)" % (-ny+1))

    if maxlag is None:
        maxlag = nx-1
    else:
        if maxlag > nx - 1:
            raise ValueError("maxlag too big (cannot be > %d)" % (nx -1))

    if minlag > maxlag:
        raise ValueError("minlag < maxlag")

    odim = []
    for i in range(nd-1):
        odim.append(x.shape[i])
    odim.append(-minlag + maxlag + 1)

    out = np.empty(odim, dtype=dt, order='C')
    _xcorr1(x, y, out, minlag, maxlag)
    return out
