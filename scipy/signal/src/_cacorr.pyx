import numpy as np
cimport numpy as c_np
cimport stdlib

def xcorr(c_np.ndarray x, c_np.ndarray y, c_np.ndarray out, minlag, maxlag):
    """Cython version of autocorrelation, direct implementation. This can be
    faster than FFT for small size or for maxlag << x.shape[axis]."""
    cdef c_np.npy_intp nd, nx, ny

    nd = x.ndim
    nx = x.shape[nd-1]
    ny = y.shape[nd-1]

    dt = out.dtype
    if dt == np.double:
        acorr_double(<double*>x.data, <double*>y.data, <double*>out.data,
                     x.shape, y.shape, x.ndim, minlag, maxlag)
    #elif dt == np.single:
    #    print "Single"
    #elif dt == np.singlecomplex:
    #    print "Single Complex"
    elif dt == np.complex128:
        acorr_cdouble(<double complex*>x.data, <double complex*>y.data,
                <double complex*>out.data,
                x.shape, y.shape, x.ndim, minlag, maxlag)
    else:
        raise NotImplemented("Support for type %s not implemented" % dt)

    return out

cdef int acorr_double(double* x, double *y, double* out, c_np.npy_intp *xdims,
        c_np.npy_intp *ydims, c_np.npy_intp nd,
        c_np.npy_intp minlag, c_np.npy_intp maxlag):
    cdef c_np.npy_intp i, nx, ny, ncorr

    nx = xdims[nd-1]
    ny = ydims[nd-1]

    ncorr = 1
    for i in range(nd-1):
        ncorr *= xdims[i]

    for i in range(ncorr):
        if nx > ny:
            _acorr_double_large_nx(x, y, out, nx, ny, minlag, maxlag)
        else:
            _acorr_double_large_ny(x, y, out, nx, ny, minlag, maxlag)

        x += nx
        y += ny
        out += maxlag - minlag + 1

    return 0

cdef int _acorr_double_large_nx(double* x, double *y, double* out, c_np.npy_intp nx,
        c_np.npy_intp ny, c_np.npy_intp minlag, c_np.npy_intp maxlag):
    cdef c_np.npy_intp i, j
    cdef double acc

    for i in range(ny-1+minlag, ny+maxlag):
        acc = 0
        # -ny + 1 <= lag <= 0
        if i < ny:
            for j in range(i+1):
                acc += x[j] * y[ny-1-i+j]
        # 0 < lag <= nx - ny
        elif i < nx:
            for j in range(ny):
                acc += x[j+i-(ny-1)] * y[j]
        # ny - nx < lag <= nx - 1
        else:
            for j in range(ny-1-(i-nx)):
                acc += x[i+j-(ny-1)] * y[j]

        out[0] = acc
        out += 1

    return 0

cdef int _acorr_double_large_ny(double* x, double *y, double* out, c_np.npy_intp nx,
        c_np.npy_intp ny, c_np.npy_intp minlag, c_np.npy_intp maxlag):
    cdef c_np.npy_intp i, j
    cdef double acc

    for i in range(ny-1+minlag, ny+maxlag):
        acc = 0
        # -ny + 1 <= lag <= nx - ny
        if i < nx:
            for j in range(i+1):
                acc += x[j] * y[ny-1-i+j]
        # nx - ny < lag <= 0
        elif i < ny:
            for j in range(nx):
                acc += x[j] * y[ny-1-i+j]
        # 0 < lag <= nx - 1
        else:
            for j in range(ny-1-(i-nx)):
                acc += x[i+j-(ny-1)] * y[j]

        out[0] = acc
        out += 1

    return 0

cdef int acorr_cdouble(double complex* x, double complex *y, double complex* out,
        c_np.npy_intp *xdims,
        c_np.npy_intp *ydims, c_np.npy_intp nd,
        c_np.npy_intp minlag, c_np.npy_intp maxlag):
    cdef c_np.npy_intp i, nx, ny, ncorr

    nx = xdims[nd-1]
    ny = ydims[nd-1]

    ncorr = 1
    for i in range(nd-1):
        ncorr *= xdims[i]

    for i in range(ncorr):
        if nx > ny:
            _acorr_cdouble_large_nx(x, y, out, nx, ny, minlag, maxlag)
        else:
            _acorr_cdouble_large_ny(x, y, out, nx, ny, minlag, maxlag)

        x += nx
        y += ny
        out += maxlag - minlag + 1

    return 0

cdef int _acorr_cdouble_large_nx(double complex* x, double complex *y,
        double complex* out, c_np.npy_intp nx,
        c_np.npy_intp ny, c_np.npy_intp minlag, c_np.npy_intp maxlag):
    cdef c_np.npy_intp i, j
    cdef double complex acc

    for i in range(ny-1+minlag, ny+maxlag):
        acc = 0
        # -ny + 1 <= lag <= 0
        if i < ny:
            for j in range(i+1):
                acc = acc + x[j] * y[ny-1-i+j]
        # 0 < lag <= nx - ny
        elif i < nx:
            for j in range(ny):
                acc = acc + x[j+i-(ny-1)] * y[j]
        # ny - nx < lag <= nx - 1
        else:
            for j in range(ny-1-(i-nx)):
                acc = acc + x[i+j-(ny-1)] * y[j]

        out[0] = acc
        out += 1

    return 0

cdef int _acorr_cdouble_large_ny(double complex* x, double complex *y, double complex* out, c_np.npy_intp nx,
        c_np.npy_intp ny, c_np.npy_intp minlag, c_np.npy_intp maxlag):
    cdef c_np.npy_intp i, j
    cdef double complex acc

    for i in range(ny-1+minlag, ny+maxlag):
        acc = 0
        # -ny + 1 <= lag <= nx - ny
        if i < nx:
            for j in range(i+1):
                acc = acc + x[j] * y[ny-1-i+j]
        # nx - ny < lag <= 0
        elif i < ny:
            for j in range(nx):
                acc = acc + x[j] * y[ny-1-i+j]
        # 0 < lag <= nx - 1
        else:
            for j in range(ny-1-(i-nx)):
                acc = acc + x[i+j-(ny-1)] * y[j]

        out[0] = acc
        out += 1

    return 0

