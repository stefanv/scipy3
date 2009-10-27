import numpy as np
cimport numpy as np

# XXX: fix indexing, do not use int
ctypedef int index_t

cdef extern void dia_matvec_int_double(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, double diags[], double Xx[], double Yx[])

cdef extern void dia_matvec_int_npy_int8(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_int8 diags[], np.npy_int8 Xx[],
        np.npy_int8 Yx[])

def dia_matvec(index_t n_row, index_t n_col, index_t n_diags, 
               index_t L, np.ndarray offsets, np.ndarray diags,
               np.ndarray Xx, np.ndarray Yx):
    cdef np.ndarray safe_diags, safe_Xx, safe_Yx, safe_offsets

    if not offsets.dtype == np.int:
        raise ValueError("Expected %s for offsets, got %s" % 
                         (np.int, offsets.dtype))
    safe_offsets = np.ascontiguousarray(offsets, np.int)

    t = None
    for _t in [np.int8, np.uint8]:
        if np.can_cast(diags.dtype, t) and np.can_cast(Xx.dtype, t) \
                and np.can_cast(Yx.dtype, t):
            t = _t
            break

    if t is None:
        raise ValueError("type not supported %s - %s - %s" % \
                         (diags.dtype, Xx.dtype, Yx.dtype))

    safe_diags = np.ascontiguousarray(diags, t)
    safe_Xx = np.ascontiguousarray(Xx, t)
    safe_Yx = np.ascontiguousarray(Yx, t)
    if t == np.double:
        dia_matvec_int_double(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <double*>safe_diags.data,
                              <double*>safe_Xx.data,
                              <double*>safe_Yx.data)
    elif t == np.int8:
        dia_matvec_int_npy_int8(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_int8*>safe_diags.data,
                              <np.npy_int8*>safe_Xx.data,
                              <np.npy_int8*>safe_Yx.data)
    else:
        raise ValueError("Type %s not supported yet" % t)

    return safe_Yx
