#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

#include "neighiter.h"

PyArrayNeighIterObject*
PyArrayNeighIter_New(PyArrayIterObject *x, const npy_intp *bounds)
{
    int i;
    PyArrayNeighIterObject *ret;

    ret = malloc(sizeof(*ret));
    if (ret == NULL) {
        return NULL;
    }
    ret->nd = x->ao->nd;

    ret->_iter = x;

    /* Compute the neighborhood size and copy the shape */
    ret->size = 1;
    for(i = 0; i < ret->nd; ++i) {
        ret->_bounds[i][0] = bounds[2 * i];
        ret->_bounds[i][1] = bounds[2 * i + 1];
        ret->size *= (bounds[2*i+1] - bounds[2*i]) + 1;
    }

    for(i = 0; i < ret->nd; ++i) {
        ret->_strides[i] = x->ao->strides[i];
        ret->_dims[i] = x->ao->dimensions[i];
    }
    ret->_zero = PyArray_Zero(x->ao);

    /*
     * XXX: we force x iterator to be non contiguous because we need
     * coordinates... Modifying the iterator here is not great
     */
    x->contiguous = 0;

    PyArrayNeighIter_Reset(ret);

    return ret;
}

void PyArrayNeighIter_Delete(PyArrayNeighIterObject* iter)
{
    PyDataMem_FREE(iter->_zero);
    free(iter);
}
