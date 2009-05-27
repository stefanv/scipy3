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

    ret = PyArray_malloc(sizeof(*ret));
    PyObject_Init((PyObject *)ret, &PyArrayNeighIter_Type);
    if (ret == NULL) {
        return NULL;
    }

    Py_INCREF(x);
    ret->_internal_iter = x;

    Py_INCREF(x->ao);
    ret->base.ao = x->ao;
    ret->nd = x->ao->nd;

    /* Compute the neighborhood size and copy the shape */
    ret->base.size = 1;
    for(i = 0; i < ret->nd; ++i) {
        ret->bounds[i][0] = bounds[2 * i];
        ret->bounds[i][1] = bounds[2 * i + 1];
        ret->base.size *= (bounds[2*i+1] - bounds[2*i]) + 1;
    }

    for(i = 0; i < ret->nd; ++i) {
        ret->base.strides[i] = x->ao->strides[i];
        ret->dimensions[i] = x->ao->dimensions[i];
    }
    ret->zero = PyArray_Zero(x->ao);

    /*
     * XXX: we force x iterator to be non contiguous because we need
     * coordinates... Modifying the iterator here is not great
     */
    x->contiguous = 0;

    PyArrayNeighIter_Reset(ret);

    return ret;
}

#if 0
void PyArrayNeighIter_Delete(PyArrayNeighIterObject* iter)
{
    PyDataMem_FREE(iter->_zero);
    free(iter);
}
#endif
