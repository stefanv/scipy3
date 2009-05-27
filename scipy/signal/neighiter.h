#ifndef _SCIPY_SIGNAL_NEIGHITER_H_
#define _SCIPY_SIGNAL_NEIGHITER_H_

#include <Python.h>

//#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
//#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

#ifndef NPY_INLINE
#define NPY_INLINE inline
#endif

typedef struct {
    /* Keep this as the first item, so that casting a PyArrayNeighIterObject*
     * to a PyArrayIterObject* works */
    PyArrayIterObject base;

    npy_intp nd;

    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp bounds[NPY_MAXDIMS][2];

    /* Neighborhood points coordinates are computed relatively to the point pointed
     * by _internal_iter */
    PyArrayIterObject* _internal_iter;
    char* zero;
} PyArrayNeighIterObject;

static PyTypeObject PyArrayNeighIter_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "foo.neigh_internal_iter",          /*tp_name*/
    sizeof(PyArrayNeighIterObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    0,                         /* tp_doc */
};


/*
 * Public API
 */
PyArrayNeighIterObject*
PyArrayNeighIter_New(PyArrayIterObject* iter, const npy_intp *bounds);

NPY_INLINE int PyArrayNeighIter_Reset(PyArrayNeighIterObject* iter);
NPY_INLINE int PyArrayNeighIter_Next(PyArrayNeighIterObject* iter);

/*
 * Private API (here for inline)
 */
NPY_INLINE int _PyArrayNeighIter_IncrCoord(PyArrayNeighIterObject* iter);
NPY_INLINE int _PyArrayNeighIter_SetPtr(PyArrayNeighIterObject* iter);

/*
 * Inline implementations
 */
NPY_INLINE int PyArrayNeighIter_Reset(PyArrayNeighIterObject* iter)
{
    int i;

    for(i = 0; i < iter->nd; ++i) {
        iter->base.coordinates[i] = iter->bounds[i][0];
    }
    _PyArrayNeighIter_SetPtr(iter);

    return 0;
}

/*
 * Update to next item of the iterator
 *
 * Note: this simply increment the coordinates vector, last dimension
 * incremented first , i.e, for dimension 3
 * ...
 * -1, -1, -1
 * -1, -1,  0
 * -1, -1,  1
 *  ....
 * -1,  0, -1
 * -1,  0,  0
 *  ....
 * 0,  -1, -1
 * 0,  -1,  0
 *  ....
 */
NPY_INLINE int _PyArrayNeighIter_IncrCoord(PyArrayNeighIterObject* iter)
{
    int i, wb;

    for(i = iter->nd-1; i >= 0; --i) {
        wb = iter->base.coordinates[i] < iter->bounds[i][1];
        if (wb) {
            iter->base.coordinates[i] += 1;
            return 0;
        }
        else {
            iter->base.coordinates[i] = iter->bounds[i][0];
        }
    }

    return 0;
}

/* set the dataptr from its current coordinates */
NPY_INLINE int _PyArrayNeighIter_SetPtr(PyArrayNeighIterObject* iter)
{
    int i;
    npy_intp offset, bd;
    PyArrayIterObject *base = (PyArrayIterObject*)iter;

    base->dataptr = iter->_internal_iter->dataptr;

    for(i = 0; i < iter->nd; ++i) {
        /*
         * Handle cases where neighborhood point is outside the array
         */
        bd = base->coordinates[i] + iter->_internal_iter->coordinates[i];
        if (bd < 0 || bd > iter->dimensions[i]) {
            base->dataptr = iter->zero;
            return 1;
        }

        /*
         * At this point, the neighborhood point is guaranteed to be within the
         * array
         */
        offset = base->coordinates[i] * base->strides[i];
        base->dataptr += offset;
    }

    return 0;
}

/*
 * Advance to the next neighbour
 */
NPY_INLINE int PyArrayNeighIter_Next(PyArrayNeighIterObject* iter)
{
    _PyArrayNeighIter_IncrCoord (iter);
    _PyArrayNeighIter_SetPtr(iter);

    return 0;
}

#endif
