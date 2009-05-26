#ifndef _SCIPY_SIGNAL_NEIGHITER_H_
#define _SCIPY_SIGNAL_NEIGHITER_H_

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

typedef struct {
    npy_intp nd;
    /* size is the number of neighbors */
    npy_intp size;
    char* dataptr;

    /*
     * privates attributes
     */
    npy_intp _dims[NPY_MAXDIMS];
    npy_intp _strides[NPY_MAXDIMS];
    npy_intp _bounds[NPY_MAXDIMS][2];
    /* _coordinates are relatively to the array point */
    npy_intp _coordinates[NPY_MAXDIMS];
    PyArrayIterObject* _iter;
    char* _zero;
} PyArrayNeighIterObject;

/*
 * Public API
 */
PyArrayNeighIterObject*
PyArrayNeighIter_New(PyArrayIterObject* iter, const npy_intp *bounds);

void PyArrayNeighIter_Delete(PyArrayNeighIterObject* iter);

static NPY_INLINE int PyArrayNeighIter_Reset(PyArrayNeighIterObject* iter);
static NPY_INLINE int PyArrayNeighIter_Next(PyArrayNeighIterObject* iter);

/*
 * Private API (here for inline)
 */
static NPY_INLINE int _PyArrayNeighIter_IncrCoord(PyArrayNeighIterObject* iter);
static NPY_INLINE int _PyArrayNeighIter_SetPtr(PyArrayNeighIterObject* iter);

/*
 * Inline implementations
 */
static NPY_INLINE int PyArrayNeighIter_Reset(PyArrayNeighIterObject* iter)
{
    int i;

    for(i = 0; i < iter->nd; ++i) {
        iter->_coordinates[i] = iter->_bounds[i][0];
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
static NPY_INLINE int _PyArrayNeighIter_IncrCoord(PyArrayNeighIterObject* iter)
{
    int i, wb;

    for(i = iter->nd-1; i >= 0; --i) {
        wb = iter->_coordinates[i] < iter->_bounds[i][1];
        if (wb) {
            iter->_coordinates[i] += 1;
            return 0;
        }
        else {
            iter->_coordinates[i] = iter->_bounds[i][0];
        }
    }

    return 0;
}

/* 
 * set the dataptr from its current coordinates
 */
static NPY_INLINE int _PyArrayNeighIter_SetPtr(PyArrayNeighIterObject* iter)
{
    int i;
    npy_intp offset, bd;

    iter->dataptr = iter->_iter->dataptr;

    for(i = 0; i < iter->nd; ++i) {
        /*
         * Handle cases where neighborhood point is outside the array
         */
        bd = iter->_coordinates[i] + iter->_iter->coordinates[i];
        if (bd < 0 || bd > iter->_dims[i]) {
            iter->dataptr = iter->_zero;
            return 1;
        }

        /*
         * At this point, the neighborhood point is guaranteed to be within the
         * array
         */
        offset = iter->_coordinates[i] * iter->_strides[i];
        iter->dataptr += offset;
    }

    return 0;
}


/*
 * Advance to the next neighbour
 */
static NPY_INLINE int PyArrayNeighIter_Next(PyArrayNeighIterObject* iter)
{
    _PyArrayNeighIter_IncrCoord (iter);
    _PyArrayNeighIter_SetPtr(iter);

    return 0;
}

#endif
