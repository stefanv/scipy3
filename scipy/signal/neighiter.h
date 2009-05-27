#ifndef _SCIPY_SIGNAL_NEIGHITER_H_
#define _SCIPY_SIGNAL_NEIGHITER_H_

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

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

PyTypeObject PyArrayNeighIter_Type;

/*
 * Public API
 */

/* 
 * Main ctor
 *
 * bounds is expected to be a (2 * iter->ao->nd) arrays, such as the range
 * bound[2*i]->bounds[2*i+1] defines the range where to walk for dimension i
 * (both bounds are included in the walked coordinates). 
 *
 * Example:
 *      PyArrayIterObject *iter, *neigh_iter_base;
 *      PyArrayNeighIterObject *neigh_iter;
 *      iter = PyArray_IterNew(x);
 *       
 *      // For a 3x3 kernel
 *      bounds = {-1, 1, -1, 1};
 *      neigh_iter = PyArrayNeighIter_New(iter, bounds);
 *      // Hack so that neigh_iter_base points to iter, but giving access to
 *      // the base properties of iter (PyArrayIterObject is inherited from
 *      // PyArrayObject)
 *      neigh_iter_base = (PyArrayIterObject*)iter;
 *
 *      for(i = 0; i < iter->size; ++i) {
 *              for (j = 0; j < neigh_iter_base->size; ++j) {
 *                      // Walk around the item currently pointed by iter->dataptr
 *                      PyArrayNeighIter_Next(neigh_iter);
 *              }
 *
 *              // Move to the next point of iter
 *              PyArrayIter_Next(iter);
 *              PyArrayNeighIter_Reset(neigh_iter);
 *      }
 *
 * If the coordinates point to a point outside the array, the iterator points
 * to 0. More elaborate schemes (constants, mirroring, repeat) may be
 * implemented later.
 */
PyArrayNeighIterObject*
PyArrayNeighIter_New(PyArrayIterObject* iter, const npy_intp *bounds);

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
static NPY_INLINE int _PyArrayNeighIter_IncrCoord(PyArrayNeighIterObject* iter)
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
static NPY_INLINE int _PyArrayNeighIter_SetPtr(PyArrayNeighIterObject* iter)
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
static NPY_INLINE int PyArrayNeighIter_Next(PyArrayNeighIterObject* iter)
{
    _PyArrayNeighIter_IncrCoord (iter);
    _PyArrayNeighIter_SetPtr(iter);

    return 0;
}

#endif
