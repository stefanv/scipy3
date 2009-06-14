#ifndef _SCIPY_SIGNAL_NEIGHITER_H_
#define _SCIPY_SIGNAL_NEIGHITER_H_

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

typedef struct {
    PyObject_HEAD

    /* PyArrayIterObject part: keep this in this exact order */
    int               nd_m1;            /* number of dimensions - 1 */
    npy_intp          index, size;
    npy_intp          coordinates[NPY_MAXDIMS];/* N-dimensional loop */
    npy_intp          dims_m1[NPY_MAXDIMS];    /* ao->dimensions - 1 */
    npy_intp          strides[NPY_MAXDIMS];    /* ao->strides or fake */
    npy_intp          backstrides[NPY_MAXDIMS];/* how far to jump back */
    npy_intp          factors[NPY_MAXDIMS];     /* shape factors */
    PyArrayObject     *ao;
    char              *dataptr;        /* pointer to current item*/
    npy_bool          contiguous;

    /* New members */
    npy_intp nd;

    /* Dimensions is the dimension of the array */
    npy_intp dimensions[NPY_MAXDIMS];
    /* Bounds of the neighborhood to iterate over */
    npy_intp bounds[NPY_MAXDIMS][2];

    /* Neighborhoodborhood points coordinates are computed relatively to the point pointed
     * by _internal_iter */
    PyArrayIterObject* _internal_iter;
    /* To keep a reference to the zero representation correponding to the dtype
     * of the array we iterate over */
    char* zero;
} PyArrayNeighborhoodIterObject;

PyTypeObject PyArrayNeighborhoodIter_Type;

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
 *      PyArrayIterObject *iter;
 *      PyArrayNeighIterObject *neigh_iter;
 *      iter = PyArray_IterNew(x);
 *
 *      // For a 3x3 kernel
 *      bounds = {-1, 1, -1, 1};
 *      neigh_iter = PyArrayNeighIter_New(iter, bounds);
 *
 *      for(i = 0; i < iter->size; ++i) {
 *              for (j = 0; j < neigh_iter->size; ++j) {
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
PyArrayNeighborhoodIterObject*
PyArrayNeighborhoodIter_New(PyArrayIterObject* iter, const npy_intp *bounds);

static NPY_INLINE int PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter);

/*
 * Private API (here for inline)
 */
static NPY_INLINE int _PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int _PyArrayNeighborhoodIter_SetPtr(PyArrayNeighborhoodIterObject* iter);

/*
 * Inline implementations
 */
static NPY_INLINE int PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter)
{
    int i;

    for(i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    _PyArrayNeighborhoodIter_SetPtr(iter);

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
#define _UPDATE_COORD_ITER(c) \
    wb = iter->coordinates[c] < iter->bounds[c][1]; \
    if (wb) { \
        iter->coordinates[c] += 1; \
        return 0; \
    } \
    else { \
        iter->coordinates[c] = iter->bounds[c][0]; \
    }

static NPY_INLINE int _PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter)
{
    int i, wb;

    for(i = iter->nd-1; i >= 0; --i) {
        _UPDATE_COORD_ITER(i)
    }

    return 0;
}

/*
 * Version optimized for 2d arrays, manual loop unrolling
 */
static NPY_INLINE int _PyArrayNeighborhoodIter_IncrCoord2D(PyArrayNeighborhoodIterObject* iter)
{
    int wb;

    _UPDATE_COORD_ITER(1)
    _UPDATE_COORD_ITER(0)

    return 0;
}
#undef _UPDATE_COORD_ITER

#define _INF_SET_PTR(c) \
    bd = iter->coordinates[c] + iter->_internal_iter->coordinates[c]; \
    if (bd < 0 || bd > iter->dimensions[c]) { \
        iter->dataptr = iter->zero; \
        return 1; \
    } \
    offset = iter->coordinates[c] * iter->strides[c]; \
    iter->dataptr += offset;

/* set the dataptr from its current coordinates */
static NPY_INLINE int _PyArrayNeighborhoodIter_SetPtr(PyArrayNeighborhoodIterObject* iter)
{
    int i;
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

    for(i = 0; i < iter->nd; ++i) {
        _INF_SET_PTR(i)
    }

    return 0;
}

static NPY_INLINE int _PyArrayNeighborhoodIter_SetPtr2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

    _INF_SET_PTR(0)
    _INF_SET_PTR(1)

    return 0;
}
#undef _INF_SET_PTR

/*
 * Advance to the next neighbour
 */
static NPY_INLINE int PyArrayNeighborhoodIter_Next2D(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord2D(iter);
    _PyArrayNeighborhoodIter_SetPtr2D(iter);

    return 0;
}

static NPY_INLINE int PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord (iter);
    _PyArrayNeighborhoodIter_SetPtr(iter);

    return 0;
}
#endif
