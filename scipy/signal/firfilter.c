#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/noprefix.h>

#include "sigtools.h"
static int elsizes[] = {sizeof(Bool),
						sizeof(byte),
                        sizeof(ubyte),
                        sizeof(short),
                        sizeof(ushort),
                        sizeof(int),
						sizeof(uint),
						sizeof(long),
                        sizeof(ulong),
                        sizeof(longlong),
						sizeof(ulonglong),
                        sizeof(float),
                        sizeof(double),
						sizeof(longdouble),
                        sizeof(cfloat),
                        sizeof(cdouble),
						sizeof(clongdouble),
                        sizeof(void *),
						0,0,0,0};

typedef void (OneMultAddFunction) (char *, char *, char *);

#define MAKE_ONEMULTADD(fname, type) \
static void fname ## _onemultadd(char *sum, char *term1, char *term2) { \
  (*((type *) sum)) += (*((type *) term1)) * \
  (*((type *) term2)); return; }

MAKE_ONEMULTADD(UBYTE, ubyte)
MAKE_ONEMULTADD(USHORT, ushort)
MAKE_ONEMULTADD(UINT, uint)
MAKE_ONEMULTADD(ULONG, ulong)
MAKE_ONEMULTADD(ULONGLONG, ulonglong)

MAKE_ONEMULTADD(BYTE, byte)
MAKE_ONEMULTADD(SHORT, short)
MAKE_ONEMULTADD(INT, int)
MAKE_ONEMULTADD(LONG, long)
MAKE_ONEMULTADD(LONGLONG, longlong)

MAKE_ONEMULTADD(FLOAT, float)
MAKE_ONEMULTADD(DOUBLE, double)
MAKE_ONEMULTADD(LONGDOUBLE, longdouble)
 
#ifdef __GNUC__
MAKE_ONEMULTADD(CFLOAT, __complex__ float)
MAKE_ONEMULTADD(CDOUBLE, __complex__ double)
MAKE_ONEMULTADD(CLONGDOUBLE, __complex__ long double)
#else
#define MAKE_C_ONEMULTADD(fname, type) \
static void fname ## _onemultadd(char *sum, char *term1, char *term2) { \
  ((type *) sum)[0] += ((type *) term1)[0] * ((type *) term2)[0] \
    - ((type *) term1)[1] * ((type *) term2)[1]; \
  ((type *) sum)[1] += ((type *) term1)[0] * ((type *) term2)[1] \
    + ((type *) term1)[1] * ((type *) term2)[0]; \
  return; }
MAKE_C_ONEMULTADD(CFLOAT, float)
MAKE_C_ONEMULTADD(CDOUBLE, double)
MAKE_C_ONEMULTADD(CLONGDOUBLE, longdouble)
#endif /* __GNUC__ */

static OneMultAddFunction *OneMultAdd[]={NULL,
					 					 BYTE_onemultadd,
					 					 UBYTE_onemultadd,
					 					 SHORT_onemultadd,
                                         USHORT_onemultadd,
					 					 INT_onemultadd,
                                         UINT_onemultadd,
					 					 LONG_onemultadd,
					 					 ULONG_onemultadd,
					 					 LONGLONG_onemultadd,
					 					 ULONGLONG_onemultadd,
					 					 FLOAT_onemultadd,
					 					 DOUBLE_onemultadd,
					 					 LONGDOUBLE_onemultadd,
					 					 CFLOAT_onemultadd,
					 					 CDOUBLE_onemultadd,
					 					 CLONGDOUBLE_onemultadd,
                                         NULL, NULL, NULL, NULL};


/* This could definitely be more optimized... */

int pylab_convolve_2d(PyArrayIterObject* itSignal,
			intp* Signals,
			PyArrayIterObject* itKernel,
			intp* Kerns,
			PyArrayIterObject* itOut,
			intp* Outs,
			int flag,
			char  *fillvalue)
{
  int boundary, outsize, convolve, type_size, type_num;
  int i,j;
  PyArrayNeighborhoodIterObject *curSignal;
  int bounds[4];
  OneMultAddFunction *mult_and_add;
char *sum=NULL, *value=NULL;


  boundary = flag & BOUNDARY_MASK;  /* flag can be fill, reflecting, circular */
  outsize = flag & OUTSIZE_MASK;
  convolve = flag & FLIP_MASK;
  type_num = (flag & TYPE_MASK) >> TYPE_SHIFT;
  

  mult_and_add = OneMultAdd[type_num];
  if (mult_and_add == NULL) return -5;  /* Not available for this type */

  if (type_num < 0 || type_num > MAXTYPES) return -4;  /* Invalid type */
  type_size = elsizes[type_num];

  if ((sum = calloc(type_size,2))==NULL) return -3; /* No memory */
  value = sum + type_size;

 /*need to setup neighbor hood iterator over signal*/
  switch (outsize){

  case (VALID):
	bounds[0] = convolve ? (Kerns[0]-1) : 0;
	bounds[2] = convolve ? (Kerns[1]-1) : 0;
	break;
  case (SAME):
	bounds[0] = convolve ? (((Kerns[0]-1)>>1)) : (((--Kerns[0]-1) >> 1));
	bounds[2] = convolve ? (((Kerns[1]-1)>>1)) : (((--Kerns[1]-1) >> 1));
	break;
  case (FULL):
	bounds[0] = convolve ? 0 : (0-Kerns[0]+1);
	bounds[2] = convolve ? 0 : (0-Kerns[1]+1);
	break;
  default:
	return -1;
}
  bounds[1] = bounds[0];
  bounds[3] = bounds[2];
  if (convolve) {
  	bounds[0] +=Outs[0];
  	bounds[2] +=Outs[1];
 }
  else{
  	bounds[1] += Outs[0];
  	bounds[3] += Outs[1];
 }
  
  curSignal = (PyArrayNeighborhoodIterObject *)PyArray_NeighborhoodIterNew(itSignal, bounds);
  
if (boundary == VALID){
  for (i = 0; i<itOut->size;++i){/*iterate over each element in the output array*/
	memset(sum, 0, type_size);
	PyArray_ITER_NEXT((PyObject *)itOut);
	for (j = 0; i<itKernel->size;++j){
		PyArray_ITER_NEXT((PyObject *)itKernel);
		mult_and_add(sum,(char *)itKernel,(char *)curSignal);
	}

	//memcpy(itOut, sum, type_size);
	PyArray_ITER_RESET((PyObject *)itKernel);
  }
}

return 0;

}


