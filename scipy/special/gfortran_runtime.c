#include <Python.h>
#include <numpy/npy_math.h>

typedef npy_cfloat GFC_COMPLEX_4;
typedef npy_cdouble GFC_COMPLEX_8;
typedef int GFC_INTEGER_4;
typedef unsigned int GFC_UINTEGER_4;

extern void __chkstk(void);

GFC_COMPLEX_8 clog(GFC_COMPLEX_8 x)
{
	return npy_clog(x);
}

GFC_COMPLEX_8 cpow(GFC_COMPLEX_8 x, GFC_COMPLEX_8 y)
{
	return npy_cpow(x, y);
}

GFC_COMPLEX_8 csin(GFC_COMPLEX_8 x)
{
	return npy_csin(x);
}

GFC_COMPLEX_8 ccos(GFC_COMPLEX_8 x)
{
	return npy_ccos(x);
}

GFC_COMPLEX_8 cexp(GFC_COMPLEX_8 x)
{
	return npy_cexp(x);
}

GFC_COMPLEX_8 csqrt(GFC_COMPLEX_8 x)
{
	return npy_csqrt(x);
}

float __powidf2(float x, int m)
{
  unsigned int n = m < 0 ? -m : m;
  float y = n % 2 ? x : 1;
  float ret;
  while (n >>= 1)
    {
      x = x * x;
      if (n % 2)
	y = y * x;
    }
  ret = m < 0 ? 1/y : y;
fprintf(stderr, "%s: %f, %d -> %f\n", __FUNCTION__, x, m, ret);
  return ret;
}


/* XXX: Nonsense */
long int lround(double x)
{
	fprintf(stderr, "%s\n", __FUNCTION__);
	return (long int)x;
}

void _gfortran_runtime_error(const char* msg, ...)
{
	fprintf(stderr, "%s\n", __FUNCTION__);
	printf("Fortran runtime error\n");
}

void _gfortran_runtime_error_at(const char* where, 
		const char* msg, ...)
{
	fprintf(stderr, "%s\n", __FUNCTION__);
	printf("Fortran runtime error\n");
}

void _gfortran_stop_numeric(unsigned int code)
{
	printf("Fortran stop numeric\n");
	exit(code);
}

void ___chkstk()
{
	fprintf(stderr, "%s\n", __FUNCTION__);
	__chkstk();
}

float
__powisf2 (float x, int m)
{
  unsigned int n = m < 0 ? -m : m;
  float y = n % 2 ? x : 1;
	fprintf(stderr, "%s\n", __FUNCTION__);
  while (n >>= 1)
    {
      x = x * x;
      if (n % 2)
	y = y * x;
    }
  return m < 0 ? 1/y : y;
}

double trunc(double x)
{
	fprintf(stderr, "%s\n", __FUNCTION__);
	return x;
}

GFC_COMPLEX_8
_gfortran_pow_c8_i4 (GFC_COMPLEX_8 a, GFC_INTEGER_4 b)
{
  GFC_COMPLEX_8 pow, x;
  GFC_INTEGER_4 n;
  GFC_UINTEGER_4 u;
  
	fprintf(stderr, "%s\n", __FUNCTION__);
	return npy_cpow(a, npy_cpack(b, 0));
 #if 0
  n = b;
  x = a;
  pow = 1;
  if (n != 0)
    {
      if (n < 0)
	{

	  u = -n;
	  x = pow / x;
	}
      else
	{
	   u = n;
	}
      for (;;)
	{
	  if (u & 1)
	    pow *= x;
	  u >>= 1;
	  if (u)
	    x *= x;
	  else
	    break;
	}
    }
  return pow;
#endif
}

GFC_INTEGER_4
_gfortran_pow_i4_i4 (GFC_INTEGER_4 a, GFC_INTEGER_4 b)
{
  GFC_INTEGER_4 pow, x;
  GFC_INTEGER_4 n;
  GFC_UINTEGER_4 u;
  
	//fprintf(stderr, "%s: %d, %d", __FUNCTION__, a, b);
  n = b;
  x = a;
  pow = 1;
  if (n != 0)
    {
      if (n < 0)
	{
	  if (x == 1)
	    return 1;
	  if (x == -1)
	    return (n & 1) ? -1 : 1;
	  return (x == 0) ? 1 / x : 0;
	}
      else
	{
	   u = n;
	}
      for (;;)
	{
	  if (u & 1)
	    pow *= x;
	  u >>= 1;
	  if (u)
	    x *= x;
	  else
	    break;
	}
    }
	//fprintf(stderr, "... %d", pow);
  return pow;
}
