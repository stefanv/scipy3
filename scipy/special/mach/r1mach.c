 /* C source for R1MACH -- remove the * in column 1 */
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "common.h"

float F_FUNC(r1mach,R1MACH)(long *i)
{
	switch(*i){
	  case 1: return FLT_MIN;
	  case 2: return FLT_MAX;
	  case 3: return FLT_EPSILON/FLT_RADIX;
	  case 4: return FLT_EPSILON;
	  case 5: return log10(FLT_RADIX);
	  }
	fprintf(stderr, "invalid argument: r1mach(%ld)\n", *i);
	exit(1); return 0; /* else complaint of missing return value */
}
