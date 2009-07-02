#include <stdlib.h>
#include <string.h>

#define MEMSET memset

typedef char CHARTYPE;
typedef unsigned int gfc_charlen_type ;

void
_gfortran_concat_string (gfc_charlen_type destlen, CHARTYPE * dest,
               gfc_charlen_type len1, const CHARTYPE * s1,
               gfc_charlen_type len2, const CHARTYPE * s2)
{
  if (len1 >= destlen)
    {
      memcpy (dest, s1, destlen * sizeof (CHARTYPE));
      return;
    }
  memcpy (dest, s1, len1 * sizeof (CHARTYPE));
  dest += len1;
  destlen -= len1;

  if (len2 >= destlen)
    {
      memcpy (dest, s2, destlen * sizeof (CHARTYPE));
      return;
    }

  memcpy (dest, s2, len2 * sizeof (CHARTYPE));
  MEMSET (&dest[len2], ' ', destlen - len2);
}

