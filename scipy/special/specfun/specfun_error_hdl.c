#include <stdio.h>
#include <string.h>

#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif

/*
 * Example to call this in fortran: 
 *
 * CALL SPERHD("GAMMA", "INVALID ENTRY")
 *
 * XXX: this is dumb, and should be replaced by something better (same as Pauli
 * error handler)
 */
int F_FUNC(sperhd, SPERHD)(char* fname, char *msg, 
                int fname_n, int msg_n)
{
        char tmp[256], tmp2[256];

        if (fname_n > 255) {
                goto er;
        }
        if (msg_n > 255) {
                goto er;
        }

        memcpy(tmp, fname, fname_n);
        tmp[fname_n] = '\0';

        memcpy(tmp2, msg, msg_n);
        tmp2[msg_n] = '\0';

        printf("** ERROR %s: %s\n", tmp, tmp2);

        return 0;

er:
        printf("** Too long errror in sperhd**");
        return -1;
}
