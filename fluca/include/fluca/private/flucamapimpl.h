#if !defined(FLUCAMAPIMPL_H)
#define FLUCAMAPIMPL_H

#include <flucamap.h>
#include <fluca/private/flucaimpl.h>

struct _FlucaMapKV {
    PetscObject key, value;
    PetscInt hash;
    struct _FlucaMapKV *prev, *next;
};

struct _FlucaMapKVList {
    struct _FlucaMapKV *front, *back;
};

struct _p_FlucaMap {
    PETSCHEADER(int);

    struct _FlucaMapKVList *buckets;
    PetscInt size; /* number of elements */
    PetscInt bucketsize;

    PetscErrorCode (*hash)(PetscObject, PetscInt *);             /* hash function of key */
    PetscErrorCode (*eq)(PetscObject, PetscObject, PetscBool *); /* equality function of key */
};

#endif
