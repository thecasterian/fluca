#if !defined(FLUCASOLIMPL_H)
#define FLUCASOLIMPL_H

#include <flucamesh.h>
#include <flucasol.h>
#include <impl/flucaimpl.h>

FLUCA_EXTERN PetscBool SolRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode SolRegisterAll(void);

typedef struct _SolOps *SolOps;

struct _SolOps {
    PetscErrorCode (*setmesh)(Sol, Mesh);
    PetscErrorCode (*destroy)(Sol);
    PetscErrorCode (*view)(Sol, PetscViewer);
};

struct _p_Sol {
    PETSCHEADER(struct _SolOps);

    /* Data ----------------------------------------------------------------- */
    Mesh mesh;
    Vec v[3];
    Vec p;
    void *data; /* implementation-specific data */
};

#endif
