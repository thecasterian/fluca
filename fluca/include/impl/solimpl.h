#if !defined(FLUCASOLIMPL_H)
#define FLUCASOLIMPL_H

#include <flucamesh.h>
#include <flucasol.h>
#include <impl/flucaimpl.h>
#include <petscvec.h>

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

    /* Parameters ----------------------------------------------------------- */

    /* Data ----------------------------------------------------------------- */
    Mesh mesh;
    Vec u, v, w, p;
    void *data; /* implementation-specific data */

    /* Status --------------------------------------------------------------- */
};

#endif
