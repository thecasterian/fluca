#if !defined(FLUCAMESHIMPL_H)
#define FLUCAMESHIMPL_H

#include <flucamesh.h>
#include <impl/flucaimpl.h>
#include <petscdm.h>

FLUCA_EXTERN PetscBool MeshRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode MeshRegisterAll(void);
FLUCA_EXTERN PetscLogEvent MESH_SetUp;

typedef struct _MeshOps *MeshOps;

struct _MeshOps {
    PetscErrorCode (*setup)(Mesh);
    PetscErrorCode (*destroy)(Mesh);
    PetscErrorCode (*view)(Mesh, PetscViewer);
};

typedef enum {
    MESH_STATE_INITIAL,
    MESH_STATE_SETUP,
} MeshStateType;

struct _p_Mesh {
    PETSCHEADER(struct _MeshOps);

    /* Parameters ----------------------------------------------------------- */
    PetscInt dim; /* dimension */

    /* User functions and contexts ------------------------------------------ */
    void *ctx;    /* user context */
    PetscErrorCode (*ctxdestroy)(void **);

    /* Data ----------------------------------------------------------------- */
    void *data;   /* implementation-specific data */

    /* Status --------------------------------------------------------------- */
    MeshStateType state;
    PetscBool setupcalled;
};

#endif
