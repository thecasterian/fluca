#if !defined(FLUCAMESHIMPL_H)
#define FLUCAMESHIMPL_H

#include <flucamesh.h>
#include <impl/flucaimpl.h>
#include <petscdm.h>

#define MESH_MIN_DIM 2
#define MESH_MAX_DIM 3

FLUCA_EXTERN PetscBool MeshRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode MeshRegisterAll(void);
FLUCA_EXTERN PetscLogEvent MESH_SetUp;

typedef struct _MeshOps *MeshOps;

struct _MeshOps {
    PetscErrorCode (*setfromoptions)(Mesh, PetscOptionItems *);
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
    void *ctx; /* user context */
    PetscErrorCode (*ctxdestroy)(void **);

    /* Data ----------------------------------------------------------------- */
    void *data; /* implementation-specific data */
    PetscInt seqnum;
    PetscReal seqval;

    /* Status --------------------------------------------------------------- */
    MeshStateType state;
    PetscBool setupcalled;
};

PetscErrorCode MeshBoundaryTypeToDMBoundaryType(MeshBoundaryType type, DMBoundaryType *dmtype);

#endif
