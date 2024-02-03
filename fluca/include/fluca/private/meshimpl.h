#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>
#include <petscdm.h>

#define MESH_MIN_DIM 2
#define MESH_MAX_DIM 3

FLUCA_EXTERN PetscBool      MeshRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode MeshRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  MESH_SetUp;

typedef struct _MeshOps *MeshOps;

struct _MeshOps {
  PetscErrorCode (*setfromoptions)(Mesh, PetscOptionItems *);
  PetscErrorCode (*setup)(Mesh);
  PetscErrorCode (*destroy)(Mesh);
  PetscErrorCode (*getdm)(Mesh, DM *);
  PetscErrorCode (*getfacedm)(Mesh, DM *);
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

  /* Data ----------------------------------------------------------------- */
  void *data; /* implementation-specific data */

  /* Status --------------------------------------------------------------- */
  MeshStateType state;
};

PetscErrorCode MeshBoundaryTypeToDMBoundaryType(MeshBoundaryType type, DMBoundaryType *dmtype);
