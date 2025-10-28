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
  PetscErrorCode (*setfromoptions)(Mesh, PetscOptionItems);
  PetscErrorCode (*setup)(Mesh);
  PetscErrorCode (*destroy)(Mesh);
  PetscErrorCode (*view)(Mesh, PetscViewer);
  PetscErrorCode (*createglobalvector)(Mesh, MeshDMType, Vec *);
  PetscErrorCode (*getnumberboundaries)(Mesh, PetscInt *);
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);

  /* Parameters ----------------------------------------------------------- */
  PetscInt dim; /* dimension */

  /* Data ----------------------------------------------------------------- */
  DM    sdm;  /* DM for cell-centered scalar variables */
  DM    vdm;  /* DM for cell-centered vector variables */
  DM    Sdm;  /* DM for face-centered scalar variables */
  DM    Vdm;  /* DM for face-centered vector variables */
  void *data; /* implementation-specific data */

  /* Status --------------------------------------------------------------- */
  PetscBool setupcalled; /* whether MeshSetUp() has been called */
};
