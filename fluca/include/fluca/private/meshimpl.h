#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>

#define MESH_MIN_DIM 1
#define MESH_MAX_DIM 3

FLUCA_EXTERN PetscBool      MeshRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode MeshRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  MESH_SetUp;

typedef struct _MeshOps *MeshOps;

struct _MeshOps {
  PetscErrorCode (*setfromoptions)(Mesh, PetscOptionItems);
  PetscErrorCode (*setup)(Mesh); /* validates mesh->dm */
  PetscErrorCode (*destroy)(Mesh);
  PetscErrorCode (*view)(Mesh, PetscViewer);
  PetscErrorCode (*load)(Mesh, PetscViewer); /* replaces mesh->dm */
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);

  /* Data ----------------------------------------------------------------- */
  DM       dm;   /* user-provided or loaded DMStag; referenced */
  PetscInt dim;  /* spatial dimension, from dm */
  void    *data; /* implementation-specific data */

  /* Status --------------------------------------------------------------- */
  PetscBool setupcalled; /* after MeshSetUp() the DM can no longer change */
};

/* MESHCARTESIAN CGNS I/O, defined in impls/cartesian/cartesiancgns.c */
FLUCA_INTERN PetscErrorCode MeshView_Cartesian_CGNS(Mesh, PetscViewer);
FLUCA_INTERN PetscErrorCode MeshLoad_Cartesian_CGNS(Mesh, PetscViewer);
