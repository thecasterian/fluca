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

  PetscErrorCode (*viewveccomponents)(Mesh, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], PetscViewer);
  PetscErrorCode (*loadveccomponents)(Mesh, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], PetscViewer);
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

/* Write or read components [c0, c0 + ncomp) of a vector on a DMStag compatible with the Mesh DM, under the CGNS name
   name (suffixed X, Y, Z when ncomp > 1). loc is DMSTAG_ELEMENT for cell data or DMSTAG_LEFT for face data, which
   covers the faces normal to every direction. Exported for Phys; not part of the public API. */
FLUCA_EXTERN PetscErrorCode MeshViewVecComponents_Internal(Mesh, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], PetscViewer);
FLUCA_EXTERN PetscErrorCode MeshLoadVecComponents_Internal(Mesh, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], PetscViewer);

/* MESHCARTESIAN CGNS I/O, defined in impls/cartesian/cartesiancgns.c */
FLUCA_INTERN PetscErrorCode MeshView_Cartesian_CGNS(Mesh, PetscViewer);
FLUCA_INTERN PetscErrorCode MeshLoad_Cartesian_CGNS(Mesh, PetscViewer);
FLUCA_INTERN PetscErrorCode MeshViewVecComponents_Cartesian(Mesh, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], PetscViewer);
FLUCA_INTERN PetscErrorCode MeshLoadVecComponents_Cartesian(Mesh, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], PetscViewer);
