#pragma once

#include <fluca/private/meshimpl.h>
#include <flucameshcart.h>

FLUCA_EXTERN PetscLogEvent MESHCART_CreateFromFile;

typedef struct {
  PetscInt             N[MESH_MAX_DIM];        /* global number of elements */
  PetscInt             nRanks[MESH_MAX_DIM];   /* number of processes */
  PetscInt            *l[MESH_MAX_DIM];        /* ownership range */
  MeshCartBoundaryType bndTypes[MESH_MAX_DIM]; /* boundary types */

  PetscInt refineFactor[MESH_MAX_DIM]; /* refinement factor */

  PetscScalar *coordLoaded[MESH_MAX_DIM]; /* Coordinates loaded from viewer */
} Mesh_Cart;

FLUCA_INTERN PetscErrorCode MeshView_Cart_CGNS(Mesh, PetscViewer);
FLUCA_INTERN PetscErrorCode MeshLoad_Cart_CGNS(Mesh, PetscViewer);

FLUCA_INTERN PetscErrorCode VecView_Cart(Vec, PetscViewer);
FLUCA_INTERN PetscErrorCode VecView_Cart_Local(Vec, PetscViewer);
FLUCA_INTERN PetscErrorCode VecView_Cart_Local_CGNS(Vec, PetscViewer);
