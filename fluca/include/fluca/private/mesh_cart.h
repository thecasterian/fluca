#pragma once

#include <fluca/private/meshimpl.h>
#include <flucameshcart.h>

FLUCA_EXTERN PetscLogEvent MESHCART_CreateFromFile;

typedef struct {
  PetscInt         N[MESH_MAX_DIM];        /* global number of elements */
  PetscInt         nRanks[MESH_MAX_DIM];   /* number of processes */
  PetscInt        *l[MESH_MAX_DIM];        /* ownership range */
  MeshBoundaryType bndTypes[MESH_MAX_DIM]; /* boundary types */

  PetscInt refineFactor[MESH_MAX_DIM]; /* refinement factor */

  DM dm;  /* DMStag for element-centered variables */
  DM fdm; /* DMStag for face-centered variables */
} Mesh_Cart;
