#if !defined(FLUCA_PRIVATE_MESH_CART_H)
#define FLUCA_PRIVATE_MESH_CART_H

#include <fluca/private/meshimpl.h>
#include <flucameshcart.h>

typedef struct {
    PetscInt N[MESH_MAX_DIM];                /* global number of elements */
    PetscInt nRanks[MESH_MAX_DIM];           /* number of processes */
    PetscInt *l[MESH_MAX_DIM];               /* ownership range */
    MeshBoundaryType bndTypes[MESH_MAX_DIM]; /* boundary types */

    DM dm;  /* DMStag for element-centered variables */
    DM fdm; /* DMStag for face-centered variables */
} Mesh_Cart;

#endif
