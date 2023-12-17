#if !defined(FLUCA_PRIVATE_MESH_CARTESIAN_H)
#define FLUCA_PRIVATE_MESH_CARTESIAN_H

#include <fluca/private/meshimpl.h>

typedef struct {
    PetscInt N[MESH_MAX_DIM];                /* global number of elements */
    PetscInt nRanks[MESH_MAX_DIM];           /* number of processes */
    PetscInt *l[MESH_MAX_DIM];               /* ownership range */
    MeshBoundaryType bndTypes[MESH_MAX_DIM]; /* boundary types */

    DM dm;                /* DMStag for element-centered variables */
    DM fdm;               /* DMStag for face-centered variables */
    DM bdm[MESH_MAX_DIM]; /* DMStag for element-centered variables at boundaries normal to each direction */
} Mesh_Cartesian;

#endif
