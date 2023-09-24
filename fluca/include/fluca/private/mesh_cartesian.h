#if !defined(FLUCA_MESH_CARTESIAN_H)
#define FLUCA_MESH_CARTESIAN_H

#include <fluca/private/meshimpl.h>

typedef struct {
    PetscInt N[MESH_MAX_DIM];                /* global number of elements */
    PetscInt nRanks[MESH_MAX_DIM];           /* number of processes */
    PetscInt *l[MESH_MAX_DIM];               /* ownership range */
    MeshBoundaryType bndTypes[MESH_MAX_DIM]; /* boundary types */

    PetscMPIInt rank[MESH_MAX_DIM]; /* location in grid of ranks */

    DM dm;  /* DMDA for element-centered variables */
    DM fdm; /* DMStag for face-centered variables */

    PetscReal *c[MESH_MAX_DIM];  /* element coordinates */
    PetscReal *cf[MESH_MAX_DIM]; /* face coordinates */
} Mesh_Cartesian;

#endif
