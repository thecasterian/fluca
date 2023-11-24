#if !defined(FLUCA_PRIVATE_MESH_CARTESIAN_H)
#define FLUCA_PRIVATE_MESH_CARTESIAN_H

#include <fluca/private/meshimpl.h>

typedef struct {
    PetscInt N[MESH_MAX_DIM];                /* global number of elements */
    PetscInt nRanks[MESH_MAX_DIM];           /* number of processes */
    PetscInt *l[MESH_MAX_DIM];               /* ownership range */
    MeshBoundaryType bndTypes[MESH_MAX_DIM]; /* boundary types */

    PetscMPIInt rank[MESH_MAX_DIM]; /* location in grid of ranks */
    MPI_Comm subcomm[MESH_MAX_DIM]; /* communicator for each dimension */

    DM dm;  /* DMDA for element-centered variables */
    DM fdm; /* DMStag for face-centered variables */

    PetscReal *c[MESH_MAX_DIM];      /* element coordinates */
    PetscReal *cf[MESH_MAX_DIM];     /* face coordinates */
    PetscReal *w[MESH_MAX_DIM];      /* element widths */
    PetscReal *rf[MESH_MAX_DIM];     /* interpolation factor at a face */
    PetscReal (*a[MESH_MAX_DIM])[2]; /* second-order derivative coefficients */
} Mesh_Cartesian;

#endif
