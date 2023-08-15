#if !defined(FLUCA_MESH_CARTESIAN_H)
#define FLUCA_MESH_CARTESIAN_H

#include <impl/meshimpl.h>

typedef struct {
    MeshBoundaryType bndx, bndy, bndz; /* boundary types */

    PetscInt M, N, P;       /* global number of elements */
    PetscInt m, n, p;       /* number of processes */
    PetscInt *lx, *ly, *lz; /* ownership range */
    PetscInt s;             /* stencil width */

    DM dm;  /* DMDA for element-centered variables */
    DM vdm; /* DMStag for velocities at face */

    DM cxdm, cydm, czdm; /* 1d DMStag for coordinates */
    Vec cx, cy, cz;      /* coordinate and its derivatives vectors */
} Mesh_Cartesian;

#endif
