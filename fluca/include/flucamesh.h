#pragma once

#include <flucasys.h>
#include <petscdmstag.h>

/* Mesh - The computational grid: a user-provided DMStag plus what the solver needs to know about it */
typedef struct _p_Mesh *Mesh;

typedef const char *MeshType;
#define MESHCARTESIAN "cartesian" /* Rectilinear grid given by the DMStag product coordinates, no immersed boundary */

FLUCA_EXTERN PetscClassId MESH_CLASSID;

FLUCA_EXTERN PetscErrorCode MeshInitializePackage(void);
FLUCA_EXTERN PetscErrorCode MeshFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode MeshCreate(MPI_Comm, Mesh *);
FLUCA_EXTERN PetscErrorCode MeshSetType(Mesh, MeshType);
FLUCA_EXTERN PetscErrorCode MeshGetType(Mesh, MeshType *);
FLUCA_EXTERN PetscErrorCode MeshSetDM(Mesh, DM);
FLUCA_EXTERN PetscErrorCode MeshGetDM(Mesh, DM *);
FLUCA_EXTERN PetscErrorCode MeshGetDimension(Mesh, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshSetFromOptions(Mesh);
FLUCA_EXTERN PetscErrorCode MeshSetUp(Mesh);
FLUCA_EXTERN PetscErrorCode MeshView(Mesh, PetscViewer);
FLUCA_EXTERN PetscErrorCode MeshViewFromOptions(Mesh, PetscObject, const char[]);
FLUCA_EXTERN PetscErrorCode MeshLoad(Mesh, PetscViewer);
FLUCA_EXTERN PetscErrorCode MeshDestroy(Mesh *);

/* MESHCARTESIAN */
FLUCA_EXTERN PetscErrorCode MeshCartesianCreate(DM, Mesh *);

FLUCA_EXTERN PetscFunctionList MeshList;
FLUCA_EXTERN PetscErrorCode    MeshRegister(const char[], PetscErrorCode (*)(Mesh));
