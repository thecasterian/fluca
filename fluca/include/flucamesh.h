#pragma once

#include <flucasys.h>
#include <petscdm.h>

typedef struct _p_Mesh *Mesh;

typedef const char *MeshType;
#define MESHCART "cart"

FLUCA_EXTERN PetscClassId MESH_CLASSID;

FLUCA_EXTERN PetscErrorCode MeshInitializePackage(void);
FLUCA_EXTERN PetscErrorCode MeshFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode MeshCreate(MPI_Comm, Mesh *);
FLUCA_EXTERN PetscErrorCode MeshSetType(Mesh, MeshType);
FLUCA_EXTERN PetscErrorCode MeshGetType(Mesh, MeshType *);
FLUCA_EXTERN PetscErrorCode MeshSetDimension(Mesh, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshGetDimension(Mesh, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshSetFromOptions(Mesh);
FLUCA_EXTERN PetscErrorCode MeshSetUp(Mesh);
FLUCA_EXTERN PetscErrorCode MeshView(Mesh, PetscViewer);
FLUCA_EXTERN PetscErrorCode MeshViewFromOptions(Mesh, PetscObject, const char[]);
FLUCA_EXTERN PetscErrorCode MeshDestroy(Mesh *);

FLUCA_EXTERN PetscErrorCode MeshGetDM(Mesh, DM *);
FLUCA_EXTERN PetscErrorCode MeshGetFaceDM(Mesh, DM *);
FLUCA_EXTERN PetscErrorCode MeshGetNumberBoundaries(Mesh, PetscInt *);

FLUCA_EXTERN PetscFunctionList MeshList;
FLUCA_EXTERN PetscErrorCode    MeshRegister(const char[], PetscErrorCode (*)(Mesh));
