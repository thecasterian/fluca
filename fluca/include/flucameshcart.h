#pragma once

#include <flucamesh.h>

typedef enum {
  MESHCART_PREV,
  MESHCART_NEXT,
} MeshCartCoordinateStencilLocation;
FLUCA_EXTERN const char *MeshCartCoordinateStencilLocations[];

FLUCA_EXTERN PetscErrorCode MeshCartCreate2d(MPI_Comm, MeshBoundaryType, MeshBoundaryType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt *, const PetscInt *, Mesh *);
FLUCA_EXTERN PetscErrorCode MeshCartCreate3d(MPI_Comm, MeshBoundaryType, MeshBoundaryType, MeshBoundaryType, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt *, const PetscInt *, const PetscInt *, Mesh *);

FLUCA_EXTERN PetscErrorCode MeshCartSetGlobalSizes(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartGetGlobalSizes(Mesh, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartSetNumRanks(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartGetNumRanks(Mesh, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartSetBoundaryTypes(Mesh, MeshBoundaryType, MeshBoundaryType, MeshBoundaryType);
FLUCA_EXTERN PetscErrorCode MeshCartGetBoundaryTypes(Mesh, MeshBoundaryType *, MeshBoundaryType *, MeshBoundaryType *);
FLUCA_EXTERN PetscErrorCode MeshCartSetOwnershipRanges(Mesh, const PetscInt[], const PetscInt[], const PetscInt[]);
FLUCA_EXTERN PetscErrorCode MeshCartGetOwnershipRanges(Mesh, const PetscInt *[], const PetscInt *[], const PetscInt *[]);
FLUCA_EXTERN PetscErrorCode MeshCartSetRefinementFactor(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartGetRefinementFactor(Mesh, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartSetUniformCoordinates(Mesh, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateArrays(Mesh, PetscScalar ***, PetscScalar ***, PetscScalar ***);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateArraysRead(Mesh, const PetscScalar ***, const PetscScalar ***, const PetscScalar ***);
FLUCA_EXTERN PetscErrorCode MeshCartRestoreCoordinateArrays(Mesh, PetscScalar ***, PetscScalar ***, PetscScalar ***);
FLUCA_EXTERN PetscErrorCode MeshCartRestoreCoordinateArraysRead(Mesh, const PetscScalar ***, const PetscScalar ***, const PetscScalar ***);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateLocationSlot(Mesh, MeshCartCoordinateStencilLocation, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetLocalSizes(Mesh, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetCorners(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetIsFirstRank(Mesh, PetscBool *, PetscBool *, PetscBool *);
FLUCA_EXTERN PetscErrorCode MeshCartGetIsLastRank(Mesh, PetscBool *, PetscBool *, PetscBool *);

FLUCA_EXTERN PetscErrorCode MeshCartCreateFromFile(MPI_Comm, const char[], const char[], Mesh *);
FLUCA_EXTERN PetscErrorCode MeshCartCreateCGNS(MPI_Comm, PetscInt, Mesh *);
FLUCA_EXTERN PetscErrorCode MeshCartCreateCGNSFromFile(MPI_Comm, const char[], Mesh *);
