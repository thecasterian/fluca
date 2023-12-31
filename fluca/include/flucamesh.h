#if !defined(FLUCAMESH_H)
#define FLUCAMESH_H

#include <flucameshtypes.h>
#include <flucasol.h>
#include <petscdm.h>

FLUCA_EXTERN PetscClassId MESH_CLASSID;

FLUCA_EXTERN PetscErrorCode MeshInitializePackage(void);
FLUCA_EXTERN PetscErrorCode MeshFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode MeshCreate(MPI_Comm, Mesh *);
FLUCA_EXTERN PetscErrorCode MeshSetType(Mesh, MeshType);
FLUCA_EXTERN PetscErrorCode MeshGetType(Mesh, MeshType *);
FLUCA_EXTERN PetscErrorCode MeshSetDim(Mesh, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshGetDim(Mesh, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshSetFromOptions(Mesh);
FLUCA_EXTERN PetscErrorCode MeshSetUp(Mesh);
FLUCA_EXTERN PetscErrorCode MeshView(Mesh, PetscViewer);
FLUCA_EXTERN PetscErrorCode MeshViewFromOptions(Mesh, PetscObject, const char *);
FLUCA_EXTERN PetscErrorCode MeshDestroy(Mesh *);

FLUCA_EXTERN PetscErrorCode MeshGetDM(Mesh, DM *);
FLUCA_EXTERN PetscErrorCode MeshGetFaceDM(Mesh, DM *);
FLUCA_EXTERN PetscErrorCode MeshSetOutputSequenceNumber(Mesh, PetscInt, PetscReal);
FLUCA_EXTERN PetscErrorCode MeshGetOutputSequenceNumber(Mesh, PetscInt *, PetscReal *);

FLUCA_EXTERN PetscFunctionList MeshList;
FLUCA_EXTERN PetscErrorCode MeshRegister(const char *, PetscErrorCode (*)(Mesh));

typedef enum {
    MESHCARTESIAN_PREV,
    MESHCARTESIAN_NEXT,
} MeshCartesianCoordinateStencilLocation;

FLUCA_EXTERN PetscErrorCode MeshCartesianSetSizes(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetNumProcs(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetBoundaryType(Mesh, MeshBoundaryType, MeshBoundaryType, MeshBoundaryType);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetOwnershipRanges(Mesh, const PetscInt *, const PetscInt *, const PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetUniformCoordinates(Mesh, PetscReal, PetscReal, PetscReal, PetscReal,
                                                               PetscReal, PetscReal);
FLUCA_EXTERN PetscErrorCode MeshCartesianGetCoordinateArrays(Mesh, PetscReal ***, PetscReal ***, PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartesianGetCoordinateArraysRead(Mesh, const PetscReal ***, const PetscReal ***,
                                                                 const PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartesianRestoreCoordinateArrays(Mesh, PetscReal ***, PetscReal ***, PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartesianRestoreCoordinateArraysRead(Mesh, const PetscReal ***, const PetscReal ***,
                                                                     const PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartesianGetCoordinateLocationSlot(Mesh, MeshCartesianCoordinateStencilLocation,
                                                                   PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartesianGetInfo(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *,
                                                 PetscInt *, PetscInt *, MeshBoundaryType *, MeshBoundaryType *,
                                                 MeshBoundaryType *);
FLUCA_EXTERN PetscErrorCode MeshCartesianGetCorners(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *,
                                                    PetscInt *);

#endif
