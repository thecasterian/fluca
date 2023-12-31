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
} MeshCartCoordinateStencilLocation;
FLUCA_EXTERN const char *MeshCartCoordinateStencilLocations[];

FLUCA_EXTERN PetscErrorCode MeshCartSetSizes(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartSetNumProcs(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartSetBoundaryType(Mesh, MeshBoundaryType, MeshBoundaryType, MeshBoundaryType);
FLUCA_EXTERN PetscErrorCode MeshCartSetOwnershipRanges(Mesh, const PetscInt *, const PetscInt *, const PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartSetUniformCoordinates(Mesh, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal,
                                                          PetscReal);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateArrays(Mesh, PetscReal ***, PetscReal ***, PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateArraysRead(Mesh, const PetscReal ***, const PetscReal ***,
                                                            const PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartRestoreCoordinateArrays(Mesh, PetscReal ***, PetscReal ***, PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartRestoreCoordinateArraysRead(Mesh, const PetscReal ***, const PetscReal ***,
                                                                const PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateLocationSlot(Mesh, MeshCartCoordinateStencilLocation, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetInfo(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *,
                                            PetscInt *, PetscInt *, MeshBoundaryType *, MeshBoundaryType *,
                                            MeshBoundaryType *);
FLUCA_EXTERN PetscErrorCode MeshCartGetCorners(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *,
                                               PetscInt *);

#endif
