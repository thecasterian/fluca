#if !defined(FLUCAMESHCART_H)
#define FLUCAMESHCART_H

#include <flucamesh.h>

typedef enum {
    MESHCART_PREV,
    MESHCART_NEXT,
} MeshCartCoordinateStencilLocation;
FLUCA_EXTERN const char *MeshCartCoordinateStencilLocations[];

FLUCA_EXTERN PetscErrorCode MeshCartSetGlobalSizes(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartGetGlobalSizes(Mesh, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartSetNumRanks(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartGetNumRanks(Mesh, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartSetBoundaryTypes(Mesh, MeshBoundaryType, MeshBoundaryType, MeshBoundaryType);
FLUCA_EXTERN PetscErrorCode MeshCartGetBoundaryTypes(Mesh, MeshBoundaryType *, MeshBoundaryType *, MeshBoundaryType *);
FLUCA_EXTERN PetscErrorCode MeshCartSetOwnershipRanges(Mesh, const PetscInt *, const PetscInt *, const PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetOwnershipRanges(Mesh, const PetscInt **, const PetscInt **, const PetscInt **);
FLUCA_EXTERN PetscErrorCode MeshCartSetUniformCoordinates(Mesh, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal,
                                                          PetscReal);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateArrays(Mesh, PetscReal ***, PetscReal ***, PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateArraysRead(Mesh, const PetscReal ***, const PetscReal ***,
                                                            const PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartRestoreCoordinateArrays(Mesh, PetscReal ***, PetscReal ***, PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartRestoreCoordinateArraysRead(Mesh, const PetscReal ***, const PetscReal ***,
                                                                const PetscReal ***);
FLUCA_EXTERN PetscErrorCode MeshCartGetCoordinateLocationSlot(Mesh, MeshCartCoordinateStencilLocation, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetLocalSizes(Mesh, PetscInt *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetCorners(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *,
                                               PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartGetIsFirstRank(Mesh, PetscBool *, PetscBool *, PetscBool *);
FLUCA_EXTERN PetscErrorCode MeshCartGetIsLastRank(Mesh, PetscBool *, PetscBool *, PetscBool *);

#endif
