#if !defined(FLUCAMESH_H)
#define FLUCAMESH_H

#include <flucasys.h>

typedef struct _p_Mesh *Mesh;

typedef const char *MeshType;
#define MESHCARTESIAN "cartesian"

FLUCA_EXTERN PetscClassId MESH_CLASSID;

FLUCA_EXTERN PetscErrorCode MeshInitializePackage(void);
FLUCA_EXTERN PetscErrorCode MeshFinalizePackage(void);

typedef enum {
    MESH_BOUNDARY_NOT_PERIODIC,
    MESH_BOUNDARY_PERIODIC,
} MeshBoundaryType;
FLUCA_EXTERN const char *MeshBoundaryTypes[];

FLUCA_EXTERN PetscErrorCode MeshCreate(MPI_Comm, Mesh *);
FLUCA_EXTERN PetscErrorCode MeshSetType(Mesh, MeshType);
FLUCA_EXTERN PetscErrorCode MeshGetType(Mesh, MeshType *);
FLUCA_EXTERN PetscErrorCode MeshSetDim(Mesh, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshGetDim(Mesh, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshSetUp(Mesh);
FLUCA_EXTERN PetscErrorCode MeshView(Mesh, PetscViewer);
FLUCA_EXTERN PetscErrorCode MeshViewFromOptions(Mesh, PetscObject, const char *);
FLUCA_EXTERN PetscErrorCode MeshDestroy(Mesh *);

FLUCA_EXTERN PetscFunctionList MeshList;
FLUCA_EXTERN PetscErrorCode MeshRegister(const char *, PetscErrorCode (*)(Mesh));

FLUCA_EXTERN PetscErrorCode MeshCartesianSetSize(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetNumProcs(Mesh, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetBoundaryType(Mesh, MeshBoundaryType, MeshBoundaryType, MeshBoundaryType);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetOwnershipRanges(Mesh, const PetscInt *, const PetscInt *, const PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshCartesianSetStencilWidth(Mesh, PetscInt);
FLUCA_EXTERN PetscErrorCode MeshCartesianGetInfo(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *,
                                                 PetscInt *, PetscInt *, PetscInt *, MeshBoundaryType *,
                                                 MeshBoundaryType *, MeshBoundaryType *);

#endif
