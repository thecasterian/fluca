#pragma once

#include <flucasys.h>
#include <petscdmstag.h>
#include <flucaibbody.h>

/* Mesh - The computational grid: a user-provided DMStag plus what the solver needs to know about it */
typedef struct _p_Mesh *Mesh;

typedef const char *MeshType;
#define MESHCARTESIAN "cartesian" /* Rectilinear grid given by the DMStag product coordinates; immersed bodies via MeshAddBody */

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
/* Reads the grid and, when the file has them, the immersed bodies: one IBBODYPRESAMPLED body per body id, with zero
   velocity (set it with MeshGetBody() and IBBodySetVelocity()). Bodies added after MeshLoad() follow the loaded ones.
   A loaded mesh writes its bodies with MeshView() only after MeshSetUp(). */
FLUCA_EXTERN PetscErrorCode MeshLoad(Mesh, PetscViewer);
FLUCA_EXTERN PetscErrorCode MeshDestroy(Mesh *);

/* MESHCARTESIAN */
FLUCA_EXTERN PetscErrorCode MeshCartesianCreate(DM, Mesh *);

/* Immersed boundary */
typedef enum {
  MESH_IB_FLUID,
  MESH_IB_FORCING, /* not solid, with a solid star neighbour; carries a forcing record */
  MESH_IB_SOLID,
} MeshIBPointType;
FLUCA_EXTERN const char *MeshIBPointTypes[];

/* Ghost values (phi and body id) that have an owner, periodic images included, are taken from their owners, so owned
   values must already be right: a body crossing a periodic boundary needs a level set that is periodic on the owned grid */
FLUCA_EXTERN PetscErrorCode MeshAddBody(Mesh, IBBody);
FLUCA_EXTERN PetscErrorCode MeshGetNumBodies(Mesh, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshGetBody(Mesh, PetscInt, IBBody *);
FLUCA_EXTERN PetscErrorCode MeshGetIBLevelSet(Mesh, Vec *);
/* Exact for owned points. At ghost points whose star neighbour lies outside the local ghosted range, that neighbour is not examined, so a FORCING point may be reported as FLUID. */
FLUCA_EXTERN PetscErrorCode MeshGetIBPointType(Mesh, DMStagStencilLocation, PetscInt, PetscInt, PetscInt, MeshIBPointType *);

FLUCA_EXTERN PetscFunctionList MeshList;
FLUCA_EXTERN PetscErrorCode    MeshRegister(const char[], PetscErrorCode (*)(Mesh));
