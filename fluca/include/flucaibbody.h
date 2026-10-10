#pragma once

#include <flucasys.h>
#include <petscviewer.h>

/* IBBody - A rigid body immersed in the flow, described by a level set (phi < 0 inside the solid) */
typedef struct _p_IBBody *IBBody;

typedef const char *IBBodyType;
#define IBBODYLEVELSET "levelset" /* phi given by a user callback */
#define IBBODYSURFACE  "surface"  /* phi computed from a closed polyline (2D) or triangle surface (3D) */

typedef PetscErrorCode IBBodyLevelSetFn(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi);
typedef PetscErrorCode IBBodyVelocityFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar u[]);

FLUCA_EXTERN PetscClassId IBBODY_CLASSID;

FLUCA_EXTERN PetscErrorCode IBBodyInitializePackage(void);
FLUCA_EXTERN PetscErrorCode IBBodyFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode IBBodyCreate(MPI_Comm, IBBody *);
FLUCA_EXTERN PetscErrorCode IBBodySetType(IBBody, IBBodyType);
FLUCA_EXTERN PetscErrorCode IBBodyGetType(IBBody, IBBodyType *);
FLUCA_EXTERN PetscErrorCode IBBodySetFromOptions(IBBody);
FLUCA_EXTERN PetscErrorCode IBBodySetUp(IBBody);
FLUCA_EXTERN PetscErrorCode IBBodyView(IBBody, PetscViewer);
FLUCA_EXTERN PetscErrorCode IBBodyDestroy(IBBody *);

FLUCA_EXTERN PetscErrorCode IBBodySetVelocity(IBBody, PetscInt, const PetscScalar[]);
FLUCA_EXTERN PetscErrorCode IBBodySetVelocityFunction(IBBody, IBBodyVelocityFn *, IBBodyVelocityFn *, void *);

/* IBBODYLEVELSET */
FLUCA_EXTERN PetscErrorCode IBBodyLevelSetSetFunction(IBBody, IBBodyLevelSetFn *, void *);

FLUCA_EXTERN PetscFunctionList IBBodyList;
FLUCA_EXTERN PetscErrorCode    IBBodyRegister(const char[], PetscErrorCode (*)(IBBody));
