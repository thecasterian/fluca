#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaibbody.h>

#define IBBODY_MAX_DIM     3
#define IBBODY_FD_DOT_STEP 1e-5

FLUCA_EXTERN PetscBool      IBBodyRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode IBBodyRegisterAll(void);

typedef struct _IBBodyOps *IBBodyOps;

struct _IBBodyOps {
  PetscErrorCode (*setfromoptions)(IBBody, PetscOptionItems);
  PetscErrorCode (*setup)(IBBody);
  PetscErrorCode (*sample)(IBBody, PetscInt, const PetscInt[], const PetscReal *const[], PetscReal[]);
  PetscErrorCode (*releasegeometry)(IBBody);
  PetscErrorCode (*destroy)(IBBody);
  PetscErrorCode (*view)(IBBody, PetscViewer);
};

struct _p_IBBody {
  PETSCHEADER(struct _IBBodyOps);

  /* Velocity boundary condition ------------------------------------------ */
  PetscScalar       velocity[IBBODY_MAX_DIM]; /* constant velocity, used when vel_fn == NULL */
  IBBodyVelocityFn *vel_fn;
  IBBodyVelocityFn *vel_fn_dot; /* NULL = central difference of vel_fn in time */
  void             *vel_ctx;

  /* Data ----------------------------------------------------------------- */
  void *data; /* implementation-specific data */

  /* Status --------------------------------------------------------------- */
  PetscBool setupcalled;
  PetscBool sampled; /* a Mesh sampled the body; geometry and type are frozen */
};

typedef struct {
  IBBodyLevelSetFn *fn;
  void             *ctx;
} IBBody_LevelSet;

/* Sample phi on the tensor-product points (coord[0][i], coord[1][j], coord[2][k]); n[d] = 1 for d >= dim.
   phi is indexed by i + n[0] * (j + n[1] * k). */
FLUCA_EXTERN PetscErrorCode IBBodySample_Internal(IBBody, PetscInt, const PetscInt[], const PetscReal *const[], PetscReal[]);
FLUCA_EXTERN PetscErrorCode IBBodyReleaseGeometry_Internal(IBBody);
FLUCA_EXTERN PetscErrorCode IBBodyEvaluateVelocity_Internal(IBBody, PetscInt, PetscReal, const PetscReal[], PetscBool, PetscScalar[]);
