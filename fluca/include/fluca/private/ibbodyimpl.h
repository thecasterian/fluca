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

typedef struct {
  PetscInt   dim;   /* 2: closed polyline, 3: closed triangle surface, 0: not set */
  PetscInt   nv;    /* number of vertices */
  PetscInt   nprim; /* number of segments (2D) or triangles (3D) */
  PetscReal *v;     /* vertex coordinates, nv * dim */
  PetscInt  *prim;  /* vertex indices, nprim * dim */
} IBBody_Surface;

/* STL reader, defined in impls/surface/surfacestl.c; *x holds 9 coordinates per triangle.
   On a file error *nt is one of these negative codes and *x is NULL. */
#define IBBODY_STL_ERR_OPEN      -1 /* missing or unreadable */
#define IBBODY_STL_ERR_ENDIAN    -2 /* binary STL on a big-endian host */
#define IBBODY_STL_ERR_TRUNCATED -3 /* binary STL shorter than its triangle count */
#define IBBODY_STL_ERR_MALFORMED -4 /* ASCII STL with a bad vertex line or a vertex count that is not a positive multiple of 3 */
FLUCA_INTERN PetscErrorCode IBBodySurfaceReadSTL_Internal(const char[], PetscInt *, PetscReal **);

FLUCA_EXTERN PetscErrorCode IBBodySample_Internal(IBBody, PetscInt, const PetscInt[], const PetscReal *const[], PetscReal[]);
FLUCA_EXTERN PetscErrorCode IBBodyReleaseGeometry_Internal(IBBody);
FLUCA_EXTERN PetscErrorCode IBBodyEvaluateVelocity_Internal(IBBody, PetscInt, PetscReal, const PetscReal[], PetscBool, PetscScalar[]);
