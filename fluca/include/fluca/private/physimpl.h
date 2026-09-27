#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaphys.h>

FLUCA_EXTERN PetscBool      PhysRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode PhysRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  PHYS_SetUp;

#define PHYS_MAX_PROPERTIES 8

typedef struct {
  char       *name;
  PetscScalar value;
} PhysProperty;

#define PHYS_MAX_FIELDS 8

typedef struct {
  char             *name;
  PhysFieldLocation loc;
  PetscInt          ncomp; /* as declared: a positive count, or PETSC_DECIDE meaning one component per spatial dimension */
  IS                is;    /* created on first PhysGetFieldIS(); callers borrow it and must not destroy it */
} PhysField;

typedef struct _PhysOps *PhysOps;

struct _PhysOps {
  PetscErrorCode (*setfromoptions)(Phys, PetscOptionItems);
  PetscErrorCode (*setup)(Phys); /* optional subtype setup hook; runs before the solution DM is built */
  PetscErrorCode (*destroy)(Phys);
  PetscErrorCode (*view)(Phys, PetscViewer);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters */
  Mesh             mesh; /* problem domain; referenced */
  PhysBodyForceFn *bodyforce;
  void            *bodyforce_ctx;
  PhysBC           bcs[PHYS_MAX_FACES];
  PetscInt         nprops;
  PhysProperty     props[PHYS_MAX_PROPERTIES];

  /* Data */
  DM        sol_dm; /* solution DMStag */
  PetscInt  dim;    /* spatial dimension (from the mesh) */
  void     *data;   /* subtype-specific */
  PetscInt  nfields;
  PhysField fields[PHYS_MAX_FIELDS]; /* in declaration order */

  /* State */
  PetscBool setupcalled;
};

FLUCA_INTERN PetscErrorCode PhysRegisterProperty_Internal(Phys, const char[], PetscScalar);
FLUCA_INTERN PetscErrorCode PhysCreateSolutionDM_Internal(Phys);
