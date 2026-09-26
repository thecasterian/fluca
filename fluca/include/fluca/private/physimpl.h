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

typedef struct _PhysOps *PhysOps;

struct _PhysOps {
  PetscErrorCode (*setfromoptions)(Phys, PetscOptionItems);
  PetscErrorCode (*setup)(Phys);
  PetscErrorCode (*destroy)(Phys);
  PetscErrorCode (*view)(Phys, PetscViewer);
  PetscErrorCode (*createsolutiondm)(Phys);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters */
  DM               base_dm; /* user-provided DMStag (grid topology + coordinates) */
  PhysBodyForceFn *bodyforce;
  void            *bodyforce_ctx;
  PhysBC           bcs[PHYS_MAX_FACES];
  PetscInt         nprops;
  PhysProperty     props[PHYS_MAX_PROPERTIES];

  /* Data */
  DM       sol_dm; /* solution DMStag */
  PetscInt dim;    /* spatial dimension (extracted from base_dm) */
  void    *data;   /* subtype-specific */

  /* State */
  PetscBool setupcalled;
};

FLUCA_INTERN PetscErrorCode PhysRegisterProperty_Internal(Phys, const char[], PetscScalar);
