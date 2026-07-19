#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaphys.h>

FLUCA_EXTERN PetscBool      PhysRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode PhysRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  PHYS_SetUp;

typedef struct _PhysOps *PhysOps;

struct _PhysOps {
  PetscErrorCode (*setfromoptions)(Phys, PetscOptionItems);
  PetscErrorCode (*setup)(Phys);
  PetscErrorCode (*destroy)(Phys);
  PetscErrorCode (*view)(Phys, PetscViewer);
  PetscErrorCode (*createsolutiondm)(Phys);
  PetscErrorCode (*setupts)(Phys, TS);
  PetscErrorCode (*computeifunction)(Phys, PetscReal, Vec, Vec, Vec);
  PetscErrorCode (*computeijacobian)(Phys, PetscReal, Vec, Vec, PetscReal, Mat, Mat);
  PetscErrorCode (*computerhsfunction)(Phys, PetscReal, Vec, Vec);
  PetscErrorCode (*computerhsjacobian)(Phys, PetscReal, Vec, Mat, Mat);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters */
  DM               base_dm; /* user-provided DMStag (grid topology + coordinates) */
  PhysBodyForceFn *bodyforce;
  void            *bodyforce_ctx;

  /* Data */
  DM       sol_dm; /* solution DMStag (created by subtype during setup) */
  PetscInt dim;    /* spatial dimension (extracted from base_dm) */
  void    *data;   /* subtype-specific */

  /* State */
  PetscBool setupcalled;
};
