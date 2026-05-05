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
  PetscErrorCode (*setupseg)(Phys, Seg);
  PetscErrorCode (*computeifunction)(Phys, PetscReal, Vec, Vec, Vec);
  PetscErrorCode (*computeijacobian)(Phys, PetscReal, Vec, Vec, PetscReal, Mat, Mat);
  PetscErrorCode (*computerhsfunction)(Phys, PetscReal, Vec, Vec);
  PetscErrorCode (*computerhsjacobian)(Phys, PetscReal, Vec, Mat, Mat);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters */
  FlucaIB          ib; /* user-provided IB wrapping the base DMStag (referenced) */
  PhysBodyForceFn *bodyforce;
  void            *bodyforce_ctx;

  /* Data */
  DM       dm;     /* base DMStag, borrowed from ib (cached for fast access) */
  DM       sol_dm; /* solution DMStag (created by subtype during setup) */
  FlucaIB  sol_ib; /* IB wrapping sol_dm (created during setup, owned) */
  PetscInt dim;    /* spatial dimension (extracted from dm) */
  void    *data;   /* subtype-specific */

  /* State */
  PetscBool setupcalled;
};
