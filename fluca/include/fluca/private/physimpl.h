#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaphys.h>
#include <petscdmstag.h>

FLUCA_EXTERN PetscBool      PhysRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode PhysRegisterAll(void);

FLUCA_EXTERN PetscLogEvent Phys_SetUp;

typedef struct _PhysOps *PhysOps;

struct _PhysOps {
  PetscErrorCode (*setfromoptions)(Phys, PetscOptionItems);
  PetscErrorCode (*setup)(Phys);
  PetscErrorCode (*destroy)(Phys);
  PetscErrorCode (*view)(Phys, PetscViewer);
  PetscErrorCode (*createsolutiondm)(Phys);
  PetscErrorCode (*setupts)(Phys, TS);
  PetscErrorCode (*setupsnes)(Phys, SNES);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters ----------------------------------------------------------- */
  DM               base_dm;   /* user-provided DMStag (grid topology + coordinates) */
  PhysBodyForceFn *bodyforce; /* body force callback */
  void            *bodyforce_ctx;

  /* Data ----------------------------------------------------------------- */
  DM       sol_dm; /* solution DMStag (created by subtype during setup) */
  PetscInt dim;    /* spatial dimension (extracted from base_dm) */
  void    *data;

  /* State ---------------------------------------------------------------- */
  PetscBool setupcalled;
};
