#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaib.h>

FLUCA_EXTERN PetscBool      FlucaIBRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode FlucaIBRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  FLUCAIB_SetUp, FLUCAIB_ApplyBC;

typedef struct _FlucaIBOps *FlucaIBOps;

struct _FlucaIBOps {
  PetscErrorCode (*setfromoptions)(FlucaIB, PetscOptionItems);
  PetscErrorCode (*setup)(FlucaIB);
  PetscErrorCode (*destroy)(FlucaIB);
  PetscErrorCode (*view)(FlucaIB, PetscViewer);
  PetscErrorCode (*applybc)(FlucaIB, Vec, FlucaIBField);
};

struct _p_FlucaIB {
  PETSCHEADER(struct _FlucaIBOps);

  /* Underlying grid (referenced, not owned) */
  DM dm;

  /* State flags */
  PetscBool setupcalled;

  /* Subtype-specific data */
  void *data;
};
