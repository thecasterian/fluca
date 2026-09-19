#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaphys.h>

FLUCA_EXTERN PetscBool      PhysRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode PhysRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  PHYS_SetUp;

FLUCA_INTERN PetscErrorCode PCCreate_ABF(PC);

#define PHYS_MAX_FIELDS 8

typedef struct {
  char             *name;
  PhysFieldLocation loc;
  PetscInt          ncomp; /* components per point (per face for PHYS_FIELD_FACE) */
  PetscInt          c0;    /* first component within its location */
} PhysField;

typedef struct _PhysOps *PhysOps;

struct _PhysOps {
  PetscErrorCode (*setfromoptions)(Phys, PetscOptionItems);
  PetscErrorCode (*registerfields)(Phys);
  PetscErrorCode (*setup)(Phys);
  PetscErrorCode (*destroy)(Phys);
  PetscErrorCode (*view)(Phys, PetscViewer);
  PetscErrorCode (*getdensity)(Phys, PetscReal *);
  PetscErrorCode (*getviscosity)(Phys, PetscReal *);
  PetscErrorCode (*computemomentumsystem)(Phys, PetscReal, PetscReal, Vec, Mat, Vec);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters */
  DM               base_dm; /* user-provided DMStag (grid topology + coordinates) */
  PhysBodyForceFn *bodyforce;
  void            *bodyforce_ctx;

  /* Data */
  DM        sol_dm;                  /* solution DMStag (created by subtype during setup) */
  PetscInt  dim;                     /* spatial dimension (extracted from base_dm) */
  void     *data;                    /* subtype-specific */
  PetscInt  nfields;                 /* registered solution fields */
  PhysField fields[PHYS_MAX_FIELDS]; /* in registration order */

  /* State */
  PetscBool setupcalled;
};

FLUCA_INTERN PetscErrorCode PhysRegisterField_Internal(Phys, const char[], PhysFieldLocation, PetscInt);
FLUCA_INTERN PetscErrorCode PhysGetField_Internal(Phys, const char[], PhysFieldLocation *, PetscInt *, PetscInt *);
FLUCA_INTERN PetscErrorCode PhysGetFieldIS_Internal(Phys, const char[], IS *);
