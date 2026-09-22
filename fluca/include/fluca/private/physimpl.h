#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaphys.h>

FLUCA_EXTERN PetscBool      PhysRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode PhysRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  PHYS_SetUp;

#define PHYS_MAX_FIELDS 8

typedef struct {
  char             *name;
  PhysFieldLocation loc;
  PetscInt          ncomp; /* components per point (per face for PHYS_FIELD_FACE) */
  PetscInt          c0;    /* first component within its location */
  PhysEquationRole  role;
  PetscBool         nullspace_const; /* field is determined only up to a constant */
} PhysField;

#define PHYS_MAX_PROPERTIES 8

typedef struct {
  char              *name;
  PhysFieldLocation  loc;
  PhysPropertySource source;
  PetscScalar        constant; /* valid when source is PHYS_PROPERTY_CONSTANT */
} PhysProperty;

typedef struct _PhysOps *PhysOps;

struct _PhysOps {
  PetscErrorCode (*setfromoptions)(Phys, PetscOptionItems);
  PetscErrorCode (*registerfields)(Phys);
  PetscErrorCode (*setup)(Phys);
  PetscErrorCode (*destroy)(Phys);
  PetscErrorCode (*view)(Phys, PetscViewer);
  PetscErrorCode (*computemomentumsystem)(Phys, PetscReal, PetscReal, Vec, Mat, Vec);
  PetscErrorCode (*computecouplingsystem)(Phys, PetscReal, PetscReal, Mat, Vec);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters */
  DM               base_dm; /* user-provided DMStag (grid topology + coordinates) */
  PhysBodyForceFn *bodyforce;
  void            *bodyforce_ctx;

  /* Data */
  DM           sol_dm;                  /* solution DMStag (created during SegSetUp(), once every field is declared) */
  PetscInt     dim;                     /* spatial dimension (extracted from base_dm) */
  void        *data;                    /* subtype-specific */
  PetscInt     nfields;                 /* registered solution fields */
  PhysField    fields[PHYS_MAX_FIELDS]; /* in registration order */
  PetscInt     nprops;
  PhysProperty props[PHYS_MAX_PROPERTIES];

  /* State */
  PetscBool setupcalled;
};

/* Seg calls these two from its own library to complete the two-phase setup, so they need public
   visibility even though they are internal API. */
FLUCA_EXTERN PetscErrorCode PhysDeclareField_Internal(Phys, const char[], PhysFieldLocation, PetscInt, PhysEquationRole);
FLUCA_EXTERN PetscErrorCode PhysCreateSolutionDM_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysDeclareConstantNullSpace_Internal(Phys, const char[]);
FLUCA_INTERN PetscErrorCode PhysGetField_Internal(Phys, const char[], PhysFieldLocation *, PetscInt *, PetscInt *);
FLUCA_INTERN PetscErrorCode PhysGetFieldIS_Internal(Phys, const char[], IS *);
FLUCA_INTERN PetscErrorCode PhysRegisterProperty_Internal(Phys, const char[], PhysFieldLocation, PhysPropertySource);
FLUCA_INTERN PetscErrorCode PhysSetPropertyConstant_Internal(Phys, const char[], PetscScalar);
