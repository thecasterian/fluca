#pragma once

#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

/* Phys - Physical Model */
typedef struct _p_Phys *Phys;

/* Phys types */
typedef const char *PhysType;
#define PHYSINS "ins"

/* Body force callback */
typedef PetscErrorCode PhysBodyForceFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx);

/* INS boundary condition types */
typedef enum {
  PHYS_INS_BC_NONE,
  PHYS_INS_BC_VELOCITY,
} PhysINSBCType;

/* INS boundary condition callback: returns value of field component at boundary coordinates */
typedef PetscErrorCode PhysINSBCFn(PetscInt dim, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx);

typedef struct {
  PhysINSBCType type;
  PhysINSBCFn  *fn;
  void         *ctx;
} PhysINSBC;

FLUCA_EXTERN PetscClassId   PHYS_CLASSID;
FLUCA_EXTERN PetscErrorCode PhysInitializePackage(void);
FLUCA_EXTERN PetscErrorCode PhysFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode PhysCreate(MPI_Comm, Phys *);
FLUCA_EXTERN PetscErrorCode PhysSetType(Phys, PhysType);
FLUCA_EXTERN PetscErrorCode PhysGetType(Phys, PhysType *);
FLUCA_EXTERN PetscErrorCode PhysSetUp(Phys);
FLUCA_EXTERN PetscErrorCode PhysDestroy(Phys *);
FLUCA_EXTERN PetscErrorCode PhysView(Phys, PetscViewer);
FLUCA_EXTERN PetscErrorCode PhysViewFromOptions(Phys, PetscObject, const char[]);

FLUCA_EXTERN PetscErrorCode PhysSetBaseDM(Phys, DM);
FLUCA_EXTERN PetscErrorCode PhysGetBaseDM(Phys, DM *);
FLUCA_EXTERN PetscErrorCode PhysGetSolutionDM(Phys, DM *);
FLUCA_EXTERN PetscErrorCode PhysSetBodyForce(Phys, PhysBodyForceFn *, void *);

FLUCA_EXTERN PetscErrorCode PhysSetUpTS(Phys, TS);

FLUCA_EXTERN PetscErrorCode PhysComputeIFunction(Phys, PetscReal, Vec, Vec, Vec);
FLUCA_EXTERN PetscErrorCode PhysComputeIJacobian(Phys, PetscReal, Vec, Vec, PetscReal, Mat, Mat);

FLUCA_EXTERN PetscErrorCode PhysSetFromOptions(Phys);
FLUCA_EXTERN PetscErrorCode PhysSetOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysAppendOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysGetOptionsPrefix(Phys, const char *[]);

/* PHYSINS specific */
FLUCA_EXTERN PetscErrorCode PhysINSSetDensity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysINSGetDensity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysINSSetViscosity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysINSGetViscosity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysINSSetBoundaryCondition(Phys, PetscInt, PhysINSBC);
FLUCA_EXTERN PetscErrorCode PhysINSGetBoundaryCondition(Phys, PetscInt, PhysINSBC *);

FLUCA_EXTERN PetscFunctionList PhysList;
FLUCA_EXTERN PetscErrorCode    PhysRegister(const char[], PetscErrorCode (*)(Phys));
