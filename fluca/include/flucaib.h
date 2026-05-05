#pragma once

#include <flucasys.h>
#include <petscdm.h>
#include <petscvec.h>

/* FlucaIB - Immersed Boundary representation wrapping a DMStag.
 *
 * FlucaIB is the canonical grid abstraction at the solver level: Phys, Seg, and other
 * solver-side modules consume FlucaIB rather than a bare DM. Subtypes encapsulate
 * different IB methods (e.g., ghost cell), or, in the trivial case (FLUCAIB_NONE),
 * "no immersed body" — a uniform pass-through over the underlying DMStag. */
typedef struct _p_FlucaIB *FlucaIB;

/* FlucaIB types */
typedef const char *FlucaIBType;
#define FLUCAIB_NONE "none" /* Trivial: bare DMStag, no body */

/* Field tags for FlucaIBApplyBC */
typedef enum {
  FLUCAIB_VELOCITY = 0,
  FLUCAIB_PRESSURE = 1
} FlucaIBField;

FLUCA_EXTERN PetscClassId   FLUCAIB_CLASSID;
FLUCA_EXTERN PetscErrorCode FlucaIBInitializePackage(void);
FLUCA_EXTERN PetscErrorCode FlucaIBFinalizePackage(void);

/* Lifecycle */
FLUCA_EXTERN PetscErrorCode FlucaIBCreate(MPI_Comm, FlucaIB *);
FLUCA_EXTERN PetscErrorCode FlucaIBCreateNone(MPI_Comm, DM, FlucaIB *);
FLUCA_EXTERN PetscErrorCode FlucaIBSetType(FlucaIB, FlucaIBType);
FLUCA_EXTERN PetscErrorCode FlucaIBGetType(FlucaIB, FlucaIBType *);
FLUCA_EXTERN PetscErrorCode FlucaIBSetFromOptions(FlucaIB);
FLUCA_EXTERN PetscErrorCode FlucaIBSetUp(FlucaIB);
FLUCA_EXTERN PetscErrorCode FlucaIBDestroy(FlucaIB *);
FLUCA_EXTERN PetscErrorCode FlucaIBView(FlucaIB, PetscViewer);
FLUCA_EXTERN PetscErrorCode FlucaIBViewFromOptions(FlucaIB, PetscObject, const char[]);

/* Options prefix */
FLUCA_EXTERN PetscErrorCode FlucaIBSetOptionsPrefix(FlucaIB, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaIBAppendOptionsPrefix(FlucaIB, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaIBGetOptionsPrefix(FlucaIB, const char *[]);

/* DM */
FLUCA_EXTERN PetscErrorCode FlucaIBSetDM(FlucaIB, DM);
FLUCA_EXTERN PetscErrorCode FlucaIBGetDM(FlucaIB, DM *);

/* Operations (subtype-dispatched) */
FLUCA_EXTERN PetscErrorCode FlucaIBApplyBC(FlucaIB, Vec, FlucaIBField);

/* Registration */
FLUCA_EXTERN PetscFunctionList FlucaIBList;
FLUCA_EXTERN PetscErrorCode    FlucaIBRegister(const char[], PetscErrorCode (*)(FlucaIB));
