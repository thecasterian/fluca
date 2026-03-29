#pragma once

#include <flucasys.h>
#include <flucafd.h>
#include <petscdmstag.h>

/* Seg - Segregated Time Integrator */
typedef struct _p_Seg *Seg;

/* Seg types */
typedef const char *SegType;
#define SEGSRK "srk" /* Segregated Runge-Kutta */

FLUCA_EXTERN PetscClassId   SEG_CLASSID;
FLUCA_EXTERN PetscErrorCode SegInitializePackage(void);
FLUCA_EXTERN PetscErrorCode SegFinalizePackage(void);

/* Explicit RHS callback: evaluates explicit right-hand side at (t, Y) into F.
   For INS-SRK, this computes: -(conv(u) + G(p) - source) / rho for velocity, 0 for pressure.
   The callback is responsible for any state updates (e.g., mass flux) needed before evaluation. */
typedef PetscErrorCode SegRHSFn(PetscReal t, Vec Y, Vec F, void *ctx);

/* Lifecycle */
FLUCA_EXTERN PetscErrorCode SegCreate(MPI_Comm, Seg *);
FLUCA_EXTERN PetscErrorCode SegSetType(Seg, SegType);
FLUCA_EXTERN PetscErrorCode SegGetType(Seg, SegType *);
FLUCA_EXTERN PetscErrorCode SegSetFromOptions(Seg);
FLUCA_EXTERN PetscErrorCode SegSetUp(Seg);
FLUCA_EXTERN PetscErrorCode SegReset(Seg);
FLUCA_EXTERN PetscErrorCode SegDestroy(Seg *);
FLUCA_EXTERN PetscErrorCode SegView(Seg, PetscViewer);
FLUCA_EXTERN PetscErrorCode SegViewFromOptions(Seg, PetscObject, const char[]);

/* Options prefix */
FLUCA_EXTERN PetscErrorCode SegSetOptionsPrefix(Seg, const char[]);
FLUCA_EXTERN PetscErrorCode SegAppendOptionsPrefix(Seg, const char[]);
FLUCA_EXTERN PetscErrorCode SegGetOptionsPrefix(Seg, const char *[]);

/* DM */
FLUCA_EXTERN PetscErrorCode SegSetDM(Seg, DM);
FLUCA_EXTERN PetscErrorCode SegGetDM(Seg, DM *);

/* Solution */
FLUCA_EXTERN PetscErrorCode SegSetSolution(Seg, Vec);
FLUCA_EXTERN PetscErrorCode SegGetSolution(Seg, Vec *);

/* RHS function */
FLUCA_EXTERN PetscErrorCode SegSetRHSFunction(Seg, SegRHSFn *, void *);

/* Time management */
FLUCA_EXTERN PetscErrorCode SegSetTimeStepSize(Seg, PetscReal);
FLUCA_EXTERN PetscErrorCode SegGetTimeStepSize(Seg, PetscReal *);
FLUCA_EXTERN PetscErrorCode SegSetTime(Seg, PetscReal);
FLUCA_EXTERN PetscErrorCode SegGetTime(Seg, PetscReal *);
FLUCA_EXTERN PetscErrorCode SegSetMaxTime(Seg, PetscReal);
FLUCA_EXTERN PetscErrorCode SegGetMaxTime(Seg, PetscReal *);
FLUCA_EXTERN PetscErrorCode SegSetMaxSteps(Seg, PetscInt);
FLUCA_EXTERN PetscErrorCode SegGetMaxSteps(Seg, PetscInt *);
FLUCA_EXTERN PetscErrorCode SegGetStepNumber(Seg, PetscInt *);

/* Stepping */
FLUCA_EXTERN PetscErrorCode SegStep(Seg);
FLUCA_EXTERN PetscErrorCode SegSolve(Seg);

/* Registration */
FLUCA_EXTERN PetscFunctionList SegList;
FLUCA_EXTERN PetscErrorCode    SegRegister(const char[], PetscErrorCode (*)(Seg));

/* SEGSRK specific */
FLUCA_EXTERN PetscErrorCode SegSRKSetLaplacian(Seg, PetscInt, FlucaFD);
FLUCA_EXTERN PetscErrorCode SegSRKSetGradient(Seg, PetscInt, FlucaFD);
FLUCA_EXTERN PetscErrorCode SegSRKSetDivergence(Seg, FlucaFD);
FLUCA_EXTERN PetscErrorCode SegSRKSetFieldIS(Seg, PetscInt, IS, IS, IS[]);
FLUCA_EXTERN PetscErrorCode SegSRKSetStabilization(Seg, FlucaFD);
FLUCA_EXTERN PetscErrorCode SegSRKSetDensity(Seg, PetscReal);
