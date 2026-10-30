#pragma once

#include <flucaphys.h>
#include <petscis.h>
#include <petscsnes.h>

typedef struct _p_NS *NS;

typedef const char *NSType;
#define NSCNLINEAR "cnlinear"

typedef enum {
  NS_CONVERGED_ITERATING      = 0,
  NS_CONVERGED_TIME           = 1,
  NS_CONVERGED_ITS            = 2,
  NS_DIVERGED_NONLINEAR_SOLVE = -1,
} NSConvergedReason;
FLUCA_EXTERN const char *const *NSConvergedReasons;

FLUCA_EXTERN PetscClassId NS_CLASSID;

FLUCA_EXTERN PetscErrorCode NSInitializePackage(void);
FLUCA_EXTERN PetscErrorCode NSFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode NSCreate(MPI_Comm, NS *);
FLUCA_EXTERN PetscErrorCode NSSetType(NS, NSType);
FLUCA_EXTERN PetscErrorCode NSGetType(NS, NSType *);
FLUCA_EXTERN PetscErrorCode NSSetPhys(NS, Phys);
FLUCA_EXTERN PetscErrorCode NSGetPhys(NS, Phys *);
FLUCA_EXTERN PetscErrorCode NSSetTimeStepSize(NS, PetscReal);
FLUCA_EXTERN PetscErrorCode NSGetTimeStepSize(NS, PetscReal *);
FLUCA_EXTERN PetscErrorCode NSSetTimeStep(NS, PetscInt);
FLUCA_EXTERN PetscErrorCode NSGetTimeStep(NS, PetscInt *);
FLUCA_EXTERN PetscErrorCode NSSetTime(NS, PetscReal);
FLUCA_EXTERN PetscErrorCode NSGetTime(NS, PetscReal *);
FLUCA_EXTERN PetscErrorCode NSSetMaxTime(NS, PetscReal);
FLUCA_EXTERN PetscErrorCode NSGetMaxTime(NS, PetscReal *);
FLUCA_EXTERN PetscErrorCode NSSetMaxSteps(NS, PetscInt);
FLUCA_EXTERN PetscErrorCode NSGetMaxSteps(NS, PetscInt *);
FLUCA_EXTERN PetscErrorCode NSSetFromOptions(NS);
FLUCA_EXTERN PetscErrorCode NSSetUp(NS);
FLUCA_EXTERN PetscErrorCode NSStep(NS);
FLUCA_EXTERN PetscErrorCode NSSolve(NS);
FLUCA_EXTERN PetscErrorCode NSView(NS, PetscViewer);
FLUCA_EXTERN PetscErrorCode NSViewFromOptions(NS, PetscObject, const char[]);
FLUCA_EXTERN PetscErrorCode NSDestroy(NS *);

FLUCA_EXTERN PetscErrorCode NSSetConvergedReason(NS, NSConvergedReason);
FLUCA_EXTERN PetscErrorCode NSGetConvergedReason(NS, NSConvergedReason *);
FLUCA_EXTERN PetscErrorCode NSSetErrorIfStepFailed(NS, PetscBool);
FLUCA_EXTERN PetscErrorCode NSGetErrorIfStepFailed(NS, PetscBool *);
FLUCA_EXTERN PetscErrorCode NSCheckDiverged(NS);

FLUCA_EXTERN PetscErrorCode NSFormJacobian(NS, Vec, Mat);
FLUCA_EXTERN PetscErrorCode NSFormFunction(NS, Vec, Vec);

FLUCA_EXTERN PetscErrorCode NSGetSNES(NS, SNES *);
FLUCA_EXTERN PetscErrorCode NSGetSolution(NS, Vec *);
FLUCA_EXTERN PetscErrorCode NSGetField(NS, const char[], IS *);
FLUCA_EXTERN PetscErrorCode NSGetSolutionSubVector(NS, const char[], Vec *);
FLUCA_EXTERN PetscErrorCode NSRestoreSolutionSubVector(NS, const char[], Vec *);
FLUCA_EXTERN PetscErrorCode NSViewSolution(NS, PetscViewer);
FLUCA_EXTERN PetscErrorCode NSViewSolutionFromOptions(NS, PetscObject, const char[]);
FLUCA_EXTERN PetscErrorCode NSLoadSolution(NS, PetscViewer);

FLUCA_EXTERN PetscErrorCode NSMonitorSet(NS, PetscErrorCode (*)(NS, void *), void *, PetscCtxDestroyFn *);
FLUCA_EXTERN PetscErrorCode NSMonitorCancel(NS);
FLUCA_EXTERN PetscErrorCode NSMonitor(NS);
FLUCA_EXTERN PetscErrorCode NSMonitorSetFromOptions(NS, const char[], const char[], const char[], PetscErrorCode (*)(NS, PetscViewerAndFormat *), PetscErrorCode (*)(NS, PetscViewerAndFormat *));
FLUCA_EXTERN PetscErrorCode NSMonitorDefault(NS, PetscViewerAndFormat *);
FLUCA_EXTERN PetscErrorCode NSMonitorSolution(NS, PetscViewerAndFormat *);

FLUCA_EXTERN PetscFunctionList NSList;
FLUCA_EXTERN PetscErrorCode    NSRegister(const char[], PetscErrorCode (*)(NS));

#define PCABF "abf"

FLUCA_EXTERN PetscErrorCode PCABFSetFieldIS(PC, IS, IS, IS);
FLUCA_EXTERN PetscErrorCode PCABFGetSubKSPs(PC, KSP *, KSP *);

typedef enum {
  PC_ABF_AINV_ID,
  PC_ABF_AINV_DIAG,
  PC_ABF_AINV_ROWSUM,
} PCABFAinvType;
FLUCA_EXTERN const char *const PCABFAinvTypes[];

FLUCA_EXTERN PetscErrorCode PCABFSetSchurComplementAinvType(PC, PCABFAinvType);
FLUCA_EXTERN PetscErrorCode PCABFSetUpperTriangularAinvType(PC, PCABFAinvType);
