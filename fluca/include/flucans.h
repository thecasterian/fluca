#pragma once

#include <flucamesh.h>
#include <flucansbc.h>
#include <petscis.h>

typedef struct _p_NS *NS;

typedef const char *NSType;
#define NSCNLINEAR "cnlinear"

typedef enum {
  NS_FSM,
} NSSolver;
FLUCA_EXTERN const char *const NSSolvers[];

typedef enum {
  NS_CONVERGED_ITERATING      = 0,
  NS_CONVERGED_TIME           = 1,
  NS_CONVERGED_ITS            = 2,
  NS_DIVERGED_NONLINEAR_SOLVE = -1,
} NSConvergedReason;
FLUCA_EXTERN const char *const *NSConvergedReasons;

#define NS_FIELD_VELOCITY             "Velocity"
#define NS_FIELD_FACE_NORMAL_VELOCITY "FaceNormalVelocity"
#define NS_FIELD_PRESSURE             "Pressure"

FLUCA_EXTERN PetscClassId NS_CLASSID;

FLUCA_EXTERN PetscErrorCode NSInitializePackage(void);
FLUCA_EXTERN PetscErrorCode NSFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode NSCreate(MPI_Comm, NS *);
FLUCA_EXTERN PetscErrorCode NSSetType(NS, NSType);
FLUCA_EXTERN PetscErrorCode NSGetType(NS, NSType *);
FLUCA_EXTERN PetscErrorCode NSSetMesh(NS, Mesh);
FLUCA_EXTERN PetscErrorCode NSGetMesh(NS, Mesh *);
FLUCA_EXTERN PetscErrorCode NSSetDensity(NS, PetscReal);
FLUCA_EXTERN PetscErrorCode NSGetDensity(NS, PetscReal *);
FLUCA_EXTERN PetscErrorCode NSSetViscosity(NS, PetscReal);
FLUCA_EXTERN PetscErrorCode NSGetViscosity(NS, PetscReal *);
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
FLUCA_EXTERN PetscErrorCode NSSetBoundaryCondition(NS, PetscInt, NSBoundaryCondition);
FLUCA_EXTERN PetscErrorCode NSGetBoundaryCondition(NS, PetscInt, NSBoundaryCondition *);
FLUCA_EXTERN PetscErrorCode NSSetSolver(NS, NSSolver);
FLUCA_EXTERN PetscErrorCode NSGetSolver(NS, NSSolver *);
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

typedef enum {
  NS_INIT_JACOBIAN,
  NS_UPDATE_JACOBIAN,
} NSFormJacobianType;
FLUCA_EXTERN PetscErrorCode NSFormJacobian(NS, Vec, Mat, NSFormJacobianType);
FLUCA_EXTERN PetscErrorCode NSFormFunction(NS, Vec, Vec);

FLUCA_EXTERN PetscErrorCode NSGetSolution(NS, Vec *);
FLUCA_EXTERN PetscErrorCode NSGetNumFields(NS, PetscInt *);
FLUCA_EXTERN PetscErrorCode NSGetField(NS, const char[], PetscInt *, MeshDMType *, IS *);
FLUCA_EXTERN PetscErrorCode NSGetFieldByIndex(NS, PetscInt, const char *[], MeshDMType *, IS *);
FLUCA_EXTERN PetscErrorCode NSGetSolutionSubVector(NS, const char[], Vec *);
FLUCA_EXTERN PetscErrorCode NSRestoreSolutionSubVector(NS, const char[], Vec *);
FLUCA_EXTERN PetscErrorCode NSViewSolution(NS, PetscViewer);
FLUCA_EXTERN PetscErrorCode NSViewSolutionFromOptions(NS, PetscObject, const char[]);
FLUCA_EXTERN PetscErrorCode NSLoadSolution(NS, PetscViewer);

FLUCA_EXTERN PetscErrorCode NSMonitorSet(NS, PetscErrorCode (*)(NS, void *), void *, PetscErrorCode (*)(void **));
FLUCA_EXTERN PetscErrorCode NSMonitorCancel(NS);
FLUCA_EXTERN PetscErrorCode NSMonitor(NS);
FLUCA_EXTERN PetscErrorCode NSMonitorSetFromOptions(NS, const char[], const char[], const char[], PetscErrorCode (*)(NS, PetscViewerAndFormat *), PetscErrorCode (*)(NS, PetscViewerAndFormat *));
FLUCA_EXTERN PetscErrorCode NSMonitorDefault(NS, PetscViewerAndFormat *);
FLUCA_EXTERN PetscErrorCode NSMonitorSolution(NS, PetscViewerAndFormat *);

FLUCA_EXTERN PetscFunctionList NSList;
FLUCA_EXTERN PetscErrorCode    NSRegister(const char[], PetscErrorCode (*)(NS));
