#pragma once

#include <flucasol.h>

typedef struct _p_NS *NS;

typedef const char *NSType;
#define NSFSM "fsm"

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
FLUCA_EXTERN PetscErrorCode NSSetFromOptions(NS);
FLUCA_EXTERN PetscErrorCode NSSetUp(NS);
FLUCA_EXTERN PetscErrorCode NSSolve(NS, PetscInt);
FLUCA_EXTERN PetscErrorCode NSGetSol(NS, Sol *);
FLUCA_EXTERN PetscErrorCode NSDestroy(NS *);
FLUCA_EXTERN PetscErrorCode NSView(NS, PetscViewer);
FLUCA_EXTERN PetscErrorCode NSViewFromOptions(NS, PetscObject, const char[]);

FLUCA_EXTERN PetscErrorCode NSMonitorSet(NS, PetscErrorCode (*)(NS, void *), void *, PetscErrorCode (*)(void **));
FLUCA_EXTERN PetscErrorCode NSMonitorCancel(NS);
FLUCA_EXTERN PetscErrorCode NSMonitor(NS);
FLUCA_EXTERN PetscErrorCode NSMonitorSetFrequency(NS, PetscInt);
FLUCA_EXTERN PetscErrorCode NSMonitorSetFromOptions(NS, const char[], const char[], const char[], PetscErrorCode (*)(NS, PetscViewerAndFormat *), PetscErrorCode (*)(NS, PetscViewerAndFormat *));
FLUCA_EXTERN PetscErrorCode NSMonitorDefault(NS, PetscViewerAndFormat *);
FLUCA_EXTERN PetscErrorCode NSMonitorSolution(NS, PetscViewerAndFormat *);

FLUCA_EXTERN PetscFunctionList NSList;
FLUCA_EXTERN PetscErrorCode    NSRegister(const char[], PetscErrorCode (*)(NS));
