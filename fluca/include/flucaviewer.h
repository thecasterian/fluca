#pragma once

#include <petscviewer.h>
#include <flucasys.h>

#define PETSCVIEWERFLUCACGNS "flucacgns"

FLUCA_EXTERN PetscErrorCode FlucaOptionsCreateViewer(MPI_Comm, PetscOptions, const char[], const char[], PetscViewer *, PetscViewerFormat *, PetscBool *);
PETSC_EXTERN PetscErrorCode FlucaObjectViewFromOptions(PetscObject, PetscObject, const char[]);

FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSOpen(MPI_Comm, const char[], PetscFileMode, PetscViewer *);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSSetBatchSize(PetscViewer, PetscInt);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSGetBatchSize(PetscViewer, PetscInt *);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSSetIncludeCoord(PetscViewer, PetscBool);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSGetIncludeCoord(PetscViewer, PetscBool *);

FLUCA_EXTERN PetscViewer PETSC_VIEWER_FLUCACGNS_(MPI_Comm);
