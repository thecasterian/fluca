#pragma once

#include <fluca/private/flucaviewerimpl.h>
#include <pcgnslib.h>
#include <cgns_io.h>
#include <petscdmstag.h>

#define CGNS_MAX_DIM 3

typedef struct {
  char         *filename_template;
  char         *filename;
  PetscFileMode filemode;

  int file_num;
  int base, zone, sol;

  PetscSegBuffer output_steps;
  PetscSegBuffer output_times;
  PetscInt       last_step;
  PetscInt       batch_size;
  PetscBool      include_coord;
} PetscViewer_FlucaCGNS;

FLUCA_EXTERN PetscErrorCode FlucaGetCGNSDataType_Internal(PetscDataType, CGNS_ENUMT(DataType_t) *);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSFileOpen_Internal(PetscViewer, PetscInt);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSCheckBatch_Internal(PetscViewer);

FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSBeginStep_Internal(PetscViewer, PetscInt, PetscReal);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSWriteDMStagComponents_Internal(PetscViewer, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[]);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSReadDMStagComponents_Internal(PetscViewer, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[]);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSWriteDMStagComponentsToSolution_Internal(PetscViewer, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], int, PetscBool);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSReadDMStagComponentsFromSolution_Internal(PetscViewer, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[], int);

#define CGNSCall(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)
