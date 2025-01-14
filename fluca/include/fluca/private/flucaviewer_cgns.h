#pragma once

#include <fluca/private/flucaviewerimpl.h>

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
} PetscViewer_FlucaCGNS;

FLUCA_EXTERN PetscErrorCode PetscViewerFileOpen_FlucaCGNS_Internal(PetscViewer, PetscInt);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSCheckBatch_Internal(PetscViewer);

#define CGNSCall(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)
