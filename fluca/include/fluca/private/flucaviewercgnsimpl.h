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

/* Fields of a vector on a DMStag, written to or read from the FlowSolution of the current output step. A field is the
   components [c0, c0 + ncomp) at loc, which is DMSTAG_ELEMENT for cell data or DMSTAG_LEFT for face data covering the
   faces normal to every direction, stored under name (suffixed X, Y, Z when ncomp > 1). Writing needs
   PetscViewerFlucaCGNSBeginStep_Internal() and then the grid zone (MeshView()) first; reading takes the last step in the
   file and sets the DM's output sequence number from it. */
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSBeginStep_Internal(PetscViewer, PetscInt, PetscReal);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSWriteDMStagComponents_Internal(PetscViewer, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[]);
FLUCA_EXTERN PetscErrorCode PetscViewerFlucaCGNSReadDMStagComponents_Internal(PetscViewer, Vec, DMStagStencilLocation, PetscInt, PetscInt, const char[]);

#define CGNSCall(ierr) \
  do { \
    int _cgns_ier = (ierr); \
    PetscCheck(!_cgns_ier, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", _cgns_ier, cg_get_error()); \
  } while (0)
