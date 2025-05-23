#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>
#include <flucasol.h>

FLUCA_EXTERN PetscBool      SolRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode SolRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  SOL_LoadFromFile;

typedef struct _SolOps *SolOps;

struct _SolOps {
  PetscErrorCode (*setmesh)(Sol, Mesh);
  PetscErrorCode (*destroy)(Sol);
  PetscErrorCode (*view)(Sol, PetscViewer);
  PetscErrorCode (*loadcgns)(Sol, PetscInt);
};

struct _p_Sol {
  PETSCHEADER(struct _SolOps);

  /* Data ----------------------------------------------------------------- */
  Mesh  mesh;
  Vec   v[3];
  Vec   p;
  void *data; /* implementation-specific data */
};
