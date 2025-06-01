#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>
#include <flucans.h>

#define MAXNSMONITORS 10

FLUCA_EXTERN PetscBool      NSRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode NSRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  NS_SetUp;
FLUCA_EXTERN PetscLogEvent  NS_LoadSolutionFromFile;
FLUCA_EXTERN PetscLogEvent  NS_Solve;

typedef struct _NSOps *NSOps;

struct _NSOps {
  PetscErrorCode (*setfromoptions)(NS, PetscOptionItems);
  PetscErrorCode (*setup)(NS);
  PetscErrorCode (*iterate)(NS);
  PetscErrorCode (*destroy)(NS);
  PetscErrorCode (*view)(NS, PetscViewer);
  PetscErrorCode (*viewsolution)(NS, PetscViewer);
  PetscErrorCode (*loadsolutioncgns)(NS, PetscInt);
};

struct _p_NS {
  PETSCHEADER(struct _NSOps);

  /* Parameters ----------------------------------------------------------- */
  PetscReal rho; /* density */
  PetscReal mu;  /* dynamic viscosity */
  PetscReal dt;  /* time step size */

  /* Data ----------------------------------------------------------------- */
  PetscInt  step; /* current time step */
  PetscReal t;    /* current time */
  Mesh      mesh; /* mesh */
  void     *data; /* implementation-specific data */

  /* State ---------------------------------------------------------------- */
  PetscBool setupcalled; /* whether NSSetUp() has been called */

  /* Monitor -------------------------------------------------------------- */
  PetscInt num_mons;
  PetscErrorCode (*mons[MAXNSMONITORS])(NS, void *);
  void *mon_ctxs[MAXNSMONITORS];
  PetscErrorCode (*mon_ctx_destroys[MAXNSMONITORS])(void **);
  PetscInt mon_freq;
};
