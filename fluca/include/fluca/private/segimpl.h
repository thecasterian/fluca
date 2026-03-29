#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaseg.h>

FLUCA_EXTERN PetscBool      SegRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode SegRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  SEG_SetUp, SEG_Step;

typedef struct _SegOps *SegOps;

struct _SegOps {
  PetscErrorCode (*setfromoptions)(Seg, PetscOptionItems);
  PetscErrorCode (*setup)(Seg);
  PetscErrorCode (*step)(Seg);
  PetscErrorCode (*reset)(Seg);
  PetscErrorCode (*destroy)(Seg);
  PetscErrorCode (*view)(Seg, PetscViewer);
};

struct _p_Seg {
  PETSCHEADER(struct _SegOps);

  /* Solution */
  Vec sol; /* solution vector (borrowed, not owned) */
  DM  dm;  /* solution DM (borrowed, not owned) */

  /* Time state */
  PetscReal t;           /* current time */
  PetscReal dt;          /* time step size */
  PetscReal max_time;    /* maximum simulation time */
  PetscInt  step_number; /* current step number */
  PetscInt  max_steps;   /* maximum number of steps */

  /* Explicit RHS callback */
  SegRHSFn *rhsfn;
  void     *rhsfn_ctx;

  /* State flags */
  PetscBool setupcalled;

  /* Subtype-specific data */
  void *data;
};
