#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucaseg.h>

#define MAXSEGMONITORS 10

FLUCA_EXTERN PetscBool     SegRegisterAllCalled;
FLUCA_EXTERN PetscLogEvent SEG_SetUp;
FLUCA_EXTERN PetscLogEvent SEG_Step;

FLUCA_INTERN PetscErrorCode SegCreate_CNLinear(Seg);
FLUCA_INTERN PetscErrorCode PCCreate_ABF(PC);

/* How a step turns the solution of the coupled system into the new state of a field */
typedef enum {
  SEG_FIELD_UPDATE_VALUE,     /* the solve returns the new value -> VecCopy */
  SEG_FIELD_UPDATE_INCREMENT, /* the solve returns a correction  -> VecAXPY */
} SegFieldUpdate;

#define SEG_MAX_FIELDS 8

/* One row block of the coupled system, in the order the subtype writes it back */
typedef struct {
  const char    *name;   /* field name on the attached Phys; static storage */
  IS             is;     /* its entries of the solution vector; owned by the Seg */
  SegFieldUpdate update; /* how the step applies the solve to it */
} SegFieldEntry;

typedef struct _SegOps *SegOps;

struct _SegOps {
  PetscErrorCode (*setfromoptions)(Seg, PetscOptionItems);
  PetscErrorCode (*setup)(Seg);
  PetscErrorCode (*presolve)(Seg);
  PetscErrorCode (*step)(Seg);
  PetscErrorCode (*destroy)(Seg);
  PetscErrorCode (*view)(Seg, PetscViewer);
};

struct _p_Seg {
  PETSCHEADER(struct _SegOps);

  /* Parameters */
  Phys      phys;      /* physical model providing the rows of the coupled system */
  PetscReal dt;        /* time step size */
  PetscReal max_time;  /* final time */
  PetscInt  max_steps; /* maximum number of steps */

  /* Data */
  Vec   sol;  /* solution vector, owned by the caller of SegSolve() */
  void *data; /* implementation-specific data */

  /* Fields of the coupled system, filled by the subtype during setup */
  PetscInt      nfields;
  SegFieldEntry fields[SEG_MAX_FIELDS];

  /* Solver */
  KSP                ksp;               /* coupled solve of eq. (13) */
  PetscBool          errorifstepfailed; /* raise an error when a step fails */
  SegConvergedReason reason;            /* convergence reason */

  /* State */
  PetscReal t;           /* current time */
  PetscInt  step;        /* current step number */
  PetscBool setupcalled; /* whether SegSetUp() has been called */

  /* Pre-step callback */
  PetscErrorCode (*prestep)(Seg, void *);
  void *prestep_ctx;

  /* Monitors */
  PetscInt num_mons;
  PetscErrorCode (*mons[MAXSEGMONITORS])(Seg, void *);
  void *mon_ctxs[MAXSEGMONITORS];
  PetscErrorCode (*mon_ctx_destroys[MAXSEGMONITORS])(void **);
};
