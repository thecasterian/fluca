#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucafd.h>
#include <flucans.h>
#include <petscsnes.h>

#define MAXNSMONITORS 10

FLUCA_EXTERN PetscBool      NSRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode NSRegisterAll(void);
FLUCA_EXTERN PetscErrorCode NSPCRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  NS_SetUp;
FLUCA_EXTERN PetscLogEvent  NS_Step;
FLUCA_EXTERN PetscLogEvent  NS_FormJacobian;
FLUCA_EXTERN PetscLogEvent  NS_FormFunction;

/* Face stencil locations indexed by direction: LEFT for x, DOWN for y, BACK for z. PETSC_UNUSED
   because not every file including this header references it. */
PETSC_UNUSED static const DMStagStencilLocation face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* Bridges PhysBCFn (has comp) to FlucaFDBCValueFn (no comp) */
typedef struct {
  PhysBCFn *fn;
  PhysBCFn *fn_dot;
  void     *fn_ctx;
  void     *fn_dot_ctx;
  PetscInt  comp;
} NS_BCAdapter;

typedef struct _NSOps *NSOps;

struct _NSOps {
  PetscErrorCode (*setfromoptions)(NS, PetscOptionItems);
  PetscErrorCode (*setup)(NS);
  PetscErrorCode (*step)(NS);
  PetscErrorCode (*formjacobian)(NS, Vec, Mat);
  PetscErrorCode (*formfunction)(NS, Vec, Vec);
  PetscErrorCode (*destroy)(NS);
  PetscErrorCode (*view)(NS, PetscViewer);
};

struct _p_NS {
  PETSCHEADER(struct _NSOps);

  /* Parameters ----------------------------------------------------------- */
  PetscReal dt;        /* time step size */
  PetscReal max_time;  /* maximum time */
  PetscInt  max_steps; /* maximum number of steps */

  /* Data ----------------------------------------------------------------- */
  Phys      phys; /* problem statement; referenced */
  PetscInt  step; /* current time step */
  PetscReal t;    /* current time */
  void     *data; /* implementation-specific data */

  /* Solution ------------------------------------------------------------- */
  Vec sol;  /* solution vector */
  Vec sol0; /* solution vector at the beginning of the time step */

  /* Spatial operators, built by NSSetUpSpatialOperators_Internal() -------- */
  NS_BCAdapter bcadapters[PHYS_MAX_DIM][PHYS_MAX_FACES]; /* [velocity component][face] */
  FlucaFD      fd_negmu[PHYS_MAX_DIM][PHYS_MAX_DIM];     /* [d][e]: -mu d(u_d)/dx_e, nested in fd_laplacian[d] */
  FlucaFD      fd_laplacian[PHYS_MAX_DIM];               /* sum_e d/dx_e(-mu d(u_d)/dx_e) */
  FlucaFD      fd_grad_p[PHYS_MAX_DIM];                  /* dp/dx_d at cells, no BC (one-sided at walls) */
  FlucaFD      fd_negT[PHYS_MAX_DIM];                    /* -T: two-point u_e -> faces normal to e, velocity BCs */
  FlucaFD      fd_D;                                     /* sum_e d/dx_e(U_e): faces -> pressure rows */
  Mat          negR;                                     /* (-T) G_c + G^st, unscaled */
  Vec          zero;                                     /* evaluates the boundary (affine) part of an operator */

  /* Solver --------------------------------------------------------------- */
  SNES         snes;      /* non-linear solver */
  Mat          J;         /* AIJ from DMCreateMatrix() on the Phys solution DM */
  Vec          r;         /* residual vector */
  Vec          x;         /* solver solution vector */
  MatNullSpace nullspace; /* null space of J */

  PetscBool         errorifstepfailed; /* error if step fails */
  NSConvergedReason reason;            /* convergence reason */

  /* State ---------------------------------------------------------------- */
  PetscBool setupcalled; /* whether NSSetUp() has been called */

  /* Monitor -------------------------------------------------------------- */
  PetscInt num_mons;
  PetscErrorCode (*mons[MAXNSMONITORS])(NS, void *);
  void              *mon_ctxs[MAXNSMONITORS];
  PetscCtxDestroyFn *mon_ctx_destroys[MAXNSMONITORS];
};

/* Defined in interface/nsops.c */
FLUCA_INTERN PetscErrorCode NSSetUpSpatialOperators_Internal(NS);
FLUCA_INTERN PetscErrorCode NSDestroySpatialOperators_Internal(NS);
FLUCA_INTERN PetscErrorCode NSSetVelocityBCs_Internal(NS, FlucaFD, PetscInt);

/* Defined in interface/nsmon.c */
FLUCA_INTERN PetscErrorCode NSMonitorSolutionSetUp_Internal(NS, PetscViewerAndFormat *);
