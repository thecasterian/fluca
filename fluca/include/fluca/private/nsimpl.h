#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>
#include <flucans.h>
#include <flucansbc.h>
#include <petscsnes.h>

#define MAXNSMONITORS 10

FLUCA_EXTERN PetscBool      NSRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode NSRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  NS_SetUp;
FLUCA_EXTERN PetscLogEvent  NS_Step;
FLUCA_EXTERN PetscLogEvent  NS_FormJacobian;
FLUCA_EXTERN PetscLogEvent  NS_FormFunction;

typedef struct _NSOps *NSOps;

struct _NSOps {
  PetscErrorCode (*setfromoptions)(NS, PetscOptionItems);
  PetscErrorCode (*setup)(NS);
  PetscErrorCode (*step)(NS);
  PetscErrorCode (*formjacobian)(NS, Vec, Mat, NSFormJacobianType);
  PetscErrorCode (*formfunction)(NS, Vec, Vec);
  PetscErrorCode (*destroy)(NS);
  PetscErrorCode (*view)(NS, PetscViewer);
  PetscErrorCode (*viewsolution)(NS, PetscViewer);
  PetscErrorCode (*loadsolution)(NS, PetscViewer);
};

typedef struct _n_NSFieldLink *NSFieldLink;
struct _n_NSFieldLink {
  char       *fieldname;
  MeshDMType  dmtype;
  IS          is; /* indices in solution vector */
  NSFieldLink prev, next;
};

struct _p_NS {
  PETSCHEADER(struct _NSOps);

  /* Parameters ----------------------------------------------------------- */
  PetscReal rho;       /* density */
  PetscReal mu;        /* dynamic viscosity */
  PetscReal dt;        /* time step size */
  PetscReal max_time;  /* maximum time */
  PetscInt  max_steps; /* maximum number of steps */

  /* Data ----------------------------------------------------------------- */
  PetscInt             step; /* current time step */
  PetscReal            t;    /* current time */
  Mesh                 mesh; /* mesh */
  NSBoundaryCondition *bcs;  /* boundary conditions */
  void                *data; /* implementation-specific data */

  /* Solution ------------------------------------------------------------- */
  NSFieldLink fieldlink; /* list of fields */
  DM          soldm;     /* DM for solution vector */
  Vec         sol;       /* solution vector */
  Vec         sol0;      /* solution vector at the beginning of time step */

  /* Solver --------------------------------------------------------------- */
  NSSolver     solver;    /* solver type */
  SNES         snes;      /* non-linear solver */
  Mat          J;         /* Jacobian */
  Vec          r;         /* residual vector */
  Vec          x;         /* solver solution vector */
  MatNullSpace nullspace; /* null space of Jacobian */

  PetscBool         errorifstepfailed; /* error if step fails */
  NSConvergedReason reason;            /* convergence reason */

  /* State ---------------------------------------------------------------- */
  PetscBool setupcalled; /* whether NSSetUp() has been called */

  /* Monitor -------------------------------------------------------------- */
  PetscInt num_mons;
  PetscErrorCode (*mons[MAXNSMONITORS])(NS, void *);
  void *mon_ctxs[MAXNSMONITORS];
  PetscErrorCode (*mon_ctx_destroys[MAXNSMONITORS])(void **);
};

typedef struct {
  IS vis; /* index set of velocity component */
  IS Vis; /* index set of face normal velocity component */
  IS pis; /* index set of pressure component */

  Mat A;   /* operator of momentum equation */
  Mat T;   /* velocity interpolation operator */
  Mat G;   /* pressure gradient operator */
  Mat Gst; /* staggered pressure gradient operator */
  Mat D;   /* face-normal velocity divergence operator */
  Mat Lst; /* operator of pressure equation: Lst = D * Gst */

  Vec divvstar;
  Vec gradpcorr;
  Vec gradstpcorr;

  KSP kspv; /* KSP to solve velocity equation */
  KSP kspp; /* KSP to solve pressure equation */

  MatNullSpace nullspace;
} NSFSMPCCtx;

FLUCA_INTERN PetscErrorCode NSSetPreconditioner_Internal(NS, NSSolver);
