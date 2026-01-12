#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucafd.h>
#include <petscdmstag.h>

#define FLUCAFD_MAX_DIM 3

FLUCA_EXTERN PetscClassId   FLUCAFD_CLASSID;
FLUCA_EXTERN PetscBool      FlucaFDRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode FlucaFDRegisterAll(void);

typedef struct _FlucaFDOps *FlucaFDOps;

struct _FlucaFDOps {
  PetscErrorCode (*setfromoptions)(FlucaFD, PetscOptionItems);
  PetscErrorCode (*setup)(FlucaFD);
  PetscErrorCode (*getstencil)(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, DMStagStencil[], PetscScalar[]);
  PetscErrorCode (*destroy)(FlucaFD);
  PetscErrorCode (*view)(FlucaFD, PetscViewer);
};

struct _p_FlucaFD {
  PETSCHEADER(struct _FlucaFDOps);

  /* Parameters ----------------------------------------------------------- */
  FlucaFDStencilLocation   input_loc;
  PetscInt                 input_c;
  FlucaFDStencilLocation   output_loc;
  PetscInt                 output_c;
  FlucaFDBoundaryCondition bcs[2 * FLUCAFD_MAX_DIM];

  /* Data ----------------------------------------------------------------- */
  DM                  cdm;
  PetscInt            dim;
  PetscInt            N[FLUCAFD_MAX_DIM];
  PetscInt            x[FLUCAFD_MAX_DIM];
  PetscInt            n[FLUCAFD_MAX_DIM];
  PetscBool           is_first_rank[FLUCAFD_MAX_DIM];
  PetscBool           is_last_rank[FLUCAFD_MAX_DIM];
  PetscInt            stencil_width;
  const PetscScalar **arr_coord[FLUCAFD_MAX_DIM];
  PetscInt            slot_coord_prev, slot_coord_elem;
  void               *data;

  /* State ---------------------------------------------------------------- */
  PetscBool setupcalled;
};

typedef struct {
  FlucaFDDirection dir;         /* X, Y, or Z */
  PetscInt         deriv_order; /* 0, 1, 2, ... */
  PetscInt         accu_order;  /* 1, 2, 3, ... */

  PetscInt      ncols;
  DMStagStencil col[16]; /* stencil with relative indices */

  PetscInt    v_start;
  PetscInt    v_end;
  PetscScalar v_prev[16];
  PetscScalar v_next[16];
  PetscScalar (*v)[16];
} FlucaFD_Derivative;

typedef struct {
  FlucaFD inner; /* inner operator (applied first) */
  FlucaFD outer; /* outer operator (applied second) */
} FlucaFD_Composition;

typedef struct {
  FlucaFD                operand;
  PetscReal              constant; /* scale constant (if constant scaling) */
  Vec                    vec;      /* scale vector (if vector scaling) */
  FlucaFDStencilLocation vec_loc;
  PetscInt               vec_c;
  PetscBool              is_constant; /* true if constant scaling, false if vector */

  DM                    vec_dm;
  Vec                   vec_local;
  const PetscScalar   **arr_vec_1d;
  const PetscScalar  ***arr_vec_2d;
  const PetscScalar ****arr_vec_3d;
  PetscInt              vec_slot;
} FlucaFD_Scale;

typedef struct _n_FlucaFDSumOperandLink *FlucaFDSumOperandLink;
struct _n_FlucaFDSumOperandLink {
  FlucaFD               fd;
  FlucaFDSumOperandLink next;
};

typedef struct {
  FlucaFDSumOperandLink oplink;
} FlucaFD_Sum;

FLUCA_INTERN PetscErrorCode FlucaFDStencilLocationToDMStagStencilLocation_Internal(FlucaFDStencilLocation, DMStagStencilLocation *);
