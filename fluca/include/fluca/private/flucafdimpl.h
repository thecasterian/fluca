#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucafd.h>
#include <petscdmstag.h>

#define FLUCAFD_MAX_DIM          3
#define FLUCAFD_MAX_STENCIL_SIZE 32
#define FLUCAFD_ZERO_PIVOT_TOL   1e-14
#define FLUCAFD_COEFF_ATOL       1e-10
#define FLUCAFD_COEFF_RTOL       1e-8

FLUCA_EXTERN PetscBool      FlucaFDRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode FlucaFDRegisterAll(void);

typedef struct _n_FlucaFDTermInfoLink *FlucaFDTermLink;
struct _n_FlucaFDTermInfoLink {
  PetscInt              deriv_order[FLUCAFD_MAX_DIM];
  PetscInt              accu_order[FLUCAFD_MAX_DIM];
  DMStagStencilLocation input_loc;
  PetscInt              input_c;
  FlucaFDTermLink       next;
};

typedef struct _FlucaFDOps *FlucaFDOps;

struct _FlucaFDOps {
  PetscErrorCode (*setfromoptions)(FlucaFD, PetscOptionItems);
  PetscErrorCode (*setup)(FlucaFD);
  PetscErrorCode (*getstencilraw)(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, DMStagStencil[], PetscScalar[]);
  PetscErrorCode (*destroy)(FlucaFD);
  PetscErrorCode (*view)(FlucaFD, PetscViewer);
};

struct _p_FlucaFD {
  PETSCHEADER(struct _FlucaFDOps);

  /* Parameters ----------------------------------------------------------- */
  DMStagStencilLocation    input_loc;
  PetscInt                 input_c;
  DMStagStencilLocation    output_loc;
  PetscInt                 output_c;
  FlucaFDBoundaryCondition bcs[2 * FLUCAFD_MAX_DIM];

  /* Data ----------------------------------------------------------------- */
  DM                  dm;
  PetscInt            dim;
  PetscInt            N[FLUCAFD_MAX_DIM];
  PetscInt            xs[FLUCAFD_MAX_DIM];
  PetscInt            xm[FLUCAFD_MAX_DIM];
  PetscBool           is_first_rank[FLUCAFD_MAX_DIM];
  PetscBool           is_last_rank[FLUCAFD_MAX_DIM];
  PetscBool           periodic[FLUCAFD_MAX_DIM];
  PetscInt            stencil_width;
  const PetscScalar **arr_coord[FLUCAFD_MAX_DIM];
  PetscInt            slot_coord_prev, slot_coord_elem;
  FlucaFDTermLink     termlink;
  void               *data;

  /* State ---------------------------------------------------------------- */
  PetscBool setupcalled;
};

typedef struct {
  FlucaFDDirection dir;         /* X, Y, or Z */
  PetscInt         deriv_order; /* 0, 1, 2, ... */
  PetscInt         accu_order;  /* 1, 2, 3, ... */

  PetscInt      ncols;
  DMStagStencil col[FLUCAFD_MAX_STENCIL_SIZE]; /* stencil with relative indices */

  PetscInt    v_start;
  PetscInt    v_end;
  PetscScalar v_prev[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar v_next[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar (*v)[FLUCAFD_MAX_STENCIL_SIZE];
} FlucaFD_Derivative;

typedef struct {
  FlucaFD inner; /* inner operator (applied first) */
  FlucaFD outer; /* outer operator (applied second) */
} FlucaFD_Composition;

typedef struct {
  FlucaFD               operand;
  PetscScalar           constant; /* scale constant (if constant scaling) */
  Vec                   vec;      /* scale vector (if vector scaling) */
  DMStagStencilLocation vec_loc;
  PetscInt              vec_c;
  PetscBool             is_constant; /* true if constant scaling, false if vector */

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

FLUCA_EXTERN PetscBool      FlucaFDLimiterRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode FlucaFDLimiterRegisterAll(void);

typedef struct {
  FlucaFDDirection  dir;
  FlucaFDLimiterFn *limiter;

  /* Pre-computed geometry for non-uniform grids */
  PetscInt     alpha_start;
  PetscInt     alpha_end;
  PetscScalar *alpha_plus;
  PetscScalar *alpha_minus;
  PetscScalar *alpha_plus_base;
  PetscScalar *alpha_minus_base;

  /* Velocity field (face-centered, single component on left/down/back face) */
  PetscInt             vel_c;
  DM                   vel_dm;      /* original DMStag (from Vec) */
  DM                   vel_da;      /* DMDA for local storage (1 DOF) */
  Vec                  vel_local;   /* local vector on vel_da */
  VecScatter           vel_scatter; /* global DMStag vel -> local DMDA vel */
  const PetscScalar   *arr_vel_1d;
  const PetscScalar  **arr_vel_2d;
  const PetscScalar ***arr_vel_3d;

  /* Current solution field (element-centered, single component) */
  DM                   phi_dm;      /* original DMStag (from Vec) */
  DM                   phi_da;      /* DMDA for local storage (1 DOF) */
  Vec                  phi_local;   /* local vector on phi_da */
  VecScatter           phi_scatter; /* global DMStag phi -> local DMDA phi */
  const PetscScalar   *arr_phi_1d;
  const PetscScalar  **arr_phi_2d;
  const PetscScalar ***arr_phi_3d;

  FlucaFD fd_grad; /* gradient operator (element -> face) */
} FlucaFD_SecondOrderTVD;

FLUCA_INTERN PetscErrorCode FlucaFDValidateOperand_Internal(FlucaFD, FlucaFD);

FLUCA_INTERN PetscErrorCode FlucaFDValidateStencilLocation_Internal(DMStagStencilLocation);
FLUCA_INTERN PetscErrorCode FlucaFDUseFaceCoordinate_Internal(DMStagStencilLocation, PetscInt, PetscBool *);
FLUCA_INTERN PetscErrorCode FlucaFDGetCoordinate_Internal(const PetscScalar **, PetscInt, PetscInt, PetscInt, PetscInt, PetscScalar, PetscScalar, PetscScalar *);
FLUCA_INTERN PetscErrorCode FlucaFDGetGhostCorners_Internal(FlucaFD, PetscInt, PetscBool, PetscInt *, PetscInt *, PetscInt *);
FLUCA_INTERN PetscErrorCode FlucaFDSolveLinearSystem_Internal(PetscInt, PetscScalar[], PetscScalar[], PetscScalar[]);
FLUCA_INTERN PetscErrorCode FlucaFDAddStencilPoint_Internal(DMStagStencil, PetscScalar, PetscInt *, DMStagStencil[], PetscScalar[]);
FLUCA_INTERN PetscErrorCode FlucaFDRemoveOffGridPoints_Internal(FlucaFD, PetscInt *, DMStagStencil[], PetscScalar[]);
FLUCA_INTERN PetscErrorCode FlucaFDRemoveZeroStencilPoints_Internal(PetscInt *, DMStagStencil[], PetscScalar[]);

FLUCA_INTERN PetscErrorCode FlucaFDTermLinkCreate_Internal(FlucaFDTermLink *);
FLUCA_INTERN PetscErrorCode FlucaFDTermLinkDuplicate_Internal(FlucaFDTermLink, FlucaFDTermLink *);
FLUCA_INTERN PetscErrorCode FlucaFDTermLinkAppend_Internal(FlucaFDTermLink *, FlucaFDTermLink);
FLUCA_INTERN PetscErrorCode FlucaFDTermLinkFind_Internal(FlucaFDTermLink, FlucaFDTermLink, PetscBool *);
FLUCA_INTERN PetscErrorCode FlucaFDTermLinkDestroy_Internal(FlucaFDTermLink *);
