#pragma once

#include <fluca/private/segimpl.h>
#include <flucafd.h>
#include <flucaphys.h>

#define SEG_OPS_MAX_DIM   3
#define SEG_OPS_MAX_FACES (2 * SEG_OPS_MAX_DIM)

/* Adapter to bridge PhysLaminarBCFn (has comp) to FlucaFDBCValueFn (no comp) */
typedef struct {
  PhysLaminarBCFn *fn;     /* value callback */
  PhysLaminarBCFn *fn_dot; /* time derivative callback (may be NULL) */
  void            *fn_ctx;
  void            *fn_dot_ctx;
  PetscInt         comp; /* which solution component this adapter is wired for */
} Seg_BCAdapter;

/* Discrete operators of the coupled system (13) of the theory guide. Built once from the solution
   DM of the attached Phys, and reused by every step of the segregated solve. */
typedef struct {
  PetscInt dim;   /* spatial dimension of the solution DM */
  PetscInt c_vel; /* first velocity component (element) */
  PetscInt c_p;   /* pressure component (element) */
  PetscInt c_U;   /* face-normal velocity component (face) */

  /* BC adapters: [comp][face] — created during setup to bridge PhysLaminarBCFn to FlucaFDBCValueFn */
  Seg_BCAdapter bc_adapters[SEG_OPS_MAX_DIM + 1][SEG_OPS_MAX_FACES];

  FlucaFD fd_laplacian[SEG_OPS_MAX_DIM]; /* sum_e d/dx_e(-mu * d(u_d)/dx_e) */
  FlucaFD fd_grad_p[SEG_OPS_MAX_DIM];    /* dp/dx_d */

  /* Momentum rows of the coupled system (13) */
  IS      is_vel;                                          /* velocity entries of the solution vector */
  Vec     zero;                                            /* zero solution vector: evaluates boundary (affine) parts */
  DM      dm_face;                                         /* one DOF per face */
  Vec     ubar[SEG_OPS_MAX_DIM];                           /* ubar_d^n: u_d^n linearly interpolated to every face */
  FlucaFD fd_interp_vel[SEG_OPS_MAX_DIM][SEG_OPS_MAX_DIM]; /* [d][e]: u_d -> faces normal to e, onto dm_face */
  FlucaFD fd_visc[SEG_OPS_MAX_DIM];                        /* (dt/(2 rho)) fd_laplacian[d] = -(dt/2) nu lap(u_d) */
  FlucaFD fd_conv[SEG_OPS_MAX_DIM];                        /* (dt/2) sum_e d/dx_e(ubar_d U_e^n + ubar_d^n ubar_e) */
  FlucaFD fd_conv_U[SEG_OPS_MAX_DIM][SEG_OPS_MAX_DIM];     /* [d][e]: ubar_d on faces normal to e, times U_e^n */
  FlucaFD fd_conv_ubar[SEG_OPS_MAX_DIM][SEG_OPS_MAX_DIM];  /* [d][e]: ubar_e on faces normal to e, times ubar_d^n */
  FlucaFD fd_grad[SEG_OPS_MAX_DIM];                        /* (dt/rho) fd_grad_p[d]: the operator G */

  /* Coupling rows of the coupled system (13) */
  FlucaFD   fd_T[SEG_OPS_MAX_DIM];     /* T: u_e -> faces normal to e, fourth-order, interior cells only */
  FlucaFD   fd_negT[SEG_OPS_MAX_DIM];  /* -T */
  FlucaFD   fd_bface[SEG_OPS_MAX_DIM]; /* boundary-face right-hand side: applied to zero it is u_b . n */
  Mat       negR;                      /* -R = -(T G_c - G^st), unscaled; the step scales it by dt/rho */
  FlucaFD   fd_D;                      /* D: sum_e d/dx_e(U_e) into the pressure rows */
  PetscInt  nbface;                    /* locally owned boundary-face rows */
  PetscInt *bface;                     /* their local indices on the solution DM */
} Seg_Ops;

/* Defined in utils/ops/segops.c */
FLUCA_INTERN PetscErrorCode SegOpsBuild_Internal(Seg);
FLUCA_INTERN PetscErrorCode SegOpsDestroy_Internal(Seg);
