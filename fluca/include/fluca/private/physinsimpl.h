#pragma once

#include <fluca/private/physimpl.h>
#include <flucafd.h>

#define PHYS_INS_MAX_DIM   3
#define PHYS_INS_MAX_FACES (2 * PHYS_INS_MAX_DIM)

/* Adapter to bridge PhysINSBCFn (has comp) to FlucaFDBCValueFn (no comp) */
typedef struct {
  PhysINSBCFn *fn;     /* value callback */
  PhysINSBCFn *fn_dot; /* time derivative callback (may be NULL) */
  void        *fn_ctx;
  void        *fn_dot_ctx;
  PetscInt     comp; /* which solution component this adapter is wired for */
} PhysINS_BCAdapter;

typedef struct {
  PetscInt c_vel; /* first velocity component (element) */
  PetscInt c_p;   /* pressure component (element) */
  PetscInt c_U;   /* face-normal velocity component (face) */

  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysINSBC bcs[PHYS_INS_MAX_FACES];

  /* BC adapters: [comp][face] — created during setup to bridge PhysINSBCFn to FlucaFDBCValueFn */
  PhysINS_BCAdapter bc_adapters[PHYS_INS_MAX_DIM + 1][PHYS_INS_MAX_FACES];

  FlucaFD fd_laplacian[PHYS_INS_MAX_DIM]; /* sum_e d/dx_e(-mu * d(u_d)/dx_e) */
  FlucaFD fd_grad_p[PHYS_INS_MAX_DIM];    /* dp/dx_d */

  /* Momentum rows of the coupled system (13) */
  IS      is_vel;                                            /* velocity entries of the solution vector */
  Vec     zero;                                              /* zero solution vector: evaluates boundary (affine) parts */
  DM      dm_face;                                           /* one DOF per face */
  Vec     ubar[PHYS_INS_MAX_DIM];                            /* ubar_d^n: u_d^n linearly interpolated to every face */
  FlucaFD fd_interp_vel[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* [d][e]: u_d -> faces normal to e, onto dm_face */
  FlucaFD fd_visc[PHYS_INS_MAX_DIM];                         /* (dt/(2 rho)) fd_laplacian[d] = -(dt/2) nu lap(u_d) */
  FlucaFD fd_conv[PHYS_INS_MAX_DIM];                         /* (dt/2) sum_e d/dx_e(ubar_d U_e^n + ubar_d^n ubar_e) */
  FlucaFD fd_conv_U[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];     /* [d][e]: ubar_d on faces normal to e, times U_e^n */
  FlucaFD fd_conv_ubar[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];  /* [d][e]: ubar_e on faces normal to e, times ubar_d^n */
  FlucaFD fd_grad[PHYS_INS_MAX_DIM];                         /* (dt/rho) fd_grad_p[d]: the operator G */

  /* Coupling rows of the coupled system (13) */
  FlucaFD   fd_T[PHYS_INS_MAX_DIM];     /* T: u_e -> faces normal to e, fourth-order, interior cells only */
  FlucaFD   fd_negT[PHYS_INS_MAX_DIM];  /* -T */
  FlucaFD   fd_bface[PHYS_INS_MAX_DIM]; /* boundary-face right-hand side: applied to zero it is u_b . n */
  Mat       negR;                       /* -R = -(T G_c - G^st), unscaled; the step scales it by dt/rho */
  FlucaFD   fd_D;                       /* D: sum_e d/dx_e(U_e) into the pressure rows */
  PetscInt  nbface;                     /* locally owned boundary-face rows */
  PetscInt *bface;                      /* their local indices on the solution DM */
} Phys_INS;

/* Internal functions defined in insops.c */
FLUCA_INTERN PetscErrorCode PhysINSBuildOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysINSDestroyOperators_Internal(Phys);

/* Defined in inssystem.c */
FLUCA_INTERN PetscErrorCode PhysComputeMomentumSystem_INS(Phys, PetscReal, PetscReal, Vec, Mat, Vec);
FLUCA_INTERN PetscErrorCode PhysComputeCouplingSystem_INS(Phys, PetscReal, PetscReal, Mat, Vec);
