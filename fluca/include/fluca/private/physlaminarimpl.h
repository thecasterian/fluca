#pragma once

#include <fluca/private/physimpl.h>
#include <flucafd.h>

#define PHYS_LAMINAR_MAX_DIM   3
#define PHYS_LAMINAR_MAX_FACES (2 * PHYS_LAMINAR_MAX_DIM)

/* Adapter to bridge PhysLaminarBCFn (has comp) to FlucaFDBCValueFn (no comp) */
typedef struct {
  PhysLaminarBCFn *fn;     /* value callback */
  PhysLaminarBCFn *fn_dot; /* time derivative callback (may be NULL) */
  void            *fn_ctx;
  void            *fn_dot_ctx;
  PetscInt         comp; /* which solution component this adapter is wired for */
} PhysLaminar_BCAdapter;

typedef struct {
  PetscInt c_vel; /* first velocity component (element) */
  PetscInt c_p;   /* pressure component (element) */
  PetscInt c_U;   /* face-normal velocity component (face) */

  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysLaminarBC bcs[PHYS_LAMINAR_MAX_FACES];

  /* BC adapters: [comp][face] — created during setup to bridge PhysLaminarBCFn to FlucaFDBCValueFn */
  PhysLaminar_BCAdapter bc_adapters[PHYS_LAMINAR_MAX_DIM + 1][PHYS_LAMINAR_MAX_FACES];

  FlucaFD fd_laplacian[PHYS_LAMINAR_MAX_DIM]; /* sum_e d/dx_e(-mu * d(u_d)/dx_e) */
  FlucaFD fd_grad_p[PHYS_LAMINAR_MAX_DIM];    /* dp/dx_d */

  /* Momentum rows of the coupled system (13) */
  IS      is_vel;                                                    /* velocity entries of the solution vector */
  Vec     zero;                                                      /* zero solution vector: evaluates boundary (affine) parts */
  DM      dm_face;                                                   /* one DOF per face */
  Vec     ubar[PHYS_LAMINAR_MAX_DIM];                                /* ubar_d^n: u_d^n linearly interpolated to every face */
  FlucaFD fd_interp_vel[PHYS_LAMINAR_MAX_DIM][PHYS_LAMINAR_MAX_DIM]; /* [d][e]: u_d -> faces normal to e, onto dm_face */
  FlucaFD fd_visc[PHYS_LAMINAR_MAX_DIM];                             /* (dt/(2 rho)) fd_laplacian[d] = -(dt/2) nu lap(u_d) */
  FlucaFD fd_conv[PHYS_LAMINAR_MAX_DIM];                             /* (dt/2) sum_e d/dx_e(ubar_d U_e^n + ubar_d^n ubar_e) */
  FlucaFD fd_conv_U[PHYS_LAMINAR_MAX_DIM][PHYS_LAMINAR_MAX_DIM];     /* [d][e]: ubar_d on faces normal to e, times U_e^n */
  FlucaFD fd_conv_ubar[PHYS_LAMINAR_MAX_DIM][PHYS_LAMINAR_MAX_DIM];  /* [d][e]: ubar_e on faces normal to e, times ubar_d^n */
  FlucaFD fd_grad[PHYS_LAMINAR_MAX_DIM];                             /* (dt/rho) fd_grad_p[d]: the operator G */

  /* Coupling rows of the coupled system (13) */
  FlucaFD   fd_T[PHYS_LAMINAR_MAX_DIM];     /* T: u_e -> faces normal to e, fourth-order, interior cells only */
  FlucaFD   fd_negT[PHYS_LAMINAR_MAX_DIM];  /* -T */
  FlucaFD   fd_bface[PHYS_LAMINAR_MAX_DIM]; /* boundary-face right-hand side: applied to zero it is u_b . n */
  Mat       negR;                           /* -R = -(T G_c - G^st), unscaled; the step scales it by dt/rho */
  FlucaFD   fd_D;                           /* D: sum_e d/dx_e(U_e) into the pressure rows */
  PetscInt  nbface;                         /* locally owned boundary-face rows */
  PetscInt *bface;                          /* their local indices on the solution DM */
} Phys_Laminar;

/* Internal functions defined in laminarops.c */
FLUCA_INTERN PetscErrorCode PhysLaminarBuildOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysLaminarDestroyOperators_Internal(Phys);

/* Defined in laminarsystem.c */
FLUCA_INTERN PetscErrorCode PhysComputeMomentumSystem_Laminar(Phys, PetscReal, PetscReal, Vec, Mat, Vec);
FLUCA_INTERN PetscErrorCode PhysComputeCouplingSystem_Laminar(Phys, PetscReal, PetscReal, Mat, Vec);
