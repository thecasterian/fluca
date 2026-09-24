#pragma once

#include <flucafd.h>
#include <flucaphys.h>

/* Adapter to bridge PhysLaminarBCFn (has comp) to FlucaFDBCValueFn (no comp) */
typedef struct {
  PhysLaminarBCFn *fn;     /* value callback */
  PhysLaminarBCFn *fn_dot; /* time derivative callback (may be NULL) */
  void            *fn_ctx;
  void            *fn_dot_ctx;
  PetscInt         comp; /* which solution component this adapter is wired for */
} Seg_BCAdapter;

/* Spatial operators of the coupled system (13) of the theory guide. Built once from the solution DM
   of the attached Phys and reused by every step of the segregated solve.

   The operators are independent of the time discretization: none of them carries a time step, and a
   Seg subtype builds its own time-discrete operators on top of them. They are built only from a
   PHYSLAMINAR Phys, whose boundary conditions the adapter above forwards. */
typedef struct {
  PetscInt dim;   /* spatial dimension of the solution DM */
  PetscInt c_vel; /* first velocity component (element) */
  PetscInt c_p;   /* pressure component (element) */
  PetscInt c_U;   /* face-normal velocity component (face) */

  /* BC adapters: [comp][face] — created during setup to bridge PhysLaminarBCFn to FlucaFDBCValueFn */
  Seg_BCAdapter bc_adapters[FLUCA_MAX_DIM + 1][FLUCA_MAX_FACES];

  FlucaFD fd_laplacian[FLUCA_MAX_DIM];            /* sum_e d/dx_e(-mu * d(u_d)/dx_e) */
  FlucaFD fd_negmu[FLUCA_MAX_DIM][FLUCA_MAX_DIM]; /* [d][e]: -mu d(u_d)/dx_e, the scale inside fd_laplacian[d] */
  FlucaFD fd_grad_p[FLUCA_MAX_DIM];               /* dp/dx_d */

  Vec zero; /* zero solution vector: evaluates boundary (affine) parts */

  /* Linearization state of the momentum rows of the coupled system (13) */
  FlucaFD fd_interp_vel[FLUCA_MAX_DIM][FLUCA_MAX_DIM]; /* [d][e]: u_d -> faces normal to e, onto component 0 of a one-DOF-per-face DM */

  /* Coupling rows of the coupled system (13) */
  FlucaFD   fd_T[FLUCA_MAX_DIM];     /* T: u_e -> faces normal to e, fourth-order, interior cells only */
  FlucaFD   fd_negT[FLUCA_MAX_DIM];  /* -T */
  FlucaFD   fd_bface[FLUCA_MAX_DIM]; /* boundary-face right-hand side: applied to zero it is u_b . n */
  Mat       negR;                    /* -R = -(T G_c - G^st), unscaled; the step scales it by dt/rho */
  FlucaFD   fd_D;                    /* D: sum_e d/dx_e(U_e) into the pressure rows */
  PetscInt  nbface;                  /* locally owned boundary-face rows */
  PetscInt *bface;                   /* their local indices on the solution DM */
} SegSpatialOps;

/* Defined in utils/ops/segops.c */
FLUCA_INTERN PetscErrorCode SegSpatialOpsBuild_Internal(Phys, SegSpatialOps *);
FLUCA_INTERN PetscErrorCode SegSpatialOpsSetVelocityBCs_Internal(Phys, SegSpatialOps *, FlucaFD, PetscInt);
FLUCA_INTERN PetscErrorCode SegSpatialOpsUpdateProperties_Internal(Phys, SegSpatialOps *);
FLUCA_INTERN PetscErrorCode SegSpatialOpsDestroy_Internal(SegSpatialOps *);
