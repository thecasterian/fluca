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
  PetscReal rho; /* density */
  PetscReal mu;  /* dynamic viscosity */

  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysINSBC bcs[PHYS_INS_MAX_FACES];

  /* BC adapters: [comp][face] — created during setup to bridge PhysINSBCFn to FlucaFDBCValueFn */
  PhysINS_BCAdapter bc_adapters[PHYS_INS_MAX_DIM + 1][PHYS_INS_MAX_FACES];

  /* Implicit operators */
  FlucaFD fd_diff[PHYS_INS_MAX_DIM];   /* F_diff_d = sum_e d/dx_e(nu * d(u_d)/dx_e) */
  FlucaFD fd_grad_p[PHYS_INS_MAX_DIM]; /* G_d = (1/rho) * dp/dx_d */
  FlucaFD fd_div;                      /* D = rho * sum_d d(u_d)/dx_d */
  FlucaFD fd_pres_lap;                 /* L = sum_d d^2p/dx_d^2 (compact pressure Laplacian) */

  /* Explicit operators */
  FlucaFD fd_conv[PHYS_INS_MAX_DIM];                            /* sum_e d/dx_e(F_e * TVD_e(u_d)) */
  FlucaFD fd_tvd[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];           /* TVD_e(u_d) = TVD interpolate u_d to face e */
  FlucaFD fd_momentum_flux[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* F_e * TVD_e(u_d) */
  FlucaFD fd_interp[PHYS_INS_MAX_DIM];                          /* interp_d(u_d) = linearly interpolate u_d to face d */
  DM      dm_face;                                              /* single face DM: 1 DOF per face in each direction */
  Vec     mass_flux;                                            /* F_d = rho * interp_d(u_d) on dm_face for all d */

  /* Solver data */
  IS  is_vel;
  IS  is_p;
  IS  is_comp[PHYS_INS_MAX_DIM]; /* per-component velocity IS (for SRK) */
  Vec temp;                      /* work vector */
} Phys_INS;

/* Internal functions defined in insops.c */
FLUCA_INTERN PetscErrorCode PhysINSBuildOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysINSDestroyOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysINSCreateSolverData_Internal(Phys);

/* Ops defined in insops.c, wired from ins.c */
FLUCA_INTERN PetscErrorCode PhysSetUpSeg_INS(Phys, Seg);
