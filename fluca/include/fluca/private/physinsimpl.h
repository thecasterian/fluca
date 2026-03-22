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
  PetscReal rho;   /* density */
  PetscReal mu;    /* dynamic viscosity */
  PetscReal alpha; /* constraint feedback parameter = 1/dt */

  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysINSBC bcs[PHYS_INS_MAX_FACES];

  /* BC adapters: [comp][face] — created during setup to bridge PhysINSBCFn to FlucaFDBCValueFn */
  PhysINS_BCAdapter bc_adapters[PHYS_INS_MAX_DIM + 1][PHYS_INS_MAX_FACES];

  /* Implicit operators */
  FlucaFD fd_laplacian[PHYS_INS_MAX_DIM]; /* sum_e d/dx_e(-mu * d(u_d)/dx_e) */
  FlucaFD fd_grad_p[PHYS_INS_MAX_DIM];    /* dp/dx_d */
  FlucaFD fd_div;                         /* rho * sum_d d/dx_d(interp_d(u_d)) */
  FlucaFD fd_pstab;                       /* sigma_0 * S(p); sigma_0 = dt, S(p) = D(G(p)) - L(p) = pressure stabilization operator */

  /* Explicit operators */
  FlucaFD fd_conv[PHYS_INS_MAX_DIM];                            /* sum_e d/dx_e(F_e * TVD_e(u_d)) */
  FlucaFD fd_tvd[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];           /* TVD_e(u_d) = TVD interpolate u_d to face e */
  FlucaFD fd_momentum_flux[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* F_e * TVD_e(u_d) */
  FlucaFD fd_interp[PHYS_INS_MAX_DIM];                          /* interp_d(u_d) = linearly interpolate u_d to face d */
  DM      dm_face;                                              /* single face DM: 1 DOF per face in each direction */
  Vec     mass_flux;                                            /* F_d = rho * interp_d(u_d) on dm_face for all d */

  /* Solver data */
  Mat          J;     /* IJacobian matrix */
  Mat          J_rhs; /* RHSJacobian matrix (Picard convection) */
  IS           is_vel;
  IS           is_p;
  MatNullSpace nullspace;
  Vec          temp;       /* work vector for IFunction computation */
  PetscReal    dt_current; /* current dt for sigma_0 update detection */
  PetscBool    has_pressure_outlet;
} Phys_INS;

/* Internal functions defined in insops.c */
FLUCA_INTERN PetscErrorCode PhysINSBuildOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysINSDestroyOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysINSCreateSolverData_Internal(Phys);

/* Ops defined in insops.c, wired from ins.c */
FLUCA_INTERN PetscErrorCode PhysSetUpTS_INS(Phys, TS);
FLUCA_INTERN PetscErrorCode PhysComputeIFunction_INS(Phys, PetscReal, Vec, Vec, Vec);
FLUCA_INTERN PetscErrorCode PhysComputeIJacobian_INS(Phys, PetscReal, Vec, Vec, PetscReal, Mat, Mat);
FLUCA_INTERN PetscErrorCode PhysComputeRHSFunction_INS(Phys, PetscReal, Vec, Vec);
