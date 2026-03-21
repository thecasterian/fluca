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

  /* FlucaFD operators (implicit part) */
  FlucaFD fd_laplacian[PHYS_INS_MAX_DIM]; /* -mu * nabla^2 u_d per velocity direction */
  FlucaFD fd_grad_p[PHYS_INS_MAX_DIM];    /* dp/dx_d per velocity direction */
  FlucaFD fd_div;                         /* rho * div(interp(u)) — single Sum over directions */
  FlucaFD fd_pstab;                       /* sigma_0 * S(p) pressure stabilization */

  /* FlucaFD operators (explicit part — convection) */
  /* C_d = sum_e d/dx_e(F_e * u_d_TVD) where F_e = rho * u_e */
  FlucaFD fd_conv[PHYS_INS_MAX_DIM];                        /* summed convection per velocity dir */
  FlucaFD fd_tvd[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];       /* TVD interp: [d][e] = u_d along e */
  FlucaFD fd_scale_vel[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* mass flux scaling: [d][e] */
  FlucaFD fd_conv_comp[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* composed conv: [d][e] */
  FlucaFD fd_interp[PHYS_INS_MAX_DIM];                      /* cell-to-face interpolation per dir */
  DM      dm_face[PHYS_INS_MAX_DIM];                        /* face DMs for mass flux */
  Vec     mass_flux_face[PHYS_INS_MAX_DIM];                 /* face mass flux vectors (rho * u) */

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
