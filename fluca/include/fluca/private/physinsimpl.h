#pragma once

#include <fluca/private/physimpl.h>
#include <flucafd.h>

#define PHYS_INS_MAX_DIM   3
#define PHYS_INS_MAX_FACES (2 * PHYS_INS_MAX_DIM)

typedef struct {
  PetscReal rho;
  PetscReal mu;

  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysINSBC bcs[PHYS_INS_MAX_FACES];

  /* FlucaFD operators */
  FlucaFD fd_laplacian[PHYS_INS_MAX_DIM]; /* viscous Laplacian per velocity direction */
  FlucaFD fd_grad_p[PHYS_INS_MAX_DIM];    /* pressure gradient per velocity direction */
  FlucaFD fd_div;                         /* rho * div(interp(u)) — Sum over directions */
  FlucaFD fd_pstab;                       /* dt * (DTG - DG^st)(p) pressure stabilization */

  /* Convection operators: C_d = sum_e d/dx_e(mass_flux_e * u_d) */
  FlucaFD fd_conv[PHYS_INS_MAX_DIM];                        /* summed convection per velocity dir */
  FlucaFD fd_tvd[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];       /* TVD interp: [d][e] = u_d along e */
  FlucaFD fd_scale_vel[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* velocity scaling: [d][e] */
  FlucaFD fd_conv_comp[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* composed conv: [d][e] */
  FlucaFD fd_interp[PHYS_INS_MAX_DIM];                      /* cell-to-face interpolation per dir */
  DM      dm_face[PHYS_INS_MAX_DIM];                        /* face DMs for velocity scaling */
  Vec     mass_flux_face[PHYS_INS_MAX_DIM];                 /* face mass flux vectors */

  /* Solver data */
  Mat          J;
  IS           is_vel;
  IS           is_p;
  MatNullSpace nullspace;
  Vec          temp;
  PetscBool    has_pressure_outlet;
} Phys_INS;

FLUCA_INTERN PetscErrorCode PhysINSBuildOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysINSDestroyOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysSetUpTS_INS(Phys, TS);
FLUCA_INTERN PetscErrorCode PhysComputeIFunction_INS(Phys, PetscReal, Vec, Vec, Vec);
FLUCA_INTERN PetscErrorCode PhysComputeIJacobian_INS(Phys, PetscReal, Vec, Vec, PetscReal, Mat, Mat);
