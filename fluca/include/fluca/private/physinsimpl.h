#pragma once

#include <fluca/private/physimpl.h>
#include <flucafd.h>

#define PHYS_INS_MAX_DIM   3
#define PHYS_INS_MAX_FACES (2 * PHYS_INS_MAX_DIM)

/* BC adapter: converts PhysINSBCFn -> FlucaFDBoundaryConditionFn by binding a component */
typedef struct {
  PhysINSBCFn *fn;
  void        *ctx;
  PetscInt     comp;
} PhysINSBCAdapter;

typedef struct {
  PetscReal rho;
  PetscReal mu;

  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysINSBC        bcs[PHYS_INS_MAX_FACES];
  PhysINSBCAdapter bc_adapters[PHYS_INS_MAX_DIM * PHYS_INS_MAX_FACES]; /* [d * MAX_FACES + f] */

  /* FlucaFD operators (applied separately due to different BC types) */
  FlucaFD fd_laplacian[PHYS_INS_MAX_DIM]; /* viscous Laplacian per velocity direction */
  FlucaFD fd_grad_p[PHYS_INS_MAX_DIM];    /* pressure gradient per velocity direction */
  FlucaFD fd_div[PHYS_INS_MAX_DIM];       /* divergence per direction */

  /* Convection operators: C_d = sum_e d/dx_e(u_e * u_d) */
  FlucaFD fd_conv[PHYS_INS_MAX_DIM];                        /* summed convection per velocity dir */
  FlucaFD fd_tvd[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];       /* TVD interp: [d][e] = u_d along e */
  FlucaFD fd_scale_vel[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* velocity scaling: [d][e] */
  FlucaFD fd_conv_comp[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* composed conv: [d][e] */
  FlucaFD fd_interp[PHYS_INS_MAX_DIM];                      /* cell-to-face interpolation per dir */
  DM      dm_face[PHYS_INS_MAX_DIM];                        /* face DMs for velocity scaling */
  Vec     vel_face[PHYS_INS_MAX_DIM];                       /* face velocity vectors */

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
