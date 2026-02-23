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
  FlucaFD fd_pstab;                       /* pressure stabilization */

  /* Solver data */
  Mat          J;
  IS           is_vel;
  IS           is_p;
  MatNullSpace nullspace;
  Vec          temp;
  PetscReal    alpha; /* pressure stabilization coefficient */
  PetscBool    has_pressure_outlet;
} Phys_INS;

FLUCA_INTERN PetscErrorCode PhysINSBuildOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysINSDestroyOperators_Internal(Phys);
FLUCA_INTERN PetscErrorCode PhysSetUpTS_INS(Phys, TS);
