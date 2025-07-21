#pragma once

#include <fluca/private/nsimpl.h>
#include <petscksp.h>

typedef struct {
  NS       ns;
  PetscInt dim;
} KSPVCtx;

typedef struct {
  Vec v[3];        /* velocity at n */
  Vec v_star[3];   /* intermediate velocity between n and n+1 */
  Vec N[3];        /* convection at n */
  Vec N_prev[3];   /* convection at n-1 */
  Vec fv;          /* face velocity at n */
  Vec fv_star;     /* intermediate face velocity between n and n+1 */
  Vec p;           /* pressure at n */
  Vec p_half;      /* pressure at n-1/2 */
  Vec p_prime;     /* pressure correction between n-1/2 and n+1/2 */
  Vec p_half_prev; /* pressure at previous half time step n-3/2 */

  Mat grad_p[3];       /* gradient operators for pressure */
  Mat grad_p_prime[3]; /* gradient operators for pressure correction */
  Mat helm_v;          /* helmholtz operator for velocity */
  Mat lap_p_prime;     /* laplacian operator for pressure correction */
  Mat interp_v[3];     /* interpolation operator from velocity to face velocity */

  Mat grad_f; /* face gradient operator; computes gradient at face from cell-centered values (except for boundaries) */
  Mat div_f;  /* face divergence operator; computes divergence at cell center from face values */

  KSP     kspv[3];    /* KSP to solve intermediate velocities */
  KSPVCtx kspvctx[3]; /* context for intermediate velocity KSP */
  KSP     kspp;       /* KSP to solve pressure correction */
} NS_FSM;

FLUCA_INTERN PetscErrorCode NSFSMComputeSpatialOperators2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMSetKSPComputeFunctions2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMIterate2d_Cart_Internal(NS);

FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_CGNS_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSLoadSolutionCGNS_FSM_Cart_Internal(NS, PetscInt);
