#pragma once

#include <fluca/private/nsimpl.h>
#include <petscksp.h>

typedef struct {
  IS vis; /* index set of velocity component */
  IS Vis; /* index set of face normal velocity component */
  IS pis; /* index set of pressure component */

  Mat A;   /* operator of momentum equation */
  Mat T;   /* veloctiy interpolation operator */
  Mat G;   /* pressure gradient operator */
  Mat Gst; /* staggered pressure gradient operator */
  Mat Dst; /* staggered veloctiy divergence operator */
  Mat Lst; /* staggered pressure laplacian operator */

  KSP kspv; /* KSP to solve velocity equation */
  KSP kspp; /* KSP to solve pressure equation */

  MatNullSpace nullspace;
} NSFSMPCCtx;

typedef struct {
  Vec v_star;      /* intermediate velocity between n and n+1 */
  Vec V_star;      /* intermediate face velocity between n and n+1 */
  Vec N;           /* convection at n */
  Vec N_prev;      /* convection at n-1 */
  Vec p_half;      /* pressure at n-1/2 */
  Vec p_prime;     /* pressure correction between n-1/2 and n+1/2 */
  Vec p_half_prev; /* pressure at previous half time step n-3/2 */

  Mat Gp;     /* pressure gradient operator */
  Mat Lv;     /* velocity laplacian operator */
  Mat Tv;     /* velocity interpolation operator for divergence */
  Mat Gstp;   /* staggered pressure gradient operator */
  Mat Dstv;   /* staggered velocity divergence operator */
  Mat TvN[3]; /* velocity interpolation operators for convection */

  KSP kspv; /* KSP to solve intermediate velocities */
  KSP kspp; /* KSP to solve pressure correction */
} NS_FSM;

FLUCA_INTERN PetscErrorCode NSFSMFormFunction_Cart_Internal(SNES, Vec, Vec, void *);
FLUCA_INTERN PetscErrorCode NSFSMFormJacobian_Cart_Internal(SNES, Vec, Mat, Mat, void *);
FLUCA_INTERN PetscErrorCode NSFSMPicardComputeFunction_Internal(SNES, Vec, Vec, void *);
FLUCA_INTERN PetscErrorCode NSFSMFormInitialGuess_Internal(SNES, Vec, void *);

FLUCA_INTERN PetscErrorCode NSFSMPCApply_Internal(PC, Vec, Vec);
FLUCA_INTERN PetscErrorCode NSFSMPCDestroy_Internal(PC);

FLUCA_INTERN PetscErrorCode NSFSMComputeSpatialOperators2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMSetKSPComputeFunctions2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMIterate2d_Cart_Internal(NS);

FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_CGNS_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSLoadSolutionCGNS_FSM_Cart_Internal(NS, PetscInt);
