#pragma once

#include <fluca/private/nsimpl.h>
#include <petscksp.h>

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
  Mat grad_p_prime_f;  /* gradient operators at face for pressure correction */
  Mat helm_v;          /* helmholtz operator for velocity */
  Mat lap_p_prime;     /* laplacian operator for pressure correction */
  Mat interp_v[3];     /* interpolation operator from velocity to face velocity */
  Mat div_fv;          /* divergence operator for face velocity */

  KSP kspv[3]; /* for solving intermediate velocities */
  KSP kspp;    /* for solving pressure correction */
} NS_FSM;

FLUCA_INTERN PetscErrorCode NSFSMComputePressureGradientOperators2d_Cart_Internal(DM, const NSBoundaryCondition *, Mat[]);
FLUCA_INTERN PetscErrorCode NSFSMComputePressureCorrectionGradientOperators2d_Cart_Internal(DM, const NSBoundaryCondition *, Mat[]);
FLUCA_INTERN PetscErrorCode NSFSMComputePressureCorrectionFaceGradientOperator2d_Cart_Internal(DM, DM, const NSBoundaryCondition *, Mat);
FLUCA_INTERN PetscErrorCode NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(DM, const NSBoundaryCondition *, PetscScalar, PetscScalar, Mat);
FLUCA_INTERN PetscErrorCode NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(DM, PetscInt, const NSBoundaryCondition *, PetscReal, PetscReal, InsertMode, Vec);
FLUCA_INTERN PetscErrorCode NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(DM, const NSBoundaryCondition *, Mat);
FLUCA_INTERN PetscErrorCode NSFSMComputeVelocityInterpolationOperators2d_Cart_Internal(DM, DM, const NSBoundaryCondition *, Mat[]);
FLUCA_INTERN PetscErrorCode NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(DM, DM, PetscInt, PetscInt, const NSBoundaryCondition *, PetscReal, InsertMode, Vec);
FLUCA_INTERN PetscErrorCode NSFSMComputeFaceVelocityDivergenceOperator2d_Cart_Internal(DM, const NSBoundaryCondition *, Mat);

FLUCA_INTERN PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMCalculatePressureCorrection2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMUpdate2d_Cart_Internal(NS);

FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_CGNS_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSLoadSolutionCGNS_FSM_Cart_Internal(NS, PetscInt);
