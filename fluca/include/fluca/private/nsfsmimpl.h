#pragma once

#include <fluca/private/nsimpl.h>
#include <petscksp.h>

typedef struct {
  Vec v[3];        /* velocity */
  Vec v_star[3];   /* intermediate velocity */
  Vec N[3];        /* momentum */
  Vec N_prev[3];   /* momentum at previous time step (n-1) */
  Vec fv;          /* face velocity */
  Vec fv_star;     /* intermediate face velocity */
  Vec p;           /* pressure */
  Vec p_half;      /* pressure at half time step (n+1/2) */
  Vec p_prime;     /* pressure correction */
  Vec p_half_prev; /* pressure at previous half time step (n-1/2) */

  KSP kspv[3]; /* for solving intermediate velocities */
  KSP kspp;    /* for solving pressure correction */
} NS_FSM;

FLUCA_INTERN PetscErrorCode NSFSMCalculateConvection2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMCalculatePressureCorrection2d_Cart_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFSMUpdate2d_Cart_Internal(NS);

FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSViewSolution_FSM_Cart_CGNS_Internal(NS, PetscViewer);
FLUCA_INTERN PetscErrorCode NSLoadSolutionCGNS_FSM_Cart_Internal(NS, PetscInt);
