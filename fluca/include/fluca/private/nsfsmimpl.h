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
  Mat D;   /* face-normal veloctiy divergence operator */
  Mat Lst; /* operator of pressure equation: Lst = D * Gst */

  Vec divvstar;
  Vec gradpcorr;
  Vec gradstpcorr;

  KSP kspv; /* KSP to solve velocity equation */
  KSP kspp; /* KSP to solve pressure equation */

  MatNullSpace nullspace;
} NSFSMPCCtx;

typedef struct {
  Vec v0interp; /* velocity at n interpolated on face */
  Vec phalf;    /* pressure at n-1/2 */

  Mat       B;
  PetscBool Bcomputed;
} NS_FSM;

FLUCA_INTERN PetscErrorCode NSFSMFormFunction_Cart_Internal(SNES, Vec, Vec, void *);
FLUCA_INTERN PetscErrorCode NSFSMFormJacobian_Cart_Internal(SNES, Vec, Mat, Mat, void *);

FLUCA_INTERN PetscErrorCode NSFSMIterate2d_Cart_Internal(NS);
