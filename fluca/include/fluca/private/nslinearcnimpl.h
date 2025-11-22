#pragma once

#include <fluca/private/nsimpl.h>

typedef struct {
  Vec v0interp; /* velocity at n interpolated on face */
  Vec phalf;    /* pressure at n-1/2 */

  Mat       B;
  PetscBool Bcomputed;
} NS_CNLinear;

FLUCA_INTERN PetscErrorCode NSStep_CNLinear_Cart2d_Internal(NS);
FLUCA_INTERN PetscErrorCode NSFormJacobian_CNLinear_Cart2d_Internal(NS, Vec, Mat, NSFormJacobianType);
FLUCA_INTERN PetscErrorCode NSFormFunction_CNLinear_Cart2d_Internal(NS, Vec, Vec);
