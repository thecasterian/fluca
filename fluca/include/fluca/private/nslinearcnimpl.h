#pragma once

#include <fluca/private/nsimpl.h>

typedef struct {
  Vec v0interp; /* velocity at n interpolated on face */
  Vec phalf;    /* pressure at n-1/2 */

  Mat       B;
  PetscBool Bcomputed;
} NS_CNLinear;

FLUCA_INTERN PetscErrorCode NSCNLinearFormFunction_Cart_Internal(SNES, Vec, Vec, void *);
FLUCA_INTERN PetscErrorCode NSCNLinearFormJacobian_Cart_Internal(SNES, Vec, Mat, Mat, void *);

FLUCA_INTERN PetscErrorCode NSCNLinearIterate2d_Cart_Internal(NS);
