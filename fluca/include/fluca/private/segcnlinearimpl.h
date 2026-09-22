#pragma once

#include <fluca/private/physimpl.h>
#include <fluca/private/segimpl.h>
#include <fluca/private/segopsimpl.h>

typedef struct {
  Mat          M;     /* coupled system (13) on the solution DM */
  Mat          P;     /* MATNEST carrying the field index sets that PCABF reads */
  IS           is[3]; /* velocity, face velocity, pressure */
  MatNullSpace nullspace;
  Vec          f, x;
  Seg_Ops      ops; /* discrete operators of the coupled system */
} Seg_CNLinear;

/* Defined in impls/cnlinear/cnlinearsystem.c. Public visibility: the tests link against the shared
   library and assemble the coupled system directly. */
FLUCA_EXTERN PetscErrorCode SegCNLinearComputeMomentumSystem_Internal(Seg, PetscReal, PetscReal, Vec, Mat, Vec);
FLUCA_EXTERN PetscErrorCode SegCNLinearComputeCouplingSystem_Internal(Seg, PetscReal, PetscReal, Mat, Vec);
