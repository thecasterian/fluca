#pragma once

#include <fluca/private/segimpl.h>
#include <fluca/private/segopsimpl.h>
#include <flucaphys.h>

/* Row blocks of the coupled system, in the order SegSetUp_CNLinear() fills seg->fields */
typedef enum {
  SEG_CNLINEAR_FIELD_VELOCITY,
  SEG_CNLINEAR_FIELD_FACE_VELOCITY,
  SEG_CNLINEAR_FIELD_PRESSURE,
  SEG_CNLINEAR_NUM_FIELDS,
} SegCNLinearField;

typedef struct {
  Mat          M; /* coupled system (13) on the solution DM */
  Mat          P; /* MATNEST carrying the field index sets that PCABF reads */
  MatNullSpace nullspace;
  Vec          f, x;
  Seg_Ops      ops; /* discrete operators of the coupled system */
} Seg_CNLinear;

/* Defined in impls/cnlinear/cnlinearsystem.c. Public visibility: the tests link against the shared
   library and assemble the coupled system directly. */
FLUCA_EXTERN PetscErrorCode SegCNLinearComputeMomentumSystem_Internal(Seg, PetscReal, PetscReal, Vec, Mat, Vec);
FLUCA_EXTERN PetscErrorCode SegCNLinearComputeCouplingSystem_Internal(Seg, PetscReal, PetscReal, Mat, Vec);
