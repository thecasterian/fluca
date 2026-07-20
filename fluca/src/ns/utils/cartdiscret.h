#include <flucasys.h>
#include <petscdmstag.h>

typedef enum {
  DIR_X,
  DIR_Y,
  DIR_Z,
} Direction;

FLUCA_INTERN PetscErrorCode NSComputeFirstDerivForwardDiffNoCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xP, PetscReal xE, PetscReal xEE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFirstDerivForwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFirstDerivForwardDiffNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFirstDerivCentralDiff_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFirstDerivBackwardDiffNoCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xWW, PetscReal xW, PetscReal xP, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFirstDerivBackwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xP, PetscReal xe, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFirstDerivBackwardDiffNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xP, PetscReal xe, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeSecondDerivForwardDiffNoCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xP, PetscReal xE, PetscReal xEE, PetscReal xEEE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeSecondDerivForwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscReal xEE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeSecondDerivForwardDiffNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xe, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeSecondDerivCentralDiff_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xw, PetscReal xP, PetscReal xe, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeSecondDerivBackwardDiffNoCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xWWW, PetscReal xWW, PetscReal xW, PetscReal xP, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeSecondDerivBackwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xWW, PetscReal xW, PetscReal xP, PetscReal xe, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeSecondDerivBackwardDiffNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xw, PetscReal xP, PetscReal xe, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeConvectionLinearInterpolationPrev_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xw, PetscReal xP, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeConvectionLinearInterpolationNext_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xP, PetscReal xe, PetscReal xE, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeConvectionLinearForwardExtrapolationNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeConvectionLinearBackwardExtrapolationNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xP, PetscReal xe, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeLinearInterpolation_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xw, PetscReal xP, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeLinearForwardExtrapolationNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeLinearBackwardExtrapolationNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xWW, PetscReal xW, PetscReal xw, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFaceNormalFirstDerivForwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFaceNormalFirstDerivCentralDiff_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xP, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
FLUCA_INTERN PetscErrorCode NSComputeFaceNormalFirstDerivBackwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xWW, PetscReal xW, PetscReal xw, PetscInt *ncols, DMStagStencil col[], PetscScalar v[]);
