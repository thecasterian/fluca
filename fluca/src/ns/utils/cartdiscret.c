#include "../src/ns/utils/cartdiscret.h"

PetscErrorCode NSComputeFirstDerivForwardDiffNoCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xP, PetscReal xE, PetscReal xEE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = xE - xP;
  h2       = xEE - xP;
  v[0]     = -(h1 + h2) / (h1 * h2);
  v[1]     = -h2 / (h1 * (h1 - h2));
  v[2]     = h1 / (h2 * (h1 - h2));
  col[0].i = i;
  col[0].j = j;
  col[0].k = k;
  col[1].i = i + (dir == DIR_X ? 1 : 0);
  col[1].j = j + (dir == DIR_Y ? 1 : 0);
  col[1].k = k + (dir == DIR_Z ? 1 : 0);
  col[2].i = i + (dir == DIR_X ? 2 : 0);
  col[2].j = j + (dir == DIR_Y ? 2 : 0);
  col[2].k = k + (dir == DIR_Z ? 2 : 0);
  *ncols   = 3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeFirstDerivForwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = xP - xw;
  h2       = xE - xP;
  v[0]     = (h2 - h1) / (h1 * h2);
  v[1]     = h1 / (h2 * (h1 + h2));
  col[0].i = i;
  col[0].j = j;
  col[0].k = k;
  col[1].i = i + (dir == DIR_X ? 1 : 0);
  col[1].j = j + (dir == DIR_Y ? 1 : 0);
  col[1].k = k + (dir == DIR_Z ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeFirstDerivForwardDiffNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = xP - xw;
  h2       = xE - xP;
  v[0]     = -2. * h1 / (h2 * (2. * h1 + h2));
  v[1]     = 2. * h1 / (h2 * (2. * h1 + h2));
  col[0].i = i;
  col[0].j = j;
  col[0].k = k;
  col[1].i = i + (dir == DIR_X ? 1 : 0);
  col[1].j = j + (dir == DIR_Y ? 1 : 0);
  col[1].k = k + (dir == DIR_Z ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeFirstDerivCentralDiff_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -1. / (xE - xW);
  v[1]     = 1. / (xE - xW);
  col[0].i = i - (dir == DIR_X ? 1 : 0);
  col[0].j = j - (dir == DIR_Y ? 1 : 0);
  col[0].k = k - (dir == DIR_Z ? 1 : 0);
  col[1].i = i + (dir == DIR_X ? 1 : 0);
  col[1].j = j + (dir == DIR_Y ? 1 : 0);
  col[1].k = k + (dir == DIR_Z ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeFirstDerivBackwardDiffNoCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xWW, PetscReal xW, PetscReal xP, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = xP - xW;
  h2       = xP - xWW;
  v[0]     = -h1 / (h2 * (h1 - h2));
  v[1]     = h2 / (h1 * (h1 - h2));
  v[2]     = (h1 + h2) / (h1 * h2);
  col[0].i = i - (dir == DIR_X ? 2 : 0);
  col[0].j = j - (dir == DIR_Y ? 2 : 0);
  col[0].k = k - (dir == DIR_Z ? 2 : 0);
  col[1].i = i - (dir == DIR_X ? 1 : 0);
  col[1].j = j - (dir == DIR_Y ? 1 : 0);
  col[1].k = k - (dir == DIR_Z ? 1 : 0);
  col[2].i = i;
  col[2].j = j;
  col[2].k = k;
  *ncols   = 3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeFirstDerivBackwardDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xP, PetscReal xe, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = xe - xP;
  h2       = xP - xW;
  v[0]     = -h1 / (h2 * (h1 + h2));
  v[1]     = (h1 - h2) / (h1 * h2);
  col[0].i = i - (dir == DIR_X ? 1 : 0);
  col[0].j = j - (dir == DIR_Y ? 1 : 0);
  col[0].k = k - (dir == DIR_Z ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  col[1].k = k;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode NSComputeFirstDerivBackwardNeumannCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xP, PetscReal xe, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = xe - xP;
  h2       = xP - xW;
  v[0]     = -2. * h1 / (h2 * (2. * h1 + h2));
  v[1]     = 2. * h1 / (h2 * (2. * h1 + h2));
  col[0].i = i - (dir == DIR_X ? 1 : 0);
  col[0].j = j - (dir == DIR_Y ? 1 : 0);
  col[0].k = k - (dir == DIR_Z ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  col[1].k = k;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeSecondDerivForwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xw, PetscReal xP, PetscReal xE, PetscReal xEE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = xP - xw;
  h2 = xE - xP;
  h3 = xEE - xP;

  v[0] += 2. * (h1 - h2 - h3) / (h1 * h2 * h3);
  v[*ncols]     = 2. * (h1 - h3) / (h2 * (h1 + h2) * (h2 - h3));
  v[*ncols + 1] = 2. * (h2 - h1) / (h3 * (h1 + h3) * (h2 - h3));

  col[*ncols].i     = i + (dir == DIR_X ? 1 : 0);
  col[*ncols].j     = j + (dir == DIR_Y ? 1 : 0);
  col[*ncols].k     = k + (dir == DIR_Z ? 1 : 0);
  col[*ncols + 1].i = i + (dir == DIR_X ? 2 : 0);
  col[*ncols + 1].j = j + (dir == DIR_Y ? 2 : 0);
  col[*ncols + 1].k = k + (dir == DIR_Z ? 2 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeSecondDerivCentralDiff_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xw, PetscReal xP, PetscReal xe, PetscReal xE, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = xP - xW;
  h2 = xE - xP;
  h3 = xe - xw;

  v[0] -= (1. / (h1 * h3) + 1. / (h2 * h3));
  v[*ncols]     = 1. / (h1 * h3);
  v[*ncols + 1] = 1. / (h2 * h3);

  col[*ncols].i     = i - (dir == DIR_X ? 1 : 0);
  col[*ncols].j     = j - (dir == DIR_Y ? 1 : 0);
  col[*ncols].k     = k - (dir == DIR_Z ? 1 : 0);
  col[*ncols + 1].i = i + (dir == DIR_X ? 1 : 0);
  col[*ncols + 1].j = j + (dir == DIR_Y ? 1 : 0);
  col[*ncols + 1].k = k + (dir == DIR_Z ? 1 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeSecondDerivBackwardDiffDirichletCond_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xWW, PetscReal xW, PetscReal xP, PetscReal xe, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = xe - xP;
  h2 = xP - xW;
  h3 = xP - xWW;

  v[0] += 2. * (h1 - h2 - h3) / (h1 * h2 * h3);
  v[*ncols]     = 2. * (h1 - h3) / (h2 * (h1 + h2) * (h2 - h3));
  v[*ncols + 1] = 2. * (h2 - h1) / (h3 * (h1 + h3) * (h2 - h3));

  col[*ncols].i     = i - (dir == DIR_X ? 1 : 0);
  col[*ncols].j     = j - (dir == DIR_Y ? 1 : 0);
  col[*ncols].k     = k - (dir == DIR_Z ? 1 : 0);
  col[*ncols + 1].i = i - (dir == DIR_X ? 2 : 0);
  col[*ncols + 1].j = j - (dir == DIR_Y ? 2 : 0);
  col[*ncols + 1].k = k - (dir == DIR_Z ? 2 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeConvectionLinearInterpolationPrev_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xw, PetscReal xP, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -0.5 * v_f / h * (xP - xw) / (xP - xW);
  v[1]     = -0.5 * v_f / h * (xw - xW) / (xP - xW);
  col[0].i = i - (dir == DIR_X ? 1 : 0);
  col[0].j = j - (dir == DIR_Y ? 1 : 0);
  col[0].k = k - (dir == DIR_Z ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  col[1].k = k;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeConvectionLinearInterpolationNext_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xP, PetscReal xe, PetscReal xE, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = 0.5 * v_f / h * (xE - xe) / (xE - xP);
  v[1]     = 0.5 * v_f / h * (xe - xP) / (xE - xP);
  col[0].i = i;
  col[0].j = j;
  col[0].k = k;
  col[1].i = i + (dir == DIR_X ? 1 : 0);
  col[1].j = j + (dir == DIR_Y ? 1 : 0);
  col[1].k = k + (dir == DIR_Z ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeLinearInterpolation_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xw, PetscReal xP, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = (xP - xw) / (xP - xW);
  v[1]     = (xw - xW) / (xP - xW);
  col[0].i = i - (dir == DIR_X ? 1 : 0);
  col[0].j = j - (dir == DIR_Y ? 1 : 0);
  col[0].k = k - (dir == DIR_Z ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  col[1].k = k;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSComputeFaceNormalDerivative_Cart(Direction dir, PetscInt i, PetscInt j, PetscInt k, PetscReal xW, PetscReal xP, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -1. / (xP - xW);
  v[1]     = 1. / (xP - xW);
  col[0].i = i - (dir == DIR_X ? 1 : 0);
  col[0].j = j - (dir == DIR_Y ? 1 : 0);
  col[0].k = k - (dir == DIR_Z ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  col[1].k = k;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}
