#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

typedef enum {
  DERIV_X,
  DERIV_Y,
} DerivDirection;

static PetscErrorCode ComputeIdentityOperator_Private(DM dm, Mat Id)
{
  PetscInt      x, y, m, n;
  DMStagStencil row, col;
  PetscScalar   v;
  PetscInt      i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  col.loc = DMSTAG_ELEMENT;
  col.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;
      col.i = i;
      col.j = j;
      v     = 1.;
      PetscCall(DMStagMatSetValuesStencil(dm, Id, 1, &row, 1, &col, &v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Id, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Id, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFirstDerivForwardDiffNoCond_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_center, PetscReal coord_next, PetscReal coord_next2, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = coord_next - coord_center;
  h2       = coord_next2 - coord_center;
  v[0]     = -(h1 + h2) / (h1 * h2);
  v[1]     = -h2 / (h1 * (h1 - h2));
  v[2]     = h1 / (h2 * (h1 - h2));
  col[0].i = i;
  col[0].j = j;
  col[1].i = i + (dir == DERIV_X ? 1 : 0);
  col[1].j = j + (dir == DERIV_Y ? 1 : 0);
  col[2].i = i + (dir == DERIV_X ? 2 : 0);
  col[2].j = j + (dir == DERIV_Y ? 2 : 0);
  *ncols   = 3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFirstDerivCentralDiff_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -1. / (coord_next - coord_prev);
  v[1]     = 1. / (coord_next - coord_prev);
  col[0].i = i - (dir == DERIV_X ? 1 : 0);
  col[0].j = j - (dir == DERIV_Y ? 1 : 0);
  col[1].i = i + (dir == DERIV_X ? 1 : 0);
  col[1].j = j + (dir == DERIV_Y ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFirstDerivBackwardDiffNoCond_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev2, PetscReal coord_prev, PetscReal coord_center, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = coord_center - coord_prev;
  h2       = coord_center - coord_prev2;
  v[0]     = -h1 / (h2 * (h1 - h2));
  v[1]     = h2 / (h1 * (h1 - h2));
  v[2]     = (h1 + h2) / (h1 * h2);
  col[0].i = i - (dir == DERIV_X ? 2 : 0);
  col[0].j = j - (dir == DERIV_Y ? 2 : 0);
  col[1].i = i - (dir == DERIV_X ? 1 : 0);
  col[1].j = j - (dir == DERIV_Y ? 1 : 0);
  col[2].i = i;
  col[2].j = j;
  *ncols   = 3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputePressureGradientOperators_Private(DM dm, const NSBoundaryCondition *bcs, Mat Gp[])
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[5];
  PetscInt            ncols;
  PetscScalar         v[3];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Gp[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(Gp[1], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (i = 0; i < 5; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  /* Compute x-gradient operator */
  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;

      if (i == 0) {
        /* Left boundary */
        switch (bcs[0].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivForwardDiffNoCond_Private(DERIV_X, i, j, arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivBackwardDiffNoCond_Private(DERIV_X, i, j, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivCentralDiff_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, Gp[0], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Compute y-gradient operator */
  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;

      if (j == 0) {
        /* Down boundary */
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivForwardDiffNoCond_Private(DERIV_Y, i, j, arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivBackwardDiffNoCond_Private(DERIV_Y, i, j, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivCentralDiff_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, Gp[1], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Gp[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gp[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Gp[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gp[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivForwardDiffDirichletCond_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_bnd, PetscReal coord_center, PetscReal coord_next, PetscReal coord_next2, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = coord_center - coord_bnd;
  h2 = coord_next - coord_center;
  h3 = coord_next2 - coord_center;

  v[0] += 2. * (h1 - h2 - h3) / (h1 * h2 * h3);
  v[*ncols]     = 2. * (h1 - h3) / (h2 * (h1 + h2) * (h2 - h3));
  v[*ncols + 1] = 2. * (h2 - h1) / (h3 * (h1 + h3) * (h2 - h3));

  col[*ncols].i     = i + (dir == DERIV_X ? 1 : 0);
  col[*ncols].j     = j + (dir == DERIV_Y ? 1 : 0);
  col[*ncols + 1].i = i + (dir == DERIV_X ? 2 : 0);
  col[*ncols + 1].j = j + (dir == DERIV_Y ? 2 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivCentralDiff_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_prev_b, PetscReal coord_center, PetscReal coord_next_b, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = coord_center - coord_prev;
  h2 = coord_next - coord_center;
  h3 = coord_next_b - coord_prev_b;

  v[0] -= (1. / (h1 * h3) + 1. / (h2 * h3));
  v[*ncols]     = 1. / (h1 * h3);
  v[*ncols + 1] = 1. / (h2 * h3);

  col[*ncols].i     = i - (dir == DERIV_X ? 1 : 0);
  col[*ncols].j     = j - (dir == DERIV_Y ? 1 : 0);
  col[*ncols + 1].i = i + (dir == DERIV_X ? 1 : 0);
  col[*ncols + 1].j = j + (dir == DERIV_Y ? 1 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivBackwardDiffDirichletCond_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev2, PetscReal coord_prev, PetscReal coord_center, PetscReal coord_bnd, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = coord_bnd - coord_center;
  h2 = coord_center - coord_prev;
  h3 = coord_center - coord_prev2;

  v[0] += 2. * (h1 - h2 - h3) / (h1 * h2 * h3);
  v[*ncols]     = 2. * (h1 - h3) / (h2 * (h1 + h2) * (h2 - h3));
  v[*ncols + 1] = 2. * (h2 - h1) / (h3 * (h1 + h3) * (h2 - h3));

  col[*ncols].i     = i - (dir == DERIV_X ? 1 : 0);
  col[*ncols].j     = j - (dir == DERIV_Y ? 1 : 0);
  col[*ncols + 1].i = i - (dir == DERIV_X ? 2 : 0);
  col[*ncols + 1].j = j - (dir == DERIV_Y ? 2 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityLaplacianOperator_Private(DM dm, const NSBoundaryCondition *bcs, Mat Lv)
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[5];
  PetscInt            ncols;
  PetscScalar         v[5];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Lv, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (i = 0; i < 5; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i    = i;
      row.j    = j;
      col[0].i = i;
      col[0].j = j;
      v[0]     = 0.;
      ncols    = 1;

      if (i == 0) {
        /* Left boundary */
        switch (bcs[0].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivForwardDiffDirichletCond_Private(DERIV_X, i, j, arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffDirichletCond_Private(DERIV_X, i, j, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], arrcx[i][inextc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
      }

      if (j == 0) {
        /* Down boundary */
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivForwardDiffDirichletCond_Private(DERIV_Y, i, j, arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffDirichletCond_Private(DERIV_Y, i, j, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], arrcy[j][inextc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, Lv, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Lv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Lv, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityLaplacianBoundaryConditionVector_Private(DM dm, PetscInt axis, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
  PetscBool           isFirstRankx, isFirstRanky, isLastRankx, isLastRanky;
  DMStagStencil       row;
  PetscScalar         v;
  PetscReal           xb[2];
  PetscScalar         vb[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscReal           h1, h2, h3;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(dm, &isFirstRankx, &isFirstRanky, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  /* Left boundary */
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        row.i = 0;
        row.j = j;
        xb[0] = arrcx[0][iprevc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));

        h1 = arrcx[0][ielemc] - arrcx[0][iprevc];
        h2 = arrcx[1][ielemc] - arrcx[0][ielemc];
        h3 = arrcx[2][ielemc] - arrcx[0][ielemc];

        v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[axis];
        PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
    }

  /* Right boundary */
  if (isLastRankx) switch (bcs[1].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        row.i = M - 1;
        row.j = j;
        xb[0] = arrcx[M - 1][inextc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));

        h1 = arrcx[M - 1][inextc] - arrcx[M - 1][ielemc];
        h2 = arrcx[M - 1][ielemc] - arrcx[M - 2][ielemc];
        h3 = arrcx[M - 1][ielemc] - arrcx[M - 3][ielemc];

        v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[axis];
        PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
    }

  /* Down boundary */
  if (isFirstRanky) switch (bcs[2].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = 0;
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[0][iprevc];
        PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));

        h1 = arrcy[0][ielemc] - arrcy[0][iprevc];
        h2 = arrcy[1][ielemc] - arrcy[0][ielemc];
        h3 = arrcy[2][ielemc] - arrcy[0][ielemc];

        v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[axis];
        PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
    }

  /* Up boundary */
  if (isLastRanky) switch (bcs[3].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = N - 1;
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[N - 1][inextc];
        PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));

        h1 = arrcy[N - 1][inextc] - arrcy[N - 1][ielemc];
        h2 = arrcy[N - 1][ielemc] - arrcy[N - 2][ielemc];
        h3 = arrcy[N - 1][ielemc] - arrcy[N - 3][ielemc];

        v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[axis];
        PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeLinearInterpolation_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_prev_f, PetscReal coord_center, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = (coord_center - coord_prev_f) / (coord_center - coord_prev);
  v[1]     = (coord_prev_f - coord_prev) / (coord_center - coord_prev);
  col[0].i = i - (dir == DERIV_X ? 1 : 0);
  col[0].j = j - (dir == DERIV_Y ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityInterpolationOperators_Private(DM dm, DM fdm, const NSBoundaryCondition *bcs, Mat Tv[])
{
  PetscInt            M, N, x, y, m, n, nExtrax, nExtray;
  DMStagStencil       row, col[2];
  PetscInt            ncols, ir, ic[2];
  PetscScalar         v[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j;
  const PetscInt      dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Tv[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(Tv[1], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.c = 0;
  for (i = 0; i < 2; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  /* Compute x-interpolation operator */
  row.loc = DMSTAG_LEFT;
  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      row.i = i;
      row.j = j;
      ncols = 0;

      if (i == 0) {
        /* Left boundary */
        switch (bcs[0].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeLinearInterpolation_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC: /* Cannot happen */
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeLinearInterpolation_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], &ncols, col, v));
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Tv[0], 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Compute y-interpolation operator */
  row.loc = DMSTAG_DOWN;
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;
      ncols = 0;

      if (j == 0) {
        /* Down boundary */
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeLinearInterpolation_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC: /* Cannot happen */
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeLinearInterpolation_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], &ncols, col, v));
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Tv[1], 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Tv[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Tv[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Tv[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Tv[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityInterpolationBoundaryConditionVector_Private(DM dm, DM fdm, DerivDirection dir, PetscInt axis, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
  PetscBool           isFirstRankx, isFirstRanky, isLastRankx, isLastRanky;
  DMStagStencil       row;
  PetscReal           xb[2];
  PetscScalar         vb[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(dm, &isFirstRankx, &isFirstRanky, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  switch (dir) {
  case DERIV_X:
    /* Boundary condition for x-interpolation */
    row.loc = DMSTAG_LEFT;
    row.c   = 0;

    if (isFirstRankx) switch (bcs[0].type) {
      case NS_BC_VELOCITY:
        for (j = y; j < y + n; ++j) {
          row.i = 0;
          row.j = j;
          xb[0] = arrcx[0][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &vb[axis], INSERT_VALUES));
        }
        break;
      case NS_BC_PERIODIC:
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
      }

    if (isLastRankx) switch (bcs[1].type) {
      case NS_BC_VELOCITY:
        for (j = y; j < y + n; ++j) {
          row.i = M;
          row.j = j;
          xb[0] = arrcx[M][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &vb[axis], INSERT_VALUES));
        }
        break;
      case NS_BC_PERIODIC:
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
      }
    break;

  case DERIV_Y:
    /* Boundary condition for y-interpolation */
    row.loc = DMSTAG_DOWN;
    row.c   = 0;

    if (isFirstRanky) switch (bcs[2].type) {
      case NS_BC_VELOCITY:
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = 0;
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[0][iprevc];
          PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &vb[axis], INSERT_VALUES));
        }
        break;
      case NS_BC_PERIODIC:
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
      }

    if (isLastRanky) switch (bcs[3].type) {
      case NS_BC_VELOCITY:
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = N;
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[N][iprevc];
          PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &vb[axis], INSERT_VALUES));
        }
        break;
      case NS_BC_PERIODIC:
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
      }
    break;

  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported direction");
  }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeStaggeredVelocityGradientOperators_Private(DM dm, DM fdm, const NSBoundaryCondition *bcs, Mat Gstv[])
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[2];
  PetscInt            ir, ic[2];
  PetscScalar         v[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc;
  PetscInt            i, j;
  const PetscInt      dim = 2, ncols = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));

  PetscCall(MatSetOption(Gstv[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(Gstv[1], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  /* Compute x-gradient operator */
  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;

      v[0] = -1. / (arrcx[i][inextc] - arrcx[i][iprevc]);
      v[1] = 1. / (arrcx[i][inextc] - arrcx[i][iprevc]);

      col[0].i   = i;
      col[0].j   = j;
      col[0].loc = DMSTAG_LEFT;
      col[0].c   = 0;

      col[1].i   = i;
      col[1].j   = j;
      col[1].loc = DMSTAG_RIGHT;
      col[1].c   = 0;

      PetscCall(DMStagStencilToIndexLocal(dm, dim, 1, &row, &ir));
      PetscCall(DMStagStencilToIndexLocal(fdm, dim, ncols, col, ic));
      PetscCall(MatSetValuesLocal(Gstv[0], 1, &ir, ncols, ic, v, INSERT_VALUES));
    }

  /* Compute y-gradient operator */
  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;

      v[0] = -1. / (arrcy[j][inextc] - arrcy[j][iprevc]);
      v[1] = 1. / (arrcy[j][inextc] - arrcy[j][iprevc]);

      col[0].i   = i;
      col[0].j   = j;
      col[0].loc = DMSTAG_DOWN;
      col[0].c   = 0;

      col[1].i   = i;
      col[1].j   = j;
      col[1].loc = DMSTAG_UP;
      col[1].c   = 0;

      PetscCall(DMStagStencilToIndexLocal(dm, dim, 1, &row, &ir));
      PetscCall(DMStagStencilToIndexLocal(fdm, dim, ncols, col, ic));
      PetscCall(MatSetValuesLocal(Gstv[1], 1, &ir, ncols, ic, v, INSERT_VALUES));
    }

  PetscCall(MatAssemblyBegin(Gstv[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstv[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Gstv[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstv[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceDerivative_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_center, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -1. / (coord_center - coord_prev);
  v[1]     = 1. / (coord_center - coord_prev);
  col[0].i = i - (dir == DERIV_X ? 1 : 0);
  col[0].j = j - (dir == DERIV_Y ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeStaggeredPressureGradientOperators_Private(DM dm, DM fdm, const NSBoundaryCondition *bcs, Mat Gstp[])
{
  PetscInt            M, N, x, y, m, n, nExtrax, nExtray;
  DMStagStencil       row, col[2];
  PetscInt            ncols, ir, ic[2];
  PetscScalar         v[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            ielemc;
  PetscInt            i, j;
  const PetscInt      dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Gstp[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(Gstp[1], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.c = 0;
  for (i = 0; i < 2; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  /* Compute x-gradient operator */
  row.loc = DMSTAG_LEFT;
  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      row.i = i;
      row.j = j;
      ncols = 0;

      if (i == 0) {
        /* Left boundary */
        switch (bcs[0].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFaceDerivative_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC: /* Cannot happen */
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeFaceDerivative_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Gstp[0], 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Compute y-gradient operator */
  row.loc = DMSTAG_DOWN;
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;
      ncols = 0;

      if (j == 0) {
        /* Down boundary */
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFaceDerivative_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          break;
        case NS_BC_PERIODIC: /* Cannot happen */
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeFaceDerivative_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Gstp[1], 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  PetscCall(MatAssemblyBegin(Gstp[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstp[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Gstp[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstp[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateStaggeredVelocityDivergenceOperator_Private(Mat Gstv[], Mat *Dstv)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate(Gstv[0], MAT_COPY_VALUES, Dstv));
  PetscCall(MatAXPY(*Dstv, 1.0, Gstv[1], DIFFERENT_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperatorFromDMToDM_Private(DM dmfrom, DM dmto, Mat *A)
{
  PetscInt               entriesfrom, entriesto;
  ISLocalToGlobalMapping ltogfrom, ltogto;
  MatType                mattype;

  PetscFunctionBegin;
  PetscCall(DMStagGetEntries(dmfrom, &entriesfrom));
  PetscCall(DMStagGetEntries(dmto, &entriesto));
  PetscCall(DMGetLocalToGlobalMapping(dmfrom, &ltogfrom));
  PetscCall(DMGetLocalToGlobalMapping(dmto, &ltogto));
  PetscCall(DMGetMatType(dmfrom, &mattype));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmfrom), A));
  PetscCall(MatSetSizes(*A, entriesto, entriesfrom, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(*A, mattype));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltogto, ltogfrom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputeSpatialOperators2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      sdm, Vdm;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  PetscCall(ComputePressureGradientOperators_Private(sdm, ns->bcs, fsm->Gp));
  PetscCall(ComputeVelocityLaplacianOperator_Private(sdm, ns->bcs, fsm->Lv));
  PetscCall(ComputeVelocityInterpolationOperators_Private(sdm, Vdm, ns->bcs, fsm->Tv));
  PetscCall(ComputeStaggeredVelocityGradientOperators_Private(sdm, Vdm, ns->bcs, fsm->Gstv));
  PetscCall(ComputeStaggeredPressureGradientOperators_Private(sdm, Vdm, ns->bcs, fsm->Gstp));
  PetscCall(CreateStaggeredVelocityDivergenceOperator_Private(fsm->Gstv, &fsm->Dstv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorsIntermediateVelocity_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  KSPVCtx *kspvctx = (KSPVCtx *)ctx;
  NS       ns      = kspvctx->ns;
  DM       dm;
  Mat      Lv;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));

  PetscCall(DMCreateMatrix(dm, &Lv));
  PetscCall(ComputeVelocityLaplacianOperator_Private(dm, ns->bcs, Lv));

  PetscCall(ComputeIdentityOperator_Private(dm, Jpre));
  PetscCall(MatAXPY(Jpre, -0.5 * ns->mu * ns->dt / ns->rho, Lv, DIFFERENT_NONZERO_PATTERN));

  PetscCall(MatDestroy(&Lv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSIntermediateVelocity_Private(KSP ksp, Vec b, void *ctx)
{
  KSPVCtx *kspvctx = (KSPVCtx *)ctx;
  NS       ns      = kspvctx->ns;
  NS_FSM  *fsm     = (NS_FSM *)ns->data;
  DM       sdm;
  Vec      Gp, Lv, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));

  PetscCall(DMGetGlobalVector(sdm, &Gp));
  PetscCall(DMGetGlobalVector(sdm, &Lv));
  PetscCall(DMGetGlobalVector(sdm, &vbc));

  PetscCall(MatMult(fsm->Gp[kspvctx->axis], fsm->p_half, Gp));
  PetscCall(MatMult(fsm->Lv, fsm->v[kspvctx->axis], Lv));
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(sdm, kspvctx->axis, ns->bcs, ns->t, vbc));
  PetscCall(VecAXPY(Lv, 1., vbc));

  /* RHS of momentum equation */
  PetscCall(VecAXPBYPCZ(b, 1., 0.5 * ns->mu * ns->dt / ns->rho, 0., fsm->v[kspvctx->axis], Lv));
  PetscCall(VecAXPBYPCZ(b, -1.5 * ns->dt, 0.5 * ns->dt, 1., fsm->N[kspvctx->axis], fsm->N_prev[kspvctx->axis]));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, Gp));

  /* Add boundary condition */
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(sdm, kspvctx->axis, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(b, 0.5 * ns->mu * ns->dt / ns->rho, vbc));

  PetscCall(DMRestoreGlobalVector(sdm, &Gp));
  PetscCall(DMRestoreGlobalVector(sdm, &Lv));
  PetscCall(DMRestoreGlobalVector(sdm, &vbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorPressureCorrection_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS           ns = (NS)ctx;
  MPI_Comm     comm;
  DM           dm, fdm;
  Mat          Gstv[2], Gstp[2], GstvGstp;
  MatNullSpace nullspace;
  PetscInt     d;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMStagCreateCompatibleDMStag(dm, 0, 1, 0, 0, &fdm));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(fdm, PETSC_TRUE));

  PetscCall(CreateOperatorFromDMToDM_Private(fdm, dm, &Gstv[0]));
  PetscCall(CreateOperatorFromDMToDM_Private(fdm, dm, &Gstv[1]));
  PetscCall(CreateOperatorFromDMToDM_Private(dm, fdm, &Gstp[0]));
  PetscCall(CreateOperatorFromDMToDM_Private(dm, fdm, &Gstp[1]));
  PetscCall(ComputeStaggeredVelocityGradientOperators_Private(dm, fdm, ns->bcs, Gstv));
  PetscCall(ComputeStaggeredPressureGradientOperators_Private(dm, fdm, ns->bcs, Gstp));

  /* dt / rho * (Gstv[0] * Gstp[0] + Gstv[1] * Gstp[1]) */
  PetscCall(MatZeroEntries(Jpre));
  for (d = 0; d < 2; ++d) {
    PetscCall(MatMatMult(Gstv[d], Gstp[d], MAT_INITIAL_MATRIX, 1., &GstvGstp));
    PetscCall(MatAXPY(Jpre, 1., GstvGstp, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&GstvGstp));
  }
  PetscCall(MatScale(Jpre, ns->dt / ns->rho));

  PetscCall(MatDestroy(&Gstv[0]));
  PetscCall(MatDestroy(&Gstv[1]));
  PetscCall(MatDestroy(&Gstp[0]));
  PetscCall(MatDestroy(&Gstp[1]));
  PetscCall(DMDestroy(&fdm));

  // TODO: below is temporary for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(J, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSPressureCorrection_Private(KSP ksp, Vec b, void *ctx)
{
  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      sdm;

  MPI_Comm     comm;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(MatMult(fsm->Dstv, fsm->V_star, b));

  // TODO: below is only for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMSetKSPComputeFunctions2d_Cart_Internal(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  PetscInt d;

  PetscFunctionBegin;
  for (d = 0; d < 2; ++d) {
    PetscCall(KSPSetComputeOperators(fsm->kspv[d], ComputeOperatorsIntermediateVelocity_Private, &fsm->kspvctx[d]));
    PetscCall(KSPSetComputeRHS(fsm->kspv[d], ComputeRHSIntermediateVelocity_Private, &fsm->kspvctx[d]));
  }
  PetscCall(KSPSetComputeOperators(fsm->kspp, ComputeOperatorPressureCorrection_Private, ns));
  PetscCall(KSPSetComputeRHS(fsm->kspp, ComputeRHSPressureCorrection_Private, ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvection_Private(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  DM       sdm, Vdm;
  Vec      v_interp[2], vbc, vmult;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  PetscCall(DMGetGlobalVector(Vdm, &v_interp[0]));
  PetscCall(DMGetGlobalVector(Vdm, &v_interp[1]));
  PetscCall(DMGetGlobalVector(Vdm, &vbc));
  PetscCall(DMGetGlobalVector(Vdm, &vmult));

  /* Interpolate velocity */
  for (d = 0; d < 2; ++d) {
    PetscCall(MatMult(fsm->Tv[0], fsm->v[d], v_interp[d]));
    PetscCall(MatMultAdd(fsm->Tv[1], fsm->v[d], v_interp[d], v_interp[d]));
    PetscCall(ComputeVelocityInterpolationBoundaryConditionVector_Private(sdm, Vdm, DERIV_X, d, ns->bcs, ns->t + ns->dt, vbc));
    PetscCall(VecAXPY(v_interp[d], 1., vbc));
    PetscCall(ComputeVelocityInterpolationBoundaryConditionVector_Private(sdm, Vdm, DERIV_Y, d, ns->bcs, ns->t + ns->dt, vbc));
    PetscCall(VecAXPY(v_interp[d], 1., vbc));
  }

  /* N_i = d(U * v_i)/dx + d(V * v_i)/dy */
  for (d = 0; d < 2; ++d) {
    PetscCall(VecPointwiseMult(vmult, fsm->V, v_interp[d]));
    PetscCall(MatMult(fsm->Dstv, vmult, fsm->N[d]));
  }

  PetscCall(DMRestoreGlobalVector(Vdm, &v_interp[0]));
  PetscCall(DMRestoreGlobalVector(Vdm, &v_interp[1]));
  PetscCall(DMRestoreGlobalVector(Vdm, &vbc));
  PetscCall(DMRestoreGlobalVector(Vdm, &vmult));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeIntermediateVelocity_Private(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  DM       sdm, Vdm;
  Vec      s, vbc;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    PetscCall(KSPSolve(fsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(fsm->kspv[d], &s));
    PetscCall(VecCopy(s, fsm->v_star[d]));
  }

  /* Compute face intermediate velocity. */
  PetscCall(DMGetGlobalVector(Vdm, &vbc));

  PetscCall(MatMult(fsm->Tv[0], fsm->v_star[0], fsm->V_star));
  PetscCall(MatMultAdd(fsm->Tv[1], fsm->v_star[1], fsm->V_star, fsm->V_star));
  PetscCall(ComputeVelocityInterpolationBoundaryConditionVector_Private(sdm, Vdm, DERIV_X, 0, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(fsm->V_star, 1., vbc));
  PetscCall(ComputeVelocityInterpolationBoundaryConditionVector_Private(sdm, Vdm, DERIV_Y, 1, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(fsm->V_star, 1., vbc));

  PetscCall(DMRestoreGlobalVector(Vdm, &vbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputePressureCorrection_Private(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      sdm;
  Vec     s;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(KSPSolve(fsm->kspp, NULL, NULL));
  PetscCall(KSPGetSolution(fsm->kspp, &s));
  PetscCall(VecCopy(s, fsm->p_prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UpdateToNextTimeStep_Private(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      sdm, Vdm;
  Mat     A;
  Vec     Gpp[2], Gstpp;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  PetscCall(KSPGetOperators(fsm->kspv[0], &A, NULL));
  PetscCall(DMGetGlobalVector(sdm, &Gpp[0]));
  PetscCall(DMGetGlobalVector(sdm, &Gpp[1]));
  PetscCall(DMGetGlobalVector(Vdm, &Gstpp));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecCopy(fsm->N[1], fsm->N_prev[1]));

  PetscCall(MatMult(fsm->Gp[0], fsm->p_prime, Gpp[0]));
  PetscCall(MatMult(fsm->Gp[1], fsm->p_prime, Gpp[1]));
  PetscCall(MatMult(fsm->Gstp[0], fsm->p_prime, Gstpp));
  PetscCall(MatMultAdd(fsm->Gstp[1], fsm->p_prime, Gstpp, Gstpp));

  PetscCall(VecWAXPY(fsm->v[0], -ns->dt / ns->rho, Gpp[0], fsm->v_star[0]));
  PetscCall(VecWAXPY(fsm->v[1], -ns->dt / ns->rho, Gpp[1], fsm->v_star[1]));
  PetscCall(VecWAXPY(fsm->V, -ns->dt / ns->rho, Gstpp, fsm->V_star));

  PetscCall(MatMultAdd(A, fsm->p_prime, fsm->p_half_prev, fsm->p_half));
  PetscCall(VecAXPBYPCZ(fsm->p, 1.5, -0.5, 0., fsm->p_half, fsm->p_half_prev));

  PetscCall(DMRestoreGlobalVector(sdm, &Gpp[0]));
  PetscCall(DMRestoreGlobalVector(sdm, &Gpp[1]));
  PetscCall(DMRestoreGlobalVector(Vdm, &Gstpp));

  PetscCall(ComputeConvection_Private(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMIterate2d_Cart_Internal(NS ns)
{
  PetscFunctionBegin;
  PetscCall(ComputeIntermediateVelocity_Private(ns));
  PetscCall(ComputePressureCorrection_Private(ns));
  PetscCall(UpdateToNextTimeStep_Private(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}
