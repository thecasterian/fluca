#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

typedef enum {
  DERIV_X,
  DERIV_Y,
} DerivDirection;

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

static PetscErrorCode ComputePressureGradientOperator_Private(DM sdm, DM vdm, const NSBoundaryCondition *bcs, Mat Gp)
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[3];
  PetscInt            ncols, ir, ic[3];
  PetscScalar         v[3];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j;
  const PetscInt      dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(sdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(sdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Gp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  for (i = 0; i < 3; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  /* Compute x-gradient operator */
  row.c = 0;
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

      PetscCall(DMStagStencilToIndexLocal(vdm, dim, 1, &row, &ir));
      PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
      PetscCall(MatSetValuesLocal(Gp, 1, &ir, ncols, ic, v, INSERT_VALUES));
    }

  /* Compute y-gradient operator */
  row.c = 1;
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

      PetscCall(DMStagStencilToIndexLocal(vdm, dim, 1, &row, &ir));
      PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
      PetscCall(MatSetValuesLocal(Gp, 1, &ir, ncols, ic, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Gp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gp, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));
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
  PetscInt            M, N, x, y, m, n, dim;
  DMStagStencil       row, col[5];
  PetscInt            ncols;
  PetscScalar         v[5];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(MatSetOption(Lv, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  for (c = 0; c < dim; ++c) {
    row.loc = DMSTAG_ELEMENT;
    row.c   = c;
    for (i = 0; i < 5; ++i) {
      col[i].loc = DMSTAG_ELEMENT;
      col[i].c   = c;
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
  }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Lv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Lv, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityLaplacianBoundaryConditionVector_Private(DM vdm, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n, dim;
  PetscBool           isFirstRankx, isFirstRanky, isLastRankx, isLastRanky;
  DMStagStencil       row[3];
  PetscScalar         v[3];
  PetscReal           xb[3];
  PetscScalar         vb[3];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscReal           h1, h2, h3;
  PetscInt            i, j, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(vdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(vdm, &isFirstRankx, &isFirstRanky, NULL));
  PetscCall(DMStagGetIsLastRank(vdm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));
  PetscCall(DMGetDimension(vdm, &dim));

  PetscCall(VecSet(vbc, 0.));

  /* Initialize row stencils for all components */
  for (c = 0; c < dim; ++c) {
    row[c].loc = DMSTAG_ELEMENT;
    row[c].c   = c;
  }

  /* Left boundary */
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        for (c = 0; c < dim; ++c) {
          row[c].i = 0;
          row[c].j = j;
        }
        xb[0] = arrcx[0][iprevc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[0].velocity(dim, t, xb, vb, bcs[0].ctx_velocity));

        h1 = arrcx[0][ielemc] - arrcx[0][iprevc];
        h2 = arrcx[1][ielemc] - arrcx[0][ielemc];
        h3 = arrcx[2][ielemc] - arrcx[0][ielemc];

        for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, dim, row, v, ADD_VALUES));
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
        for (c = 0; c < dim; ++c) {
          row[c].i = M - 1;
          row[c].j = j;
        }
        xb[0] = arrcx[M - 1][inextc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[1].velocity(dim, t, xb, vb, bcs[1].ctx_velocity));

        h1 = arrcx[M - 1][inextc] - arrcx[M - 1][ielemc];
        h2 = arrcx[M - 1][ielemc] - arrcx[M - 2][ielemc];
        h3 = arrcx[M - 1][ielemc] - arrcx[M - 3][ielemc];

        for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, dim, row, v, ADD_VALUES));
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
        for (c = 0; c < dim; ++c) {
          row[c].i = i;
          row[c].j = 0;
        }
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[0][iprevc];
        PetscCall(bcs[2].velocity(dim, t, xb, vb, bcs[2].ctx_velocity));

        h1 = arrcy[0][ielemc] - arrcy[0][iprevc];
        h2 = arrcy[1][ielemc] - arrcy[0][ielemc];
        h3 = arrcy[2][ielemc] - arrcy[0][ielemc];

        for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, dim, row, v, ADD_VALUES));
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
        for (c = 0; c < dim; ++c) {
          row[c].i = i;
          row[c].j = N - 1;
        }
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[N - 1][inextc];
        PetscCall(bcs[3].velocity(dim, t, xb, vb, bcs[3].ctx_velocity));

        h1 = arrcy[N - 1][inextc] - arrcy[N - 1][ielemc];
        h2 = arrcy[N - 1][ielemc] - arrcy[N - 2][ielemc];
        h3 = arrcy[N - 1][ielemc] - arrcy[N - 3][ielemc];

        for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, dim, row, v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvectionLinearInterpolationPrev_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_prev_f, PetscReal coord_center, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -0.5 * v_f / h * (coord_center - coord_prev_f) / (coord_center - coord_prev);
  v[1]     = -0.5 * v_f / h * (coord_prev_f - coord_prev) / (coord_center - coord_prev);
  col[0].i = i - (dir == DERIV_X ? 1 : 0);
  col[0].j = j - (dir == DERIV_Y ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvectionLinearInterpolationNext_Private(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_center, PetscReal coord_next_f, PetscReal coord_next, PetscReal h, PetscScalar v_f, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = 0.5 * v_f / h * (coord_next - coord_next_f) / (coord_next - coord_center);
  v[1]     = 0.5 * v_f / h * (coord_next_f - coord_center) / (coord_next - coord_center);
  col[0].i = i;
  col[0].j = j;
  col[1].i = i + (dir == DERIV_X ? 1 : 0);
  col[1].j = j + (dir == DERIV_Y ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvectionOperator_Private(DM vdm, DM Sdm, DM Vdm, Vec V0, Vec v0interp, const NSBoundaryCondition *bcs, Mat C)
{
  PetscInt             M, N, x, y, m, n;
  DMStagStencil        row, col[2];
  PetscInt             ncols;
  PetscScalar          v[2];
  Vec                  V0local, v0interplocal;
  const PetscScalar ***arrV0, ***arrv0interp;
  const PetscScalar  **arrcx, **arrcy;
  PetscReal            hx, hy;
  PetscInt             iV0[2], iv0interp[2][2], iprevc, ielemc;
  PetscInt             i, j, l, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(vdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMGetLocalVector(Sdm, &V0local));
  PetscCall(DMGetLocalVector(Vdm, &v0interplocal));

  PetscCall(DMGlobalToLocal(Sdm, V0, INSERT_VALUES, V0local));
  PetscCall(DMGlobalToLocal(Vdm, v0interp, INSERT_VALUES, v0interplocal));
  PetscCall(DMStagVecGetArrayRead(Sdm, V0local, &arrV0));
  PetscCall(DMStagGetLocationSlot(Sdm, DMSTAG_LEFT, 0, &iV0[0]));
  PetscCall(DMStagGetLocationSlot(Sdm, DMSTAG_DOWN, 0, &iV0[1]));
  PetscCall(DMStagVecGetArrayRead(Vdm, v0interplocal, &arrv0interp));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_LEFT, 0, &iv0interp[0][0]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_DOWN, 0, &iv0interp[0][1]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_LEFT, 1, &iv0interp[1][0]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_DOWN, 1, &iv0interp[1][1]));

  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  for (i = 0; i < 2; ++i) col[i].loc = DMSTAG_ELEMENT;

  /* N_i = (1/2) * d/dx_j (v_i * V0_j + v0interp_i * v_j) */
  for (c = 0; c < 2; ++c) {
    row.c = c;

    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        hx    = arrcx[i + 1][iprevc] - arrcx[i][iprevc];
        hy    = arrcy[j + 1][iprevc] - arrcy[j][iprevc];

        /* Left cell face */
        for (l = 0; l < 2; ++l) col[l].c = c;
        ncols = 0;
        if (i == 0) {
          /* Left boundary */
          switch (bcs[0].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrV0[j][i][iV0[0]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrV0[j][i][iV0[0]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

        for (l = 0; l < 2; ++l) col[l].c = 0;
        ncols = 0;
        if (i == 0) {
          /* Left boundary */
          switch (bcs[0].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrv0interp[j][i][iv0interp[c][0]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrv0interp[j][i][iv0interp[c][0]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

        /* Right cell face */
        for (l = 0; l < 2; ++l) col[l].c = c;
        ncols = 0;
        if (i == M - 1) {
          /* Right boundary */
          switch (bcs[1].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_X, i, j, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrV0[j][i + 1][iV0[0]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_X, i, j, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrV0[j][i + 1][iV0[0]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

        for (l = 0; l < 2; ++l) col[l].c = 0;
        ncols = 0;
        if (i == M - 1) {
          /* Right boundary */
          switch (bcs[1].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_X, i, j, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrv0interp[j][i + 1][iv0interp[c][0]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_X, i, j, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrv0interp[j][i + 1][iv0interp[c][0]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

        /* Bottom cell face */
        for (l = 0; l < 2; ++l) col[l].c = c;
        ncols = 0;
        if (j == 0) {
          /* Bottom boundary */
          switch (bcs[2].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrV0[j][i][iV0[1]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for bottom boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrV0[j][i][iV0[1]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

        for (l = 0; l < 2; ++l) col[l].c = 1;
        ncols = 0;
        if (j == 0) {
          /* Bottom boundary */
          switch (bcs[2].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrv0interp[j][i][iv0interp[c][1]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for bottom boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationPrev_Private(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrv0interp[j][i][iv0interp[c][1]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

        /* Top cell face */
        for (l = 0; l < 2; ++l) col[l].c = c;
        ncols = 0;
        if (j == 0) {
          /* Top boundary */
          switch (bcs[3].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_Y, i, j, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrV0[j + 1][i][iV0[1]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for top boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_Y, i, j, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrV0[j + 1][i][iV0[1]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

        for (l = 0; l < 2; ++l) col[l].c = 1;
        ncols = 0;
        if (j == 0) {
          /* Top boundary */
          switch (bcs[3].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_Y, i, j, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrv0interp[j + 1][i][iv0interp[c][1]], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for top boundary");
          }
        } else {
          PetscCall(ComputeConvectionLinearInterpolationNext_Private(DERIV_Y, i, j, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrv0interp[j + 1][i][iv0interp[c][1]], &ncols, col, v));
        }
        if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));
      }
  }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagVecRestoreArrayRead(Sdm, V0local, &arrV0));
  PetscCall(DMStagVecRestoreArrayRead(Vdm, v0interplocal, &arrv0interp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));

  PetscCall(DMRestoreLocalVector(Sdm, &V0local));
  PetscCall(DMRestoreLocalVector(Vdm, &v0interplocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvectionBoundaryConditionVector_Private(DM vdm, const NSBoundaryCondition *bcs, PetscReal t0, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
  PetscBool           isFirstRankx, isFirstRanky, isLastRankx, isLastRanky;
  DMStagStencil       row[2];
  PetscScalar         v[2];
  PetscReal           xb[2];
  PetscScalar         vb0[2], vb[2];
  const PetscScalar **arrcx, **arrcy;
  PetscReal           hx, hy;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(vdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(vdm, &isFirstRankx, &isFirstRanky, NULL));
  PetscCall(DMStagGetIsLastRank(vdm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  for (c = 0; c < 2; ++c) {
    row[c].loc = DMSTAG_ELEMENT;
    row[c].c   = c;
  }

  /* Left boundary */
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        for (c = 0; c < 2; ++c) {
          row[c].i = 0;
          row[c].j = j;
        }
        xb[0] = arrcx[0][iprevc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[0].velocity(2, t0, xb, vb0, bcs[0].ctx_velocity));
        PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));

        hx = arrcx[0][inextc] - arrcx[0][iprevc];

        for (c = 0; c < 2; ++c) v[c] = -0.5 * (vb[c] * vb0[0] + vb0[c] * vb[0]) / hx;
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 2, row, v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
    }

  /* Right boundary */
  if (isLastRankx) switch (bcs[1].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        for (c = 0; c < 2; ++c) {
          row[c].i = M - 1;
          row[c].j = j;
        }
        xb[0] = arrcx[M - 1][inextc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[1].velocity(2, t0, xb, vb0, bcs[1].ctx_velocity));
        PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));

        hx = arrcx[M - 1][inextc] - arrcx[M - 1][iprevc];

        for (c = 0; c < 2; ++c) v[c] = 0.5 * (vb[c] * vb0[0] + vb0[c] * vb[0]) / hx;
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 2, row, v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
    }

  /* Bottom boundary */
  if (isFirstRanky) switch (bcs[2].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        for (c = 0; c < 2; ++c) {
          row[c].i = i;
          row[c].j = 0;
        }
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[0][iprevc];
        PetscCall(bcs[2].velocity(2, t0, xb, vb0, bcs[2].ctx_velocity));
        PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));

        hy = arrcy[0][inextc] - arrcy[0][iprevc];

        for (c = 0; c < 2; ++c) v[c] = -0.5 * (vb[c] * vb0[1] + vb0[c] * vb[1]) / hy;
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 2, row, v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for bottom boundary");
    }

  /* Top boundary */
  if (isLastRanky) switch (bcs[3].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        for (c = 0; c < 2; ++c) {
          row[c].i = i;
          row[c].j = N - 1;
        }
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[N - 1][iprevc];
        PetscCall(bcs[3].velocity(2, t0, xb, vb0, bcs[3].ctx_velocity));
        PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));

        hy = arrcy[N - 1][inextc] - arrcy[N - 1][iprevc];

        for (c = 0; c < 2; ++c) v[c] = 0.5 * (vb[c] * vb0[1] + vb0[c] * vb[1]) / hy;
        PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 2, row, v, ADD_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for bottom boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
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

static PetscErrorCode ComputeFaceVelocityInterpolationOperator_Private(DM vdm, DM Vdm, const NSBoundaryCondition *bcs, Mat Tv)
{
  PetscInt            M, N, x, y, m, n, nExtrax, nExtray;
  DMStagStencil       row, col[2];
  PetscInt            ncols, ir, ic[2];
  PetscScalar         v[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j, c;
  const PetscInt      dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(vdm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Tv, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  for (i = 0; i < 2; ++i) col[i].loc = DMSTAG_ELEMENT;

  for (c = 0; c < dim; ++c) {
    row.c = c;
    for (i = 0; i < 2; ++i) col[i].c = c;

    /* Interpolation in x-direction */
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
          PetscCall(DMStagStencilToIndexLocal(Vdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(Tv, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }

    /* Interpolation in y-direction */
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
          PetscCall(DMStagStencilToIndexLocal(Vdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(Tv, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }
  }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Tv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Tv, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceVelocityInterpolationBoundaryConditionVector_Private(DM Vdm, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n, dim;
  PetscBool           isFirstRankx, isFirstRanky, isLastRankx, isLastRanky;
  DMStagStencil       row[2];
  PetscReal           xb[2];
  PetscScalar         vb[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(Vdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(Vdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(Vdm, &isFirstRankx, &isFirstRanky, NULL));
  PetscCall(DMStagGetIsLastRank(Vdm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(Vdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Vdm, DMSTAG_ELEMENT, &ielemc));
  PetscCall(DMGetDimension(Vdm, &dim));

  PetscCall(VecSet(vbc, 0.));

  for (c = 0; c < dim; ++c) row[c].c = c;

  /* Left boundary */
  for (c = 0; c < dim; ++c) row[c].loc = DMSTAG_LEFT;
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        for (c = 0; c < dim; ++c) {
          row[c].i = 0;
          row[c].j = j;
        }
        xb[0] = arrcx[0][iprevc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, dim, row, vb, INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
    }

  /* Right boundary */
  if (isLastRankx) switch (bcs[1].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        for (c = 0; c < dim; ++c) {
          row[c].i = M;
          row[c].j = j;
        }
        xb[0] = arrcx[M][iprevc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, dim, row, vb, INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
    }

  /* Down boundary */
  for (c = 0; c < dim; ++c) row[c].loc = DMSTAG_DOWN;
  if (isFirstRanky) switch (bcs[2].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        for (c = 0; c < dim; ++c) {
          row[c].i = i;
          row[c].j = 0;
        }
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[0][iprevc];
        PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, dim, row, vb, INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
    }

  /* Up boundary */
  if (isLastRanky) switch (bcs[3].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        for (c = 0; c < dim; ++c) {
          row[c].i = i;
          row[c].j = N;
        }
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[N][iprevc];
        PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, dim, row, vb, INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(Vdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceNormalVelocityInterpolationOperator_Private(DM vdm, DM Sdm, const NSBoundaryCondition *bcs, Mat Tv)
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
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(vdm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Tv, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.c = 0;
  for (i = 0; i < 2; ++i) col[i].loc = DMSTAG_ELEMENT;

  /* Interpolation in x-direction */
  row.loc = DMSTAG_LEFT;
  for (i = 0; i < 2; ++i) col[i].c = 0;

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
        PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Tv, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Compute y-interpolation operator */
  row.loc = DMSTAG_DOWN;
  for (i = 0; i < 2; ++i) col[i].c = 1;

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
        PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Tv, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Tv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Tv, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceNormalVelocityInterpolationBoundaryConditionVector_Private(DM Sdm, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
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
  PetscCall(DMStagGetGlobalSizes(Sdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(Sdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(Sdm, &isFirstRankx, &isFirstRanky, NULL));
  PetscCall(DMStagGetIsLastRank(Sdm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(Sdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Sdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Sdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  row.c = 0;

  /* Left boundary */
  row.loc = DMSTAG_LEFT;
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        row.i = 0;
        row.j = j;
        xb[0] = arrcx[0][iprevc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Sdm, vbc, 1, &row, &vb[0], INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
    }

  /* Right boundary */
  if (isLastRankx) switch (bcs[1].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        row.i = M;
        row.j = j;
        xb[0] = arrcx[M][iprevc];
        xb[1] = arrcy[j][ielemc];
        PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Sdm, vbc, 1, &row, &vb[0], INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
    }

  /* Down boundary */
  row.loc = DMSTAG_DOWN;
  if (isFirstRanky) switch (bcs[2].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = 0;
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[0][iprevc];
        PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Sdm, vbc, 1, &row, &vb[1], INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
    }

  /* Up boundary */
  if (isLastRanky) switch (bcs[3].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = N;
        xb[0] = arrcx[i][ielemc];
        xb[1] = arrcy[N][iprevc];
        PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));
        PetscCall(DMStagVecSetValuesStencil(Sdm, vbc, 1, &row, &vb[1], INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(Sdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeStaggeredVelocityDivergenceOperator_Private(DM Sdm, DM sdm, const NSBoundaryCondition *bcs, Mat Dstv)
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[4];
  PetscInt            ncols, ir, ic[4];
  PetscScalar         v[4];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc;
  PetscScalar         dx, dy;
  PetscInt            i, j;
  const PetscInt      dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(sdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(sdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_RIGHT, &inextc));

  PetscCall(MatSetOption(Dstv, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (i = 0; i < 4; ++i) col[i].c = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;
      ncols = 0;

      /* x-gradient */
      dx = arrcx[i][inextc] - arrcx[i][iprevc];

      col[ncols].i   = i;
      col[ncols].j   = j;
      col[ncols].loc = DMSTAG_LEFT;
      v[ncols]       = -1. / dx;
      ncols++;

      col[ncols].i   = i;
      col[ncols].j   = j;
      col[ncols].loc = DMSTAG_RIGHT;
      v[ncols]       = 1. / dx;
      ncols++;

      /* y-gradient */
      dy = arrcy[j][inextc] - arrcy[j][iprevc];

      col[ncols].i   = i;
      col[ncols].j   = j;
      col[ncols].loc = DMSTAG_DOWN;
      v[ncols]       = -1. / dy;
      ncols++;

      col[ncols].i   = i;
      col[ncols].j   = j;
      col[ncols].loc = DMSTAG_UP;
      v[ncols]       = 1. / dy;
      ncols++;

      PetscCall(DMStagStencilToIndexLocal(sdm, dim, 1, &row, &ir));
      PetscCall(DMStagStencilToIndexLocal(Sdm, dim, ncols, col, ic));
      PetscCall(MatSetValuesLocal(Dstv, 1, &ir, ncols, ic, v, INSERT_VALUES));
    }

  PetscCall(MatAssemblyBegin(Dstv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Dstv, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));
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

static PetscErrorCode ComputeStaggeredPressureGradientOperators_Private(DM sdm, DM Sdm, const NSBoundaryCondition *bcs, Mat Gstp)
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
  PetscCall(DMStagGetGlobalSizes(sdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(sdm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Gstp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.c = 0;
  for (i = 0; i < 2; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  /* x-gradient */
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
        PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Gstp, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* y-gradient */
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
        PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Gstp, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  PetscCall(MatAssemblyBegin(Gstp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstp, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMIterate2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      vdm, Vdm;
  IS      vis, Vis, pis;
  Vec     v0, v, V, dp, solv, solV, solp, solp0, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));
  PetscCall(NSGetField(ns, NS_FIELD_VELOCITY, NULL, &vis));
  PetscCall(NSGetField(ns, NS_FIELD_FACE_NORMAL_VELOCITY, NULL, &Vis));
  PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, &pis));

  if (!fsm->TvNcomputed) {
    PetscCall(ComputeFaceVelocityInterpolationOperator_Private(vdm, Vdm, ns->bcs, fsm->TvN));
    fsm->TvNcomputed = PETSC_TRUE;
  }
  PetscCall(DMGetGlobalVector(Vdm, &vbc));
  PetscCall(VecGetSubVector(ns->sol0, vis, &v0));
  PetscCall(MatMult(fsm->TvN, v0, fsm->v0interp));
  PetscCall(ComputeFaceVelocityInterpolationBoundaryConditionVector_Private(Vdm, ns->bcs, ns->t, vbc));
  PetscCall(VecAXPY(fsm->v0interp, 1., vbc));
  PetscCall(DMRestoreGlobalVector(Vdm, &vbc));
  PetscCall(VecRestoreSubVector(ns->sol0, vis, &v0));

  PetscCall(SNESSolve(ns->snes, NULL, ns->x));

  PetscCall(VecGetSubVector(ns->x, vis, &v));
  PetscCall(VecGetSubVector(ns->x, Vis, &V));
  PetscCall(VecGetSubVector(ns->x, pis, &dp));
  PetscCall(VecGetSubVector(ns->sol, vis, &solv));
  PetscCall(VecGetSubVector(ns->sol, Vis, &solV));
  PetscCall(VecGetSubVector(ns->sol, pis, &solp));
  PetscCall(VecGetSubVector(ns->sol0, pis, &solp0));

  PetscCall(VecCopy(v, solv));
  PetscCall(VecCopy(V, solV));

  PetscCall(VecAXPBYPCZ(solp, 1., 1., 0., solp0, dp));

  PetscCall(VecRestoreSubVector(ns->x, vis, &v));
  PetscCall(VecRestoreSubVector(ns->x, Vis, &V));
  PetscCall(VecRestoreSubVector(ns->x, pis, &dp));
  PetscCall(VecRestoreSubVector(ns->sol, vis, &solv));
  PetscCall(VecRestoreSubVector(ns->sol, Vis, &solV));
  PetscCall(VecRestoreSubVector(ns->sol, pis, &solp));
  PetscCall(VecRestoreSubVector(ns->sol0, pis, &solp0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMFormFunction_Cart_Internal(SNES snes, Vec x, Vec f, void *ctx)
{
  NS  ns = (NS)ctx;
  DM  vdm, Sdm, pdm;
  IS  vis, Vis, pis;
  Vec v0, V0, p0, momrhs, interprhs, contrhs;
  Mat Gp, Lv;
  Vec Gp_p0, Lv_v, vbc;

  PetscFunctionBegin;
  PetscCall(NSGetField(ns, NS_FIELD_VELOCITY, &vdm, &vis));
  PetscCall(NSGetField(ns, NS_FIELD_FACE_NORMAL_VELOCITY, &Sdm, &Vis));
  PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, &pdm, &pis));
  PetscCall(VecGetSubVector(ns->sol0, vis, &v0));
  PetscCall(VecGetSubVector(ns->sol0, Vis, &V0));
  PetscCall(VecGetSubVector(ns->sol0, pis, &p0));
  PetscCall(VecGetSubVector(f, vis, &momrhs));
  PetscCall(VecGetSubVector(f, Vis, &interprhs));
  PetscCall(VecGetSubVector(f, pis, &contrhs));

  PetscCall(MatCreateSubMatrix(ns->J, vis, pis, MAT_INITIAL_MATRIX, &Gp));
  PetscCall(PetscObjectQuery((PetscObject)ns->J, "Laplacian", (PetscObject *)&Lv));
  PetscCall(DMGetGlobalVector(vdm, &Gp_p0));
  PetscCall(DMGetGlobalVector(vdm, &Lv_v));
  PetscCall(DMGetGlobalVector(vdm, &vbc));
  PetscCall(MatMult(Gp, p0, Gp_p0));
  PetscCall(MatMult(Lv, v0, Lv_v));
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(vdm, ns->bcs, ns->t, vbc));
  PetscCall(VecAXPY(Lv_v, 1., vbc));
  PetscCall(VecAXPBYPCZ(momrhs, 1., 0.5 * ns->mu * ns->dt / ns->rho, 0., v0, Lv_v));
  PetscCall(ComputeConvectionBoundaryConditionVector_Private(vdm, ns->bcs, ns->t, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(momrhs, -ns->dt, vbc));
  PetscCall(VecAXPY(momrhs, -1., Gp_p0));
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(vdm, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(momrhs, 0.5 * ns->mu * ns->dt / ns->rho, vbc));
  PetscCall(MatDestroy(&Gp));
  PetscCall(DMRestoreGlobalVector(vdm, &Gp_p0));
  PetscCall(DMRestoreGlobalVector(vdm, &Lv_v));
  PetscCall(DMRestoreGlobalVector(vdm, &vbc));

  PetscCall(ComputeFaceNormalVelocityInterpolationBoundaryConditionVector_Private(Sdm, ns->bcs, ns->t + ns->dt, interprhs));
  PetscCall(VecScale(interprhs, -1.));

  PetscCall(VecSet(contrhs, 0.));

  PetscCall(VecRestoreSubVector(ns->sol0, vis, &v0));
  PetscCall(VecRestoreSubVector(ns->sol0, Vis, &V0));
  PetscCall(VecRestoreSubVector(ns->sol0, pis, &p0));
  PetscCall(VecRestoreSubVector(f, vis, &momrhs));
  PetscCall(VecRestoreSubVector(f, Vis, &interprhs));
  PetscCall(VecRestoreSubVector(f, pis, &contrhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMFormJacobian_Cart_Internal(SNES snes, Vec x, Mat J, Mat Jpre, void *ctx)
{
  NS               ns  = (NS)ctx;
  NS_FSM          *fsm = (NS_FSM *)ns->data;
  DM               sdm, vdm, Sdm, Vdm;
  IS               vis, Vis, pis;
  Mat              A, Gp, Tv, negI, TRC, Dstv, Lv, Gstp, TvGp, C;
  Vec              diag, V0;
  static PetscBool firstcalled = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetStaggeredScalarDM(ns->mesh, &Sdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));
  PetscCall(NSGetField(ns, NS_FIELD_VELOCITY, NULL, &vis));
  PetscCall(NSGetField(ns, NS_FIELD_FACE_NORMAL_VELOCITY, NULL, &Vis));
  PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, &pis));
  PetscCall(MatCreateSubMatrix(Jpre, vis, vis, MAT_INITIAL_MATRIX, &A));
  PetscCall(MatCreateSubMatrix(Jpre, vis, pis, MAT_INITIAL_MATRIX, &Gp));
  PetscCall(MatCreateSubMatrix(Jpre, Vis, vis, MAT_INITIAL_MATRIX, &Tv));
  PetscCall(MatCreateSubMatrix(Jpre, Vis, Vis, MAT_INITIAL_MATRIX, &negI));
  PetscCall(MatCreateSubMatrix(Jpre, Vis, pis, MAT_INITIAL_MATRIX, &TRC));
  PetscCall(MatCreateSubMatrix(Jpre, pis, Vis, MAT_INITIAL_MATRIX, &Dstv));

  if (firstcalled) {
    PetscCall(DMCreateMatrix(vdm, &Lv));
    PetscCall(ComputeVelocityLaplacianOperator_Private(vdm, ns->bcs, Lv));

    PetscCall(ComputePressureGradientOperator_Private(sdm, vdm, ns->bcs, Gp));
    PetscCall(MatScale(Gp, ns->dt / ns->rho));

    PetscCall(ComputeFaceNormalVelocityInterpolationOperator_Private(vdm, Sdm, ns->bcs, Tv));

    PetscCall(DMGetGlobalVector(Sdm, &diag));
    PetscCall(VecSet(diag, -1.));
    PetscCall(MatDiagonalSet(negI, diag, INSERT_VALUES));
    PetscCall(DMRestoreGlobalVector(Sdm, &diag));

    PetscCall(MatDuplicate(TRC, MAT_DO_NOT_COPY_VALUES, &Gstp));
    PetscCall(ComputeStaggeredPressureGradientOperators_Private(sdm, Sdm, ns->bcs, Gstp));
    PetscCall(MatScale(Gstp, ns->dt / ns->rho));
    PetscCall(MatMatMult(Tv, Gp, MAT_INITIAL_MATRIX, 1., &TvGp));
    PetscCall(MatCopy(TvGp, TRC, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatAXPY(TRC, -1., Gstp, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&TvGp));

    PetscCall(ComputeStaggeredVelocityDivergenceOperator_Private(Sdm, sdm, ns->bcs, Dstv));

    PetscCall(PetscObjectCompose((PetscObject)Jpre, "Laplacian", (PetscObject)Lv));
    PetscCall(PetscObjectCompose((PetscObject)Jpre, "StaggeredGradient", (PetscObject)Gstp));
    PetscCall(MatDestroy(&Lv));
    PetscCall(MatDestroy(&Gstp));

    firstcalled = PETSC_FALSE;
  }

  if (ns->sol0) {
    PetscCall(MatZeroEntries(A));
    PetscCall(DMGetGlobalVector(vdm, &diag));
    PetscCall(VecSet(diag, 1.));
    PetscCall(MatDiagonalSet(A, diag, INSERT_VALUES));
    PetscCall(PetscObjectQuery((PetscObject)Jpre, "Laplacian", (PetscObject *)&Lv));
    PetscCall(ComputeVelocityLaplacianOperator_Private(vdm, ns->bcs, Lv));
    PetscCall(MatAXPY(A, -0.5 * ns->mu * ns->dt / ns->rho, Lv, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &C));
    PetscCall(VecGetSubVector(ns->sol0, Vis, &V0));
    PetscCall(ComputeConvectionOperator_Private(vdm, Sdm, Vdm, V0, fsm->v0interp, ns->bcs, C));
    PetscCall(VecRestoreSubVector(ns->sol0, Vis, &V0));
    PetscCall(MatAXPY(A, ns->dt, C, DIFFERENT_NONZERO_PATTERN));
    PetscCall(DMRestoreGlobalVector(vdm, &diag));
    PetscCall(MatDestroy(&C));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Gp));
  PetscCall(MatDestroy(&Tv));
  PetscCall(MatDestroy(&negI));
  PetscCall(MatDestroy(&TRC));
  PetscCall(MatDestroy(&Dstv));

  /* Set null space. */
  if (ns->nullspace) PetscCall(MatSetNullSpace(J, ns->nullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}
