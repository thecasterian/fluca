#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

typedef enum {
  DERIV_X,
  DERIV_Y,
} DerivDirection;

typedef enum {
  ALL_X,
  ALL_Y,
  PER_DIR,
} InterpolationType;

static PetscErrorCode ComputeIdentityOperator_Private(DM dm, Mat Id)
{
  PetscInt      x, y, m, n, dim;
  DMStagStencil row, col;
  PetscScalar   v;
  PetscInt      i, j, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMGetDimension(dm, &dim));

  row.loc = DMSTAG_ELEMENT;
  col.loc = DMSTAG_ELEMENT;

  /* Set identity for each vector component */
  for (c = 0; c < dim; ++c) {
    row.c = c;
    col.c = c;
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        col.i = i;
        col.j = j;
        v     = 1.;
        PetscCall(DMStagMatSetValuesStencil(dm, Id, 1, &row, 1, &col, &v, INSERT_VALUES));
      }
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

static PetscErrorCode ComputeVelocityInterpolationOperator_Private(DM vdm, DM Vdm, const NSBoundaryCondition *bcs, InterpolationType interp_type, Mat Tv)
{
  PetscInt            M, N, x, y, m, n, nExtrax, nExtray;
  DMStagStencil       row, col[2];
  PetscInt            cx, cy, ncols, ir, ic[2];
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
  switch (interp_type) {
  case ALL_X:
    cx = 0;
    break;
  case ALL_Y:
    cx = 1;
    break;
  case PER_DIR:
    cx = 0;
    break;
  }
  for (i = 0; i < 2; ++i) col[i].c = cx;

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

  /* Compute y-interpolation operator */
  row.loc = DMSTAG_DOWN;
  switch (interp_type) {
  case ALL_X:
    cy = 0;
    break;
  case ALL_Y:
    cy = 1;
    break;
  case PER_DIR:
    cy = 1;
    break;
  }
  for (i = 0; i < 2; ++i) col[i].c = cy;

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

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Tv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Tv, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityInterpolationBoundaryConditionVector_Private(DM vdm, DM Vdm, const NSBoundaryCondition *bcs, PetscReal t, InterpolationType interp_type, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
  PetscBool           isFirstRankx, isFirstRanky, isLastRankx, isLastRanky;
  PetscInt            cx, cy;
  DMStagStencil       row;
  PetscReal           xb[2];
  PetscScalar         vb[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(vdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(vdm, &isFirstRankx, &isFirstRanky, NULL));
  PetscCall(DMStagGetIsLastRank(vdm, &isLastRankx, &isLastRanky, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  row.c = 0;
  switch (interp_type) {
  case ALL_X:
    cx = 0;
    cy = 0;
    break;
  case ALL_Y:
    cx = 1;
    cy = 1;
    break;
  case PER_DIR:
    cx = 0;
    cy = 1;
    break;
  }

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
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, 1, &row, &vb[cx], INSERT_VALUES));
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
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, 1, &row, &vb[cx], INSERT_VALUES));
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
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, 1, &row, &vb[cy], INSERT_VALUES));
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
        PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, 1, &row, &vb[cy], INSERT_VALUES));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeStaggeredVelocityDivergenceOperator_Private(DM Vdm, DM sdm, const NSBoundaryCondition *bcs, Mat Dstv)
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
      PetscCall(DMStagStencilToIndexLocal(Vdm, dim, ncols, col, ic));
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

static PetscErrorCode ComputeStaggeredPressureGradientOperators_Private(DM sdm, DM Vdm, const NSBoundaryCondition *bcs, Mat Gstp)
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
        PetscCall(DMStagStencilToIndexLocal(Vdm, dim, 1, &row, &ir));
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
        PetscCall(DMStagStencilToIndexLocal(Vdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(Gstp, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  PetscCall(MatAssemblyBegin(Gstp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstp, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));
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
  DM      sdm, vdm, Vdm;

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  PetscCall(ComputePressureGradientOperator_Private(sdm, vdm, ns->bcs, fsm->Gp));
  PetscCall(ComputeVelocityLaplacianOperator_Private(vdm, ns->bcs, fsm->Lv));
  PetscCall(ComputeVelocityInterpolationOperator_Private(vdm, Vdm, ns->bcs, PER_DIR, fsm->Tv));
  PetscCall(ComputeStaggeredVelocityDivergenceOperator_Private(Vdm, sdm, ns->bcs, fsm->Dstv));
  PetscCall(ComputeStaggeredPressureGradientOperators_Private(sdm, Vdm, ns->bcs, fsm->Gstp));
  PetscCall(ComputeVelocityInterpolationOperator_Private(vdm, Vdm, ns->bcs, ALL_X, fsm->TvN[0]));
  PetscCall(ComputeVelocityInterpolationOperator_Private(vdm, Vdm, ns->bcs, ALL_Y, fsm->TvN[1]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorsIntermediateVelocity_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS  ns = (NS)ctx;
  DM  dm;
  Mat Lv;

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
  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      vdm;
  Vec     Gp, Lv, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));

  PetscCall(DMGetGlobalVector(vdm, &Gp));
  PetscCall(DMGetGlobalVector(vdm, &Lv));
  PetscCall(DMGetGlobalVector(vdm, &vbc));

  PetscCall(MatMult(fsm->Gp, fsm->p_half, Gp));
  PetscCall(MatMult(fsm->Lv, fsm->v, Lv));
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(vdm, ns->bcs, ns->t, vbc));
  PetscCall(VecAXPY(Lv, 1., vbc));

  /* RHS of momentum equation */
  PetscCall(VecAXPBYPCZ(b, 1., 0.5 * ns->mu * ns->dt / ns->rho, 0., fsm->v, Lv));
  PetscCall(VecAXPBYPCZ(b, -1.5 * ns->dt, 0.5 * ns->dt, 1., fsm->N, fsm->N_prev));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, Gp));

  /* Add boundary condition */
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(vdm, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(b, 0.5 * ns->mu * ns->dt / ns->rho, vbc));

  PetscCall(DMRestoreGlobalVector(vdm, &Gp));
  PetscCall(DMRestoreGlobalVector(vdm, &Lv));
  PetscCall(DMRestoreGlobalVector(vdm, &vbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorPressureCorrection_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS           ns = (NS)ctx;
  MPI_Comm     comm;
  DM           sdm, Vdm;
  Mat          Dstv, Gstp, mmult;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCall(KSPGetDM(ksp, &sdm));
  PetscCall(DMStagCreateCompatibleDMStag(sdm, 0, 1, 0, 0, &Vdm));
  PetscCall(DMSetMatrixPreallocateOnly(sdm, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(Vdm, PETSC_TRUE));

  PetscCall(CreateOperatorFromDMToDM_Private(Vdm, sdm, &Dstv));
  PetscCall(CreateOperatorFromDMToDM_Private(sdm, Vdm, &Gstp));
  PetscCall(ComputeStaggeredVelocityDivergenceOperator_Private(Vdm, sdm, ns->bcs, Dstv));
  PetscCall(ComputeStaggeredPressureGradientOperators_Private(sdm, Vdm, ns->bcs, Gstp));

  /* dt / rho * Dstv * Gstp */
  PetscCall(MatMatMult(Dstv, Gstp, MAT_INITIAL_MATRIX, 1., &mmult));
  PetscCall(MatScale(mmult, ns->dt / ns->rho));
  PetscCall(MatCopy(mmult, Jpre, DIFFERENT_NONZERO_PATTERN));

  PetscCall(MatDestroy(&Dstv));
  PetscCall(MatDestroy(&Gstp));
  PetscCall(MatDestroy(&mmult));
  PetscCall(DMDestroy(&Vdm));

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
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscCall(KSPSetComputeOperators(fsm->kspv, ComputeOperatorsIntermediateVelocity_Private, ns));
  PetscCall(KSPSetComputeRHS(fsm->kspv, ComputeRHSIntermediateVelocity_Private, ns));
  PetscCall(KSPSetComputeOperators(fsm->kspp, ComputeOperatorPressureCorrection_Private, ns));
  PetscCall(KSPSetComputeRHS(fsm->kspp, ComputeRHSPressureCorrection_Private, ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeScalarDMVectors_Private(DM sdm, DM vdm, Vec s[], Vec v)
{
  PetscInt             M, N, x, y, m, n;
  Vec                  sl[2], vl;
  const PetscScalar ***arrs[2];
  PetscScalar       ***arrv;
  PetscInt             ielems, ielemv[2], i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(sdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(sdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMGetLocalVector(sdm, &sl[0]));
  PetscCall(DMGetLocalVector(sdm, &sl[1]));
  PetscCall(DMGetLocalVector(vdm, &vl));
  PetscCall(DMGlobalToLocal(sdm, s[0], INSERT_VALUES, sl[0]));
  PetscCall(DMGlobalToLocal(sdm, s[1], INSERT_VALUES, sl[1]));
  PetscCall(DMGlobalToLocal(vdm, v, INSERT_VALUES, vl));

  PetscCall(DMStagVecGetArrayRead(sdm, sl[0], &arrs[0]));
  PetscCall(DMStagVecGetArrayRead(sdm, sl[1], &arrs[1]));
  PetscCall(DMStagVecGetArray(vdm, vl, &arrv));
  PetscCall(DMStagGetLocationSlot(sdm, DMSTAG_ELEMENT, 0, &ielems));
  PetscCall(DMStagGetLocationSlot(vdm, DMSTAG_ELEMENT, 0, &ielemv[0]));
  PetscCall(DMStagGetLocationSlot(vdm, DMSTAG_ELEMENT, 1, &ielemv[1]));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      arrv[j][i][ielemv[0]] = arrs[0][j][i][ielems];
      arrv[j][i][ielemv[1]] = arrs[1][j][i][ielems];
    }

  PetscCall(DMStagVecRestoreArrayRead(sdm, sl[0], &arrs[0]));
  PetscCall(DMStagVecRestoreArrayRead(sdm, sl[1], &arrs[1]));
  PetscCall(DMStagVecRestoreArray(vdm, vl, &arrv));

  PetscCall(DMLocalToGlobal(sdm, sl[0], INSERT_VALUES, s[0]));
  PetscCall(DMLocalToGlobal(sdm, sl[1], INSERT_VALUES, s[1]));
  PetscCall(DMLocalToGlobal(vdm, vl, INSERT_VALUES, v));
  PetscCall(DMRestoreLocalVector(sdm, &sl[0]));
  PetscCall(DMRestoreLocalVector(sdm, &sl[1]));
  PetscCall(DMRestoreLocalVector(vdm, &vl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvection_Private(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      sdm, vdm, Vdm;
  Vec     v_interp[2], vbc, vmult, N[2];

  PetscFunctionBegin;
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  PetscCall(DMGetGlobalVector(Vdm, &v_interp[0]));
  PetscCall(DMGetGlobalVector(Vdm, &v_interp[1]));
  PetscCall(DMGetGlobalVector(Vdm, &vbc));
  PetscCall(DMGetGlobalVector(Vdm, &vmult));
  PetscCall(DMGetGlobalVector(sdm, &N[0]));
  PetscCall(DMGetGlobalVector(sdm, &N[1]));

  PetscCall(MatMult(fsm->TvN[0], fsm->v, v_interp[0]));
  PetscCall(ComputeVelocityInterpolationBoundaryConditionVector_Private(vdm, Vdm, ns->bcs, ns->t + ns->dt, ALL_X, vbc));
  PetscCall(VecAXPY(v_interp[0], 1., vbc));
  PetscCall(MatMult(fsm->TvN[1], fsm->v, v_interp[1]));
  PetscCall(ComputeVelocityInterpolationBoundaryConditionVector_Private(vdm, Vdm, ns->bcs, ns->t + ns->dt, ALL_Y, vbc));
  PetscCall(VecAXPY(v_interp[1], 1., vbc));

  /* N_i = d(U * v_i)/dx + d(V * v_i)/dy */
  PetscCall(VecPointwiseMult(vmult, fsm->V, v_interp[0]));
  PetscCall(MatMult(fsm->Dstv, vmult, N[0]));
  PetscCall(VecPointwiseMult(vmult, fsm->V, v_interp[1]));
  PetscCall(MatMult(fsm->Dstv, vmult, N[1]));
  PetscCall(MergeScalarDMVectors_Private(sdm, vdm, N, fsm->N));

  PetscCall(DMRestoreGlobalVector(Vdm, &v_interp[0]));
  PetscCall(DMRestoreGlobalVector(Vdm, &v_interp[1]));
  PetscCall(DMRestoreGlobalVector(Vdm, &vbc));
  PetscCall(DMRestoreGlobalVector(Vdm, &vmult));
  PetscCall(DMRestoreGlobalVector(sdm, &N[0]));
  PetscCall(DMRestoreGlobalVector(sdm, &N[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeIntermediateVelocity_Private(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      vdm, Vdm;
  Vec     s, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  /* Solve for cell-centered intermediate velocity. */
  PetscCall(KSPSolve(fsm->kspv, NULL, NULL));
  PetscCall(KSPGetSolution(fsm->kspv, &s));
  PetscCall(VecCopy(s, fsm->v_star));

  /* Compute face intermediate velocity. */
  PetscCall(DMGetGlobalVector(Vdm, &vbc));

  PetscCall(MatMult(fsm->Tv, fsm->v_star, fsm->V_star));
  PetscCall(ComputeVelocityInterpolationBoundaryConditionVector_Private(vdm, Vdm, ns->bcs, ns->t + ns->dt, PER_DIR, vbc));
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
  DM      vdm, sdm, Vdm;
  Vec     Gpp, Gstpp;

  PetscFunctionBegin;
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  PetscCall(DMGetGlobalVector(vdm, &Gpp));
  PetscCall(DMGetGlobalVector(Vdm, &Gstpp));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N, fsm->N_prev));

  PetscCall(MatMult(fsm->Gp, fsm->p_prime, Gpp));
  PetscCall(MatMult(fsm->Gstp, fsm->p_prime, Gstpp));

  PetscCall(VecWAXPY(fsm->v, -ns->dt / ns->rho, Gpp, fsm->v_star));
  PetscCall(VecWAXPY(fsm->V, -ns->dt / ns->rho, Gstpp, fsm->V_star));

  PetscCall(VecWAXPY(fsm->p_half, 1., fsm->p_half_prev, fsm->p_prime));
  PetscCall(VecAXPBYPCZ(fsm->p, 1.5, -0.5, 0., fsm->p_half, fsm->p_half_prev));

  PetscCall(DMRestoreGlobalVector(vdm, &Gpp));
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
