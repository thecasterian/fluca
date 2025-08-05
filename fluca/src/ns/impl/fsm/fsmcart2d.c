#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

typedef enum {
  DERIV_X,
  DERIV_Y,
} DerivDirection;

static PetscErrorCode NSFSMComputeIdentityOperator2d_Cart_Private(DM dm, Mat Id)
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

static PetscErrorCode ComputeFirstDerivForwardDiffNoCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_center, PetscReal coord_next, PetscReal coord_next2, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
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

static PetscErrorCode ComputeFirstDerivForwardDiffDirichletCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_bnd, PetscReal coord_center, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = coord_center - coord_bnd;
  h2       = coord_next - coord_center;
  v[0]     = (h2 - h1) / (h1 * h2);
  v[1]     = h1 / (h2 * (h1 + h2));
  col[0].i = i;
  col[0].j = j;
  col[1].i = i + (dir == DERIV_X ? 1 : 0);
  col[1].j = j + (dir == DERIV_Y ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFirstDerivCentralDiff(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
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

static PetscErrorCode ComputeFirstDerivBackwardDiffNoCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev2, PetscReal coord_prev, PetscReal coord_center, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
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

static PetscErrorCode ComputeFirstDerivBackwardDiffDirichletCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_center, PetscReal coord_bnd, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = coord_bnd - coord_center;
  h2       = coord_center - coord_prev;
  v[0]     = -h1 / (h2 * (h1 + h2));
  v[1]     = (h1 - h2) / (h1 * h2);
  col[0].i = i - (dir == DERIV_X ? 1 : 0);
  col[0].j = j - (dir == DERIV_Y ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputeVelocityGradientOperators2d_Cart_Private(DM dm, const NSBoundaryCondition *bcs, Mat G[])
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

  PetscCall(MatSetOption(G[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(G[1], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

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
          PetscCall(ComputeFirstDerivForwardDiffDirichletCond(DERIV_X, i, j, arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivBackwardDiffDirichletCond(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][ielemc], arrcx[i][inextc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, G[0], 1, &row, ncols, col, v, INSERT_VALUES));
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
          PetscCall(ComputeFirstDerivForwardDiffDirichletCond(DERIV_Y, i, j, arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivBackwardDiffDirichletCond(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][ielemc], arrcy[j][inextc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, G[1], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(G[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(G[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputePressureGradientOperators2d_Cart_Private(DM dm, const NSBoundaryCondition *bcs, Mat G[])
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

  PetscCall(MatSetOption(G[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(G[1], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

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
          PetscCall(ComputeFirstDerivForwardDiffNoCond(DERIV_X, i, j, arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivBackwardDiffNoCond(DERIV_X, i, j, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, G[0], 1, &row, ncols, col, v, INSERT_VALUES));
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
          PetscCall(ComputeFirstDerivForwardDiffNoCond(DERIV_Y, i, j, arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeFirstDerivBackwardDiffNoCond(DERIV_Y, i, j, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, G[1], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(G[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(G[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivForwardDiffDirichletCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_bnd, PetscReal coord_center, PetscReal coord_next, PetscReal coord_next2, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
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

static PetscErrorCode ComputeSecondDerivCentralDiff(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_prev_b, PetscReal coord_center, PetscReal coord_next_b, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
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

static PetscErrorCode ComputeSecondDerivBackwardDiffDirichletCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev2, PetscReal coord_prev, PetscReal coord_center, PetscReal coord_bnd, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
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

static PetscErrorCode NSFSMComputeVelocityLaplacianOperator2d_Cart_Private(DM dm, const NSBoundaryCondition *bcs, Mat L)
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

  PetscCall(MatSetOption(L, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

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
          PetscCall(ComputeSecondDerivForwardDiffDirichletCond(DERIV_X, i, j, arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffDirichletCond(DERIV_X, i, j, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], arrcx[i][inextc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
      }

      if (j == 0) {
        /* Down boundary */
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivForwardDiffDirichletCond(DERIV_Y, i, j, arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffDirichletCond(DERIV_Y, i, j, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], arrcy[j][inextc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, L, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputeVelocityLaplacianBoundaryConditionVector2d_Cart_Private(DM dm, PetscInt axis, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
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
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  /* Left boundary */
  switch (bcs[0].type) {
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
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, INSERT_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
  }

  /* Right boundary */
  switch (bcs[1].type) {
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
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, INSERT_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
  }

  /* Down boundary */
  switch (bcs[2].type) {
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
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, INSERT_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
  }

  /* Up boundary */
  switch (bcs[3].type) {
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
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, INSERT_VALUES));
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

static PetscErrorCode NSFSMComputeVelocityDivergenceBoundaryConditionVector2d_Cart_Private(DM dm, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row;
  PetscScalar         v;
  PetscReal           xb[2];
  PetscScalar         vb[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscReal           h1, h2;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  /* Left boundary */
  switch (bcs[0].type) {
  case NS_BC_VELOCITY:
    for (j = y; j < y + n; ++j) {
      row.i = 0;
      row.j = j;
      xb[0] = arrcx[0][iprevc];
      xb[1] = arrcy[j][ielemc];
      PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));

      h1 = arrcx[0][ielemc] - arrcx[0][iprevc];
      h2 = arrcx[1][ielemc] - arrcx[0][ielemc];

      v = -h2 / (h1 * (h1 + h2)) * vb[0];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
  }

  /* Right boundary */
  switch (bcs[1].type) {
  case NS_BC_VELOCITY:
    for (j = y; j < y + n; ++j) {
      row.i = M - 1;
      row.j = j;
      xb[0] = arrcx[M - 1][inextc];
      xb[1] = arrcy[j][ielemc];
      PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));

      h1 = arrcx[M - 1][inextc] - arrcx[M - 1][ielemc];
      h2 = arrcx[M - 1][ielemc] - arrcx[M - 2][ielemc];

      v = h2 / (h1 * (h1 + h2)) * vb[0];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
  }

  /* Down boundary */
  switch (bcs[2].type) {
  case NS_BC_VELOCITY:
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = 0;
      xb[0] = arrcx[i][ielemc];
      xb[1] = arrcy[0][iprevc];
      PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));

      h1 = arrcy[0][ielemc] - arrcy[0][iprevc];
      h2 = arrcy[1][ielemc] - arrcy[0][ielemc];

      v = -h2 / (h1 * (h1 + h2)) * vb[1];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
  }

  /* Up boundary */
  switch (bcs[3].type) {
  case NS_BC_VELOCITY:
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = N - 1;
      xb[0] = arrcx[i][ielemc];
      xb[1] = arrcy[N - 1][inextc];
      PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));

      h1 = arrcy[N - 1][inextc] - arrcy[N - 1][ielemc];
      h2 = arrcy[N - 1][ielemc] - arrcy[N - 2][ielemc];

      v = h2 / (h1 * (h1 + h2)) * vb[1];
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

static PetscErrorCode NSFSMComputeConvectionBoundaryConditionVector2d_Cart_Private(DM dm, PetscInt axis, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row;
  PetscScalar         v;
  PetscReal           xb[2];
  PetscScalar         vb[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc, ielemc;
  PetscReal           h1, h2;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  /* Left boundary */
  switch (bcs[0].type) {
  case NS_BC_VELOCITY:
    for (j = y; j < y + n; ++j) {
      row.i = 0;
      row.j = j;
      xb[0] = arrcx[0][iprevc];
      xb[1] = arrcy[j][ielemc];
      PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));

      h1 = arrcx[0][ielemc] - arrcx[0][iprevc];
      h2 = arrcx[1][ielemc] - arrcx[0][ielemc];

      v = -h2 / (h1 * (h1 + h2)) * vb[0] * vb[axis];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
  }

  /* Right boundary */
  switch (bcs[1].type) {
  case NS_BC_VELOCITY:
    for (j = y; j < y + n; ++j) {
      row.i = M - 1;
      row.j = j;
      xb[0] = arrcx[M - 1][inextc];
      xb[1] = arrcy[j][ielemc];
      PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));

      h1 = arrcx[M - 1][inextc] - arrcx[M - 1][ielemc];
      h2 = arrcx[M - 1][ielemc] - arrcx[M - 2][ielemc];

      v = h2 / (h1 * (h1 + h2)) * vb[0] * vb[axis];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
  }

  /* Down boundary */
  switch (bcs[2].type) {
  case NS_BC_VELOCITY:
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = 0;
      xb[0] = arrcx[i][ielemc];
      xb[1] = arrcy[0][iprevc];
      PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));

      h1 = arrcy[0][ielemc] - arrcy[0][iprevc];
      h2 = arrcy[1][ielemc] - arrcy[0][ielemc];

      v = -h2 / (h1 * (h1 + h2)) * vb[1] * vb[axis];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
    }
    break;
  case NS_BC_PERIODIC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
  }

  /* Up boundary */
  switch (bcs[3].type) {
  case NS_BC_VELOCITY:
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = N - 1;
      xb[0] = arrcx[i][ielemc];
      xb[1] = arrcy[N - 1][inextc];
      PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));

      h1 = arrcy[N - 1][inextc] - arrcy[N - 1][ielemc];
      h2 = arrcy[N - 1][ielemc] - arrcy[N - 2][ielemc];

      v = h2 / (h1 * (h1 + h2)) * vb[1] * vb[axis];
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

PetscErrorCode NSFSMComputeSpatialOperators2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(NSFSMComputeVelocityGradientOperators2d_Cart_Private(dm, ns->bcs, fsm->Gv));
  PetscCall(NSFSMComputePressureGradientOperators2d_Cart_Private(dm, ns->bcs, fsm->Gp));
  PetscCall(NSFSMComputeVelocityLaplacianOperator2d_Cart_Private(dm, ns->bcs, fsm->Lv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorsIntermediateVelocity2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  (void)J;

  KSPVCtx *kspvctx = (KSPVCtx *)ctx;
  NS       ns      = kspvctx->ns;
  DM       dm;
  Mat      L;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));

  PetscCall(DMCreateMatrix(dm, &L));
  PetscCall(NSFSMComputeVelocityLaplacianOperator2d_Cart_Private(dm, ns->bcs, L));

  PetscCall(NSFSMComputeIdentityOperator2d_Cart_Private(dm, Jpre));
  PetscCall(MatAXPY(Jpre, -0.5 * ns->mu * ns->dt / ns->rho, L, DIFFERENT_NONZERO_PATTERN));

  PetscCall(MatDestroy(&L));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSIntermediateVelocity2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  KSPVCtx *kspvctx = (KSPVCtx *)ctx;
  NS       ns      = kspvctx->ns;
  NS_FSM  *fsm     = (NS_FSM *)ns->data;
  DM       dm;
  Vec      Gp, Lv, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &Gp));
  PetscCall(DMGetGlobalVector(dm, &Lv));
  PetscCall(DMGetGlobalVector(dm, &vbc));

  PetscCall(MatMult(fsm->Gp[kspvctx->axis], fsm->p_half, Gp));
  PetscCall(MatMult(fsm->Lv, fsm->v[kspvctx->axis], Lv));
  PetscCall(NSFSMComputeVelocityLaplacianBoundaryConditionVector2d_Cart_Private(dm, kspvctx->axis, ns->bcs, ns->t, vbc));
  PetscCall(VecAXPY(Lv, 1., vbc));

  /* RHS of momentum equation */
  PetscCall(VecAXPBYPCZ(b, 1., 0.5 * ns->mu * ns->dt / ns->rho, 0., fsm->v[kspvctx->axis], Lv));
  PetscCall(VecAXPBYPCZ(b, -1.5 * ns->dt, 0.5 * ns->dt, 1., fsm->N[kspvctx->axis], fsm->N_prev[kspvctx->axis]));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, Gp));

  /* Add boundary condition */
  PetscCall(NSFSMComputeVelocityLaplacianBoundaryConditionVector2d_Cart_Private(dm, kspvctx->axis, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(b, 0.5 * ns->mu * ns->dt / ns->rho, vbc));

  PetscCall(DMRestoreGlobalVector(dm, &Gp));
  PetscCall(DMRestoreGlobalVector(dm, &Lv));
  PetscCall(DMRestoreGlobalVector(dm, &vbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorPressureCorrection2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS           ns = (NS)ctx;
  MPI_Comm     comm;
  DM           dm;
  Mat          Gv[2], Gp[2], GvGp;
  MatNullSpace nullspace;
  PetscInt     d;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));

  PetscCall(DMCreateMatrix(dm, &Gv[0]));
  PetscCall(DMCreateMatrix(dm, &Gv[1]));
  PetscCall(DMCreateMatrix(dm, &Gp[0]));
  PetscCall(DMCreateMatrix(dm, &Gp[1]));
  PetscCall(NSFSMComputeVelocityGradientOperators2d_Cart_Private(dm, ns->bcs, Gv));
  PetscCall(NSFSMComputePressureGradientOperators2d_Cart_Private(dm, ns->bcs, Gp));

  /* G_x^(v) * G_x^(p) + G_y^(v) * G_y^(p) */
  PetscCall(MatZeroEntries(Jpre));
  for (d = 0; d < 2; ++d) {
    PetscCall(MatMatMult(Gv[d], Gp[d], MAT_INITIAL_MATRIX, 1., &GvGp));
    PetscCall(MatAXPY(Jpre, 1., GvGp, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&GvGp));
  }

  PetscCall(MatScale(Jpre, ns->dt / ns->rho));

  PetscCall(MatDestroy(&Gv[0]));
  PetscCall(MatDestroy(&Gv[1]));
  PetscCall(MatDestroy(&Gp[0]));
  PetscCall(MatDestroy(&Gp[1]));

  // TODO: below is temporary for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(J, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSPressureCorrection2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Vec     vbc;

  MPI_Comm     comm;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(DMGetGlobalVector(dm, &vbc));

  PetscCall(MatMult(fsm->Gv[0], fsm->v_star[0], b));
  PetscCall(MatMultAdd(fsm->Gv[1], fsm->v_star[1], b, b));

  PetscCall(NSFSMComputeVelocityDivergenceBoundaryConditionVector2d_Cart_Private(dm, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(b, 1., vbc));

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
    PetscCall(KSPSetComputeOperators(fsm->kspv[d], ComputeOperatorsIntermediateVelocity2d_Private, &fsm->kspvctx[d]));
    PetscCall(KSPSetComputeRHS(fsm->kspv[d], ComputeRHSIntermediateVelocity2d_Private, &fsm->kspvctx[d]));
  }
  PetscCall(KSPSetComputeOperators(fsm->kspp, ComputeOperatorPressureCorrection2d_Private, ns));
  PetscCall(KSPSetComputeRHS(fsm->kspp, ComputeRHSPressureCorrection2d_Private, ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMCalculateConvection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Vec     uu, uv, vv, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &uu));
  PetscCall(DMGetGlobalVector(dm, &uv));
  PetscCall(DMGetGlobalVector(dm, &vv));
  PetscCall(DMGetGlobalVector(dm, &vbc));

  PetscCall(VecPointwiseMult(uu, fsm->v[0], fsm->v[0]));
  PetscCall(VecPointwiseMult(uv, fsm->v[0], fsm->v[1]));
  PetscCall(VecPointwiseMult(vv, fsm->v[1], fsm->v[1]));

  /* Nx = d(uu)/dx + d(uv)/dy */
  PetscCall(MatMult(fsm->Gv[0], uu, fsm->N[0]));
  PetscCall(MatMultAdd(fsm->Gv[1], uv, fsm->N[0], fsm->N[0]));
  PetscCall(NSFSMComputeConvectionBoundaryConditionVector2d_Cart_Private(dm, 0, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(fsm->N[0], 1., vbc));

  /* Ny = d(uv)/dx + d(vv)/dy */
  PetscCall(MatMult(fsm->Gv[0], uv, fsm->N[1]));
  PetscCall(MatMultAdd(fsm->Gv[1], vv, fsm->N[1], fsm->N[1]));
  PetscCall(NSFSMComputeConvectionBoundaryConditionVector2d_Cart_Private(dm, 1, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(fsm->N[1], 1., vbc));

  PetscCall(DMRestoreGlobalVector(dm, &uu));
  PetscCall(DMRestoreGlobalVector(dm, &uv));
  PetscCall(DMRestoreGlobalVector(dm, &vv));
  PetscCall(DMRestoreGlobalVector(dm, &vbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  DM       dm;
  Vec      s;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    PetscCall(KSPSolve(fsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(fsm->kspv[d], &s));
    PetscCall(VecCopy(s, fsm->v_star[d]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMCalculatePressureCorrection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Vec     s;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(KSPSolve(fsm->kspp, NULL, NULL));
  PetscCall(KSPGetSolution(fsm->kspp, &s));
  PetscCall(VecCopy(s, fsm->p_prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMUpdate2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Mat     A;
  Vec     Gpp[2];

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(KSPGetOperators(fsm->kspv[0], &A, NULL));
  PetscCall(DMGetGlobalVector(dm, &Gpp[0]));
  PetscCall(DMGetGlobalVector(dm, &Gpp[1]));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecCopy(fsm->N[1], fsm->N_prev[1]));

  PetscCall(MatMult(fsm->Gp[0], fsm->p_prime, Gpp[0]));
  PetscCall(MatMult(fsm->Gp[1], fsm->p_prime, Gpp[1]));

  PetscCall(VecWAXPY(fsm->v[0], -ns->dt / ns->rho, Gpp[0], fsm->v_star[0]));
  PetscCall(VecWAXPY(fsm->v[1], -ns->dt / ns->rho, Gpp[1], fsm->v_star[1]));

  PetscCall(MatMultAdd(A, fsm->p_prime, fsm->p_half_prev, fsm->p_half));
  PetscCall(VecAXPBYPCZ(fsm->p, 1.5, -0.5, 0., fsm->p_half, fsm->p_half_prev));

  PetscCall(DMRestoreGlobalVector(dm, &Gpp[0]));
  PetscCall(DMRestoreGlobalVector(dm, &Gpp[1]));

  PetscCall(NSFSMCalculateConvection2d_Cart_Internal(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMIterate2d_Cart_Internal(NS ns)
{
  PetscFunctionBegin;
  PetscCall(NSFSMCalculateIntermediateVelocity2d_Cart_Internal(ns));
  PetscCall(NSFSMCalculatePressureCorrection2d_Cart_Internal(ns));
  PetscCall(NSFSMUpdate2d_Cart_Internal(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}
