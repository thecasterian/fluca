#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

typedef enum {
  DERIV_X,
  DERIV_Y,
} DerivDirection;

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

static PetscErrorCode ComputeFirstDerivForwardDiffNeumannCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_bnd, PetscReal coord_center, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = coord_center - coord_bnd;
  h2       = coord_next - coord_center;
  v[0]     = -2. * h1 / (h2 * (2. * h1 + h2));
  v[1]     = 2. * h1 / (h2 * (2. * h1 + h2));
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

static PetscErrorCode ComputeFirstDerivBackwardDiffNeumannCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_center, PetscReal coord_bnd, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = coord_bnd - coord_center;
  h2       = coord_center - coord_prev;
  v[0]     = -2. * h1 / (h2 * (2. * h1 + h2));
  v[1]     = 2. * h1 / (h2 * (2. * h1 + h2));
  col[0].i = i - (dir == DERIV_X ? 1 : 0);
  col[0].j = j - (dir == DERIV_Y ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFirstDerivFace(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -1. / (coord_next - coord_prev);
  v[1]     = 1. / (coord_next - coord_prev);
  col[0].i = i - (dir == DERIV_X ? 1 : 0);
  col[0].j = j - (dir == DERIV_Y ? 1 : 0);
  col[1].i = i;
  col[1].j = j;
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputePressureGradientOperators2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, Mat grad[])
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[5];
  PetscInt            ncols;
  PetscScalar         v[3];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

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

      PetscCall(DMStagMatSetValuesStencil(dm, grad[0], 1, &row, ncols, col, v, INSERT_VALUES));
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

      PetscCall(DMStagMatSetValuesStencil(dm, grad[1], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(grad[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(grad[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(grad[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(grad[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputePressureCorrectionGradientOperators2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, Mat grad[])
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
          PetscCall(ComputeFirstDerivForwardDiffNeumannCond(DERIV_X, i, j, arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
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
          PetscCall(ComputeFirstDerivBackwardDiffNeumannCond(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][ielemc], arrcx[i][inextc], &ncols, col, v));
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

      PetscCall(DMStagMatSetValuesStencil(dm, grad[0], 1, &row, ncols, col, v, INSERT_VALUES));
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
          PetscCall(ComputeFirstDerivForwardDiffNeumannCond(DERIV_Y, i, j, arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
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
          PetscCall(ComputeFirstDerivBackwardDiffNeumannCond(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][ielemc], arrcy[j][inextc], &ncols, col, v));
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

      PetscCall(DMStagMatSetValuesStencil(dm, grad[1], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(grad[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(grad[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(grad[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(grad[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputeFaceGradientOperator2d_Cart_Internal(DM dm, DM fdm, const NSBoundaryCondition *bcs, Mat grad)
{
  PetscInt            M, N, x, y, m, n, nExtrax, nExtray;
  DMStagStencil       row, col[2];
  PetscInt            ncols;
  PetscScalar         v[2];
  PetscInt            ir, ic[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(fdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(fdm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  row.c = 0;
  for (i = 0; i < 2; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      row.i = i;
      row.j = j;

      /* x-gradient */
      row.loc = DMSTAG_LEFT;
      if (i == 0) {
        switch (bcs[0].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivFace(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M) {
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Periodic boundary condition on right boundary but mesh is not periodic in x-direction");
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivFace(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, 2, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, 2, ncols, col, ic));
        PetscCall(MatSetValuesLocal(grad, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }

      /* y-gradient */
      row.loc = DMSTAG_DOWN;
      if (j == 0) {
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeFirstDerivFace(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N) {
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Periodic boundary condition on up boundary but mesh is not periodic in y-direction");
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeFirstDerivFace(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, 2, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, 2, ncols, col, ic));
        PetscCall(MatSetValuesLocal(grad, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(grad, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(grad, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivForwardDiffDirichletCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_bnd, PetscReal coord_center, PetscReal coord_next, PetscReal coord_next2, PetscScalar scale, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = coord_center - coord_bnd;
  h2 = coord_next - coord_center;
  h3 = coord_next2 - coord_center;

  v[0] += 2. * (h1 - h2 - h3) / (h1 * h2 * h3) * scale;
  v[*ncols]     = 2. * (h1 - h3) / (h2 * (h1 + h2) * (h2 - h3)) * scale;
  v[*ncols + 1] = 2. * (h2 - h1) / (h3 * (h1 + h3) * (h2 - h3)) * scale;

  col[*ncols].i     = i + (dir == DERIV_X ? 1 : 0);
  col[*ncols].j     = j + (dir == DERIV_Y ? 1 : 0);
  col[*ncols + 1].i = i + (dir == DERIV_X ? 2 : 0);
  col[*ncols + 1].j = j + (dir == DERIV_Y ? 2 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivForwardDiffNeumannCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_bnd, PetscReal coord_center, PetscReal coord_next_b, PetscReal coord_next, PetscScalar scale, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1 = coord_next - coord_center;
  h2 = coord_next_b - coord_bnd;

  v[0] -= 1. / (h1 * h2) * scale;
  v[*ncols] = 1. / (h1 * h2) * scale;

  col[*ncols].i = i + (dir == DERIV_X ? 1 : 0);
  col[*ncols].j = j + (dir == DERIV_Y ? 1 : 0);
  *ncols += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivCentralDiff(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_prev_b, PetscReal coord_center, PetscReal coord_next_b, PetscReal coord_next, PetscScalar scale, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = coord_center - coord_prev;
  h2 = coord_next - coord_center;
  h3 = coord_next_b - coord_prev_b;

  v[0] -= (1. / (h1 * h3) + 1. / (h2 * h3)) * scale;
  v[*ncols]     = 1. / (h1 * h3) * scale;
  v[*ncols + 1] = 1. / (h2 * h3) * scale;

  col[*ncols].i     = i - (dir == DERIV_X ? 1 : 0);
  col[*ncols].j     = j - (dir == DERIV_Y ? 1 : 0);
  col[*ncols + 1].i = i + (dir == DERIV_X ? 1 : 0);
  col[*ncols + 1].j = j + (dir == DERIV_Y ? 1 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivBackwardDiffDirichletCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev2, PetscReal coord_prev, PetscReal coord_center, PetscReal coord_bnd, PetscScalar scale, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2, h3;

  PetscFunctionBegin;
  h1 = coord_bnd - coord_center;
  h2 = coord_center - coord_prev;
  h3 = coord_center - coord_prev2;

  v[0] += 2. * (h1 - h2 - h3) / (h1 * h2 * h3) * scale;
  v[*ncols]     = 2. * (h1 - h3) / (h2 * (h1 + h2) * (h2 - h3)) * scale;
  v[*ncols + 1] = 2. * (h2 - h1) / (h3 * (h1 + h3) * (h2 - h3)) * scale;

  col[*ncols].i     = i - (dir == DERIV_X ? 1 : 0);
  col[*ncols].j     = j - (dir == DERIV_Y ? 1 : 0);
  col[*ncols + 1].i = i - (dir == DERIV_X ? 2 : 0);
  col[*ncols + 1].j = j - (dir == DERIV_Y ? 2 : 0);
  *ncols += 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeSecondDerivBackwardDiffNeumannCond(DerivDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_prev_b, PetscReal coord_center, PetscReal coord_bnd, PetscScalar scale, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1 = coord_center - coord_prev;
  h2 = coord_bnd - coord_prev_b;

  v[0] -= 1. / (h1 * h2) * scale;
  v[*ncols] = 1. / (h1 * h2) * scale;

  col[*ncols].i = i - (dir == DERIV_X ? 1 : 0);
  col[*ncols].j = j - (dir == DERIV_Y ? 1 : 0);
  *ncols += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  * Computes the Helmholtz operator for the velocity field in a matrix form:
  *   (coeff + scale * \nabla^2) v.
  */
static PetscErrorCode NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, PetscScalar coeff, PetscScalar scale, Mat helm)
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
      v[0]     = coeff;
      ncols    = 1;

      if (i == 0) {
        /* Left boundary */
        switch (bcs[0].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivForwardDiffDirichletCond(DERIV_X, i, j, arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], scale, &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], scale, &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffDirichletCond(DERIV_X, i, j, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], arrcx[i][inextc], scale, &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], scale, &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], scale, &ncols, col, v));
      }

      if (j == 0) {
        /* Down boundary */
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivForwardDiffDirichletCond(DERIV_Y, i, j, arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], scale, &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], scale, &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffDirichletCond(DERIV_Y, i, j, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], arrcy[j][inextc], scale, &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], scale, &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], scale, &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, helm, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(helm, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(helm, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(DM dm, PetscInt dim, const NSBoundaryCondition *bcs, PetscReal t, PetscReal scale, InsertMode mode, Vec vbc)
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

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, mode));
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

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, mode));
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

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, mode));
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

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, mode));
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

static PetscErrorCode NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, Mat helm)
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
          PetscCall(ComputeSecondDerivForwardDiffNeumannCond(DERIV_X, i, j, arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], 1., &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], 1., &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffNeumannCond(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], 1., &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], 1., &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff(DERIV_X, i, j, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], 1., &ncols, col, v));
      }

      if (j == 0) {
        /* Down boundary */
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivForwardDiffNeumannCond(DERIV_Y, i, j, arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], 1., &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], 1., &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(ComputeSecondDerivBackwardDiffNeumannCond(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], 1., &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], 1., &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(ComputeSecondDerivCentralDiff(DERIV_Y, i, j, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], 1., &ncols, col, v));
      }

      PetscCall(DMStagMatSetValuesStencil(dm, helm, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(helm, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(helm, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputeVelocityInterpolationOperators2d_Cart_Internal(DM dm, DM fdm, const NSBoundaryCondition *bcs, Mat interp[])
{
  PetscInt            M, N, x, y, m, n, nExtrax, nExtray;
  DMStagStencil       row, col[2];
  PetscInt            ncols;
  PetscScalar         v[2];
  PetscInt            ir, ic[2];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(fdm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(fdm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  row.c = 0;
  for (i = 0; i < 2; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      row.i = i;
      row.j = j;

      /* x-interpolation */
      row.loc = DMSTAG_LEFT;
      if (i == 0) {
        switch (bcs[0].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          v[0]     = (arrcx[i][ielemc] - arrcx[i][iprevc]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          v[1]     = (arrcx[i][iprevc] - arrcx[i - 1][ielemc]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          col[0].i = i - 1;
          col[0].j = j;
          col[1].i = i;
          col[1].j = j;
          ncols    = 2;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M) {
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Periodic boundary condition on right boundary but mesh is not periodic in x-direction");
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        v[0]     = (arrcx[i][ielemc] - arrcx[i][iprevc]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        v[1]     = (arrcx[i][iprevc] - arrcx[i - 1][ielemc]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        col[0].i = i - 1;
        col[0].j = j;
        col[1].i = i;
        col[1].j = j;
        ncols    = 2;
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, 2, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, 2, ncols, col, ic));
        PetscCall(MatSetValuesLocal(interp[0], 1, &ir, ncols, ic, v, INSERT_VALUES));
      }

      /* y-interpolation */
      row.loc = DMSTAG_DOWN;
      if (j == 0) {
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          v[0]     = (arrcy[j][ielemc] - arrcy[j][iprevc]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          v[1]     = (arrcy[j][iprevc] - arrcy[j - 1][ielemc]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          col[0].i = i;
          col[0].j = j - 1;
          col[1].i = i;
          col[1].j = j;
          ncols    = 2;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N) {
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Periodic boundary condition on up boundary but mesh is not periodic in y-direction");
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        v[0]     = (arrcy[j][ielemc] - arrcy[j][iprevc]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
        v[1]     = (arrcy[j][iprevc] - arrcy[j - 1][ielemc]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
        col[0].i = i;
        col[0].j = j - 1;
        col[1].i = i;
        col[1].j = j;
        ncols    = 2;
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, 2, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, 2, ncols, col, ic));
        PetscCall(MatSetValuesLocal(interp[1], 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(interp[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(interp[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(interp[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(interp[1], MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(DM dm, DM fdm, PetscInt interp_dim, PetscInt dim, const NSBoundaryCondition *bcs, PetscReal t, InsertMode mode, Vec vbc)
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row;
  PetscScalar         v;
  PetscReal           xb[2];
  PetscScalar         vb[2];
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

  row.c = 0;

  switch (interp_dim) {
  case 0:
    /* Left boundary */
    switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j) {
        row.loc = DMSTAG_LEFT;
        row.i   = 0;
        row.j   = j;
        xb[0]   = arrcx[0][iprevc];
        xb[1]   = arrcy[j][ielemc];
        PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));
        v = vb[dim];
        PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &v, mode));
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
        row.loc = DMSTAG_RIGHT;
        row.i   = M - 1;
        row.j   = j;
        xb[0]   = arrcx[M - 1][inextc];
        xb[1]   = arrcy[j][ielemc];
        PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));
        v = vb[dim];
        PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &v, mode));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
    }
    break;
  case 1:
    /* Down boundary */
    switch (bcs[2].type) {
    case NS_BC_VELOCITY:
      for (i = x; i < x + m; ++i) {
        row.loc = DMSTAG_DOWN;
        row.i   = i;
        row.j   = 0;
        xb[0]   = arrcx[i][ielemc];
        xb[1]   = arrcy[0][iprevc];
        PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));
        v = vb[dim];
        PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &v, mode));
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
        row.loc = DMSTAG_UP;
        row.i   = i;
        row.j   = N - 1;
        xb[0]   = arrcx[i][ielemc];
        xb[1]   = arrcy[N - 1][inextc];
        PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));
        v = vb[dim];
        PetscCall(DMStagVecSetValuesStencil(fdm, vbc, 1, &row, &v, mode));
      }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for velocity interpolation operator");
  }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMComputeFaceDivergenceOperator2d_Cart_Internal(DM dm, DM fdm, Mat D)
{
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[4];
  PetscInt            ncols;
  PetscScalar         v[4];
  PetscInt            ir, ic[4];
  const PetscScalar **arrcx, **arrcy;
  PetscInt            iprevc, inextc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (i = 0; i < 4; ++i) col[i].c = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;

      v[0]       = -1. / (arrcx[i][inextc] - arrcx[i][iprevc]);
      col[0].loc = DMSTAG_LEFT;
      col[0].i   = i;
      col[0].j   = j;
      v[1]       = 1. / (arrcx[i][inextc] - arrcx[i][iprevc]);
      col[1].loc = DMSTAG_RIGHT;
      col[1].i   = i;
      col[1].j   = j;
      v[2]       = -1. / (arrcy[j][inextc] - arrcy[j][iprevc]);
      col[2].loc = DMSTAG_DOWN;
      col[2].i   = i;
      col[2].j   = j;
      v[3]       = 1. / (arrcy[j][inextc] - arrcy[j][iprevc]);
      col[3].loc = DMSTAG_UP;
      col[3].i   = i;
      col[3].j   = j;
      ncols      = 4;

      PetscCall(DMStagStencilToIndexLocal(dm, 2, 1, &row, &ir));
      PetscCall(DMStagStencilToIndexLocal(fdm, 2, ncols, col, ic));
      PetscCall(MatSetValuesLocal(D, 1, &ir, ncols, ic, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputeSpatialOperators2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(NSFSMComputePressureGradientOperators2d_Cart_Internal(dm, ns->bcs, fsm->grad_p));
  PetscCall(NSFSMComputePressureCorrectionGradientOperators2d_Cart_Internal(dm, ns->bcs, fsm->grad_p_prime));
  PetscCall(NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(dm, ns->bcs, 1., 0.5 * ns->mu * ns->dt / ns->rho, fsm->helm_v));
  PetscCall(NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(dm, ns->bcs, fsm->lap_p_prime));
  PetscCall(NSFSMComputeVelocityInterpolationOperators2d_Cart_Internal(dm, fdm, ns->bcs, fsm->interp_v));
  PetscCall(NSFSMComputeFaceGradientOperator2d_Cart_Internal(dm, fdm, ns->bcs, fsm->grad_f));
  PetscCall(NSFSMComputeFaceDivergenceOperator2d_Cart_Internal(dm, fdm, fsm->div_f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorsIntermediateVelocity2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  (void)J;

  KSPVCtx *kspvctx = (KSPVCtx *)ctx;
  NS       ns      = kspvctx->ns;
  DM       dm;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(dm, ns->bcs, 1., -0.5 * ns->mu * ns->dt / ns->rho, Jpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSIntermediateVelocity2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  KSPVCtx *kspvctx = (KSPVCtx *)ctx;
  NS       ns      = kspvctx->ns;
  NS_FSM  *fsm     = (NS_FSM *)ns->data;
  DM       dm;
  Vec      grad_p;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &grad_p));

  PetscCall(MatMult(fsm->grad_p[kspvctx->dim], fsm->p_half, grad_p));

  PetscCall(MatMult(fsm->helm_v, fsm->v[kspvctx->dim], b));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, kspvctx->dim, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, b));

  PetscCall(VecAXPBYPCZ(b, -1.5 * ns->dt, 0.5 * ns->dt, 1., fsm->N[kspvctx->dim], fsm->N_prev[kspvctx->dim]));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, grad_p));

  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, kspvctx->dim, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, b));

  PetscCall(DMRestoreGlobalVector(dm, &grad_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperatorPressureCorrection2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS           ns = (NS)ctx;
  MPI_Comm     comm;
  DM           dm;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(dm, ns->bcs, Jpre));

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
  DM      dm, fdm;

  MPI_Comm     comm;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(MatMult(fsm->div_f, fsm->fv_star, b));
  PetscCall(VecScale(b, ns->rho / ns->dt));

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
  DM      dm, fdm;
  Vec     u_interp, v_interp, UV_u_interp, UV_v_interp;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMGetGlobalVector(fdm, &u_interp));
  PetscCall(DMGetGlobalVector(fdm, &v_interp));
  PetscCall(DMGetGlobalVector(fdm, &UV_u_interp));
  PetscCall(DMGetGlobalVector(fdm, &UV_v_interp));

  PetscCall(MatMult(fsm->interp_v[0], fsm->v[0], u_interp));
  PetscCall(MatMultAdd(fsm->interp_v[1], fsm->v[0], u_interp, u_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 0, ns->bcs, ns->t + ns->dt, ADD_VALUES, u_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 0, ns->bcs, ns->t + ns->dt, ADD_VALUES, u_interp));
  PetscCall(MatMult(fsm->interp_v[0], fsm->v[1], v_interp));
  PetscCall(MatMultAdd(fsm->interp_v[1], fsm->v[1], v_interp, v_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 1, ns->bcs, ns->t + ns->dt, ADD_VALUES, v_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 1, ns->bcs, ns->t + ns->dt, ADD_VALUES, v_interp));

  PetscCall(VecPointwiseMult(UV_u_interp, fsm->fv, u_interp));
  PetscCall(VecPointwiseMult(UV_v_interp, fsm->fv, v_interp));
  PetscCall(MatMult(fsm->div_f, UV_u_interp, fsm->N[0]));
  PetscCall(MatMult(fsm->div_f, UV_v_interp, fsm->N[1]));

  PetscCall(DMRestoreGlobalVector(fdm, &u_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &v_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_u_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_v_interp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  DM       dm, fdm;
  Vec      grad_p[2], grad_p_f, u_tilde, v_tilde, UV_tilde, s;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    PetscCall(KSPSolve(fsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(fsm->kspv[d], &s));
    PetscCall(VecCopy(s, fsm->v_star[d]));
  }

  /* Calculate face-centered intermediate velocity. */
  PetscCall(DMGetGlobalVector(dm, &grad_p[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p[1]));
  PetscCall(DMGetGlobalVector(fdm, &grad_p_f));
  PetscCall(DMGetGlobalVector(dm, &u_tilde));
  PetscCall(DMGetGlobalVector(dm, &v_tilde));
  PetscCall(DMGetGlobalVector(fdm, &UV_tilde));

  PetscCall(MatMult(fsm->grad_p[0], fsm->p_half, grad_p[0]));
  PetscCall(MatMult(fsm->grad_p[1], fsm->p_half, grad_p[1]));
  PetscCall(MatMult(fsm->grad_f, fsm->p_half, grad_p_f));

  PetscCall(VecWAXPY(u_tilde, ns->dt / ns->rho, grad_p[0], fsm->v_star[0]));
  PetscCall(VecWAXPY(v_tilde, ns->dt / ns->rho, grad_p[1], fsm->v_star[1]));

  PetscCall(MatMult(fsm->interp_v[0], u_tilde, UV_tilde));
  PetscCall(MatMultAdd(fsm->interp_v[1], v_tilde, UV_tilde, UV_tilde));

  PetscCall(VecWAXPY(fsm->fv_star, -ns->dt / ns->rho, grad_p_f, UV_tilde));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 0, ns->bcs, ns->t + ns->dt, INSERT_VALUES, fsm->fv_star));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 1, ns->bcs, ns->t + ns->dt, INSERT_VALUES, fsm->fv_star));

  PetscCall(DMRestoreGlobalVector(dm, &grad_p[0]));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p[1]));
  PetscCall(DMRestoreGlobalVector(fdm, &grad_p_f));
  PetscCall(DMRestoreGlobalVector(dm, &u_tilde));
  PetscCall(DMRestoreGlobalVector(dm, &v_tilde));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_tilde));
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
  DM      dm, fdm;
  Vec     grad_p_prime[2], grad_p_prime_f, lap_p_prime;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[1]));
  PetscCall(DMGetGlobalVector(fdm, &grad_p_prime_f));
  PetscCall(DMGetGlobalVector(dm, &lap_p_prime));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecCopy(fsm->N[1], fsm->N_prev[1]));

  PetscCall(MatMult(fsm->grad_p_prime[0], fsm->p_prime, grad_p_prime[0]));
  PetscCall(MatMult(fsm->grad_p_prime[1], fsm->p_prime, grad_p_prime[1]));
  PetscCall(MatMult(fsm->grad_f, fsm->p_prime, grad_p_prime_f));
  PetscCall(MatMult(fsm->lap_p_prime, fsm->p_prime, lap_p_prime));

  PetscCall(VecWAXPY(fsm->v[0], -ns->dt / ns->rho, grad_p_prime[0], fsm->v_star[0]));
  PetscCall(VecWAXPY(fsm->v[1], -ns->dt / ns->rho, grad_p_prime[1], fsm->v_star[1]));
  PetscCall(VecWAXPY(fsm->fv, -ns->dt / ns->rho, grad_p_prime_f, fsm->fv_star));

  PetscCall(VecAXPBYPCZ(fsm->p_half, 1., -0.5 * ns->mu * ns->dt / ns->rho, 1., fsm->p_prime, lap_p_prime));
  PetscCall(VecAXPBYPCZ(fsm->p, 1.5, -0.5, 0., fsm->p_half, fsm->p_half_prev));

  PetscCall(DMRestoreGlobalVector(dm, &grad_p_prime[0]));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p_prime[1]));
  PetscCall(DMRestoreGlobalVector(fdm, &grad_p_prime_f));
  PetscCall(DMRestoreGlobalVector(dm, &lap_p_prime));

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
