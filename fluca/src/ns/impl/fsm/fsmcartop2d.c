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

PetscErrorCode NSFSMComputePressureGradientOperator2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, Mat G[])
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

      PetscCall(DMStagMatSetValuesStencil(dm, G[0], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  // Compute y-gradient operator
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

FLUCA_INTERN PetscErrorCode NSFSMComputePressureCorrectionGradientOperator2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, Mat G[])
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

      PetscCall(DMStagMatSetValuesStencil(dm, G[0], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  // Compute y-gradient operator
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

      PetscCall(DMStagMatSetValuesStencil(dm, G[1], 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(G[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(G[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G[1], MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

FLUCA_INTERN PetscErrorCode NSFSMComputePressureCorrectionFaceGradientOperator2d_Cart_Internal(DM dm, DM fdm, const NSBoundaryCondition *bcs, Mat G)
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
          v[0]     = -1. / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          v[1]     = 1. / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
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
        v[0]     = -1. / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        v[1]     = 1. / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        col[0].i = i - 1;
        col[0].j = j;
        col[1].i = i;
        col[1].j = j;
        ncols    = 2;
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, 2, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, 2, 2, col, ic));
        PetscCall(MatSetValuesLocal(G, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }

      /* y-gradient */
      row.loc = DMSTAG_DOWN;
      if (j == 0) {
        switch (bcs[2].type) {
        case NS_BC_VELOCITY:
          ncols = 0;
          break;
        case NS_BC_PERIODIC:
          v[0]     = -1. / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          v[1]     = 1. / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
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
        v[0]     = -1. / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
        v[1]     = 1. / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
        col[0].i = i;
        col[0].j = j - 1;
        col[1].i = i;
        col[1].j = j;
        ncols    = 2;
      }

      if (ncols > 0) {
        PetscCall(DMStagStencilToIndexLocal(fdm, 2, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(dm, 2, 2, col, ic));
        PetscCall(MatSetValuesLocal(G, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));
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
PetscErrorCode NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, PetscScalar coeff, PetscScalar scale, Mat L)
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

      PetscCall(DMStagMatSetValuesStencil(dm, L, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMAddVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(DM dm, PetscInt dim, const NSBoundaryCondition *bcs, PetscReal t, PetscReal scale, Vec vbc)
{
  PetscFunctionBegin;
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
  for (j = y; j < y + n; ++j) {
    row.i = 0;
    row.j = j;

    switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      xb[0] = arrcx[0][iprevc];
      xb[1] = arrcy[j][ielemc];
      PetscCall(bcs[0].velocity(2, t, xb, vb, bcs[0].ctx_velocity));

      h1 = arrcx[0][ielemc] - arrcx[0][iprevc];
      h2 = arrcx[1][ielemc] - arrcx[0][ielemc];
      h3 = arrcx[2][ielemc] - arrcx[0][ielemc];

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      break;
    case NS_BC_PERIODIC:
      v = 0.;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
    }

    PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
  }

  /* Right boundary */
  for (j = y; j < y + n; ++j) {
    row.i = M - 1;
    row.j = j;

    switch (bcs[1].type) {
    case NS_BC_VELOCITY:
      xb[0] = arrcx[M - 1][inextc];
      xb[1] = arrcy[j][ielemc];
      PetscCall(bcs[1].velocity(2, t, xb, vb, bcs[1].ctx_velocity));

      h1 = arrcx[M - 1][inextc] - arrcx[M - 1][ielemc];
      h2 = arrcx[M - 1][ielemc] - arrcx[M - 2][ielemc];
      h3 = arrcx[M - 1][ielemc] - arrcx[M - 3][ielemc];

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      break;
    case NS_BC_PERIODIC:
      v = 0.;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
    }

    PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
  }

  /* Down boundary */
  for (i = x; i < x + m; ++i) {
    row.i = i;
    row.j = 0;

    switch (bcs[2].type) {
    case NS_BC_VELOCITY:
      xb[0] = arrcx[i][ielemc];
      xb[1] = arrcy[0][iprevc];
      PetscCall(bcs[2].velocity(2, t, xb, vb, bcs[2].ctx_velocity));

      h1 = arrcy[0][ielemc] - arrcy[0][iprevc];
      h2 = arrcy[1][ielemc] - arrcy[0][ielemc];
      h3 = arrcy[2][ielemc] - arrcy[0][ielemc];

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      break;
    case NS_BC_PERIODIC:
      v = 0.;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
    }

    PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
  }

  /* Up boundary */
  for (i = x; i < x + m; ++i) {
    row.i = i;
    row.j = N - 1;

    switch (bcs[3].type) {
    case NS_BC_VELOCITY:
      xb[0] = arrcx[i][ielemc];
      xb[1] = arrcy[N - 1][inextc];
      PetscCall(bcs[3].velocity(2, t, xb, vb, bcs[3].ctx_velocity));

      h1 = arrcy[N - 1][inextc] - arrcy[N - 1][ielemc];
      h2 = arrcy[N - 1][ielemc] - arrcy[N - 2][ielemc];
      h3 = arrcy[N - 1][ielemc] - arrcy[N - 3][ielemc];

      v = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * scale * vb[dim];
      break;
    case NS_BC_PERIODIC:
      v = 0.;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
    }

    PetscCall(DMStagVecSetValuesStencil(dm, vbc, 1, &row, &v, ADD_VALUES));
  }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(DM dm, const NSBoundaryCondition *bcs, Mat L)
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

      PetscCall(DMStagMatSetValuesStencil(dm, L, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
