#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

typedef enum {
  GRAD_X,
  GRAD_Y,
} GradientDirection;

static PetscErrorCode GradientForwardDifference(GradientDirection dir, PetscInt i, PetscInt j, PetscReal coord_center, PetscReal coord_next, PetscReal coord_next2, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
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
  col[1].i = i + (dir == GRAD_X ? 1 : 0);
  col[1].j = j + (dir == GRAD_Y ? 1 : 0);
  col[2].i = i + (dir == GRAD_X ? 2 : 0);
  col[2].j = j + (dir == GRAD_Y ? 2 : 0);
  *ncols   = 3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GradientCentralDifference(GradientDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev, PetscReal coord_next, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  v[0]     = -1. / (coord_next - coord_prev);
  v[1]     = 1. / (coord_next - coord_prev);
  col[0].i = i - (dir == GRAD_X ? 1 : 0);
  col[0].j = j - (dir == GRAD_Y ? 1 : 0);
  col[1].i = i + (dir == GRAD_X ? 1 : 0);
  col[1].j = j + (dir == GRAD_Y ? 1 : 0);
  *ncols   = 2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GradientBackwardDifference(GradientDirection dir, PetscInt i, PetscInt j, PetscReal coord_prev2, PetscReal coord_prev, PetscReal coord_center, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscReal h1, h2;

  PetscFunctionBegin;
  h1       = coord_center - coord_prev;
  h2       = coord_center - coord_prev2;
  v[0]     = -h1 / (h2 * (h1 - h2));
  v[1]     = h2 / (h1 * (h1 - h2));
  v[2]     = (h1 + h2) / (h1 * h2);
  col[0].i = i - (dir == GRAD_X ? 2 : 0);
  col[0].j = j - (dir == GRAD_Y ? 2 : 0);
  col[1].i = i - (dir == GRAD_X ? 1 : 0);
  col[1].j = j - (dir == GRAD_Y ? 1 : 0);
  col[2].i = i;
  col[2].j = j;
  *ncols   = 3;
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
          PetscCall(GradientForwardDifference(GRAD_X, i, j, arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(GradientCentralDifference(GRAD_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcs[1].type) {
        case NS_BC_VELOCITY:
          PetscCall(GradientBackwardDifference(GRAD_X, i, j, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(GradientCentralDifference(GRAD_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
        }
      } else {
        PetscCall(GradientCentralDifference(GRAD_X, i, j, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
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
          PetscCall(GradientForwardDifference(GRAD_Y, i, j, arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(GradientCentralDifference(GRAD_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcs[3].type) {
        case NS_BC_VELOCITY:
          PetscCall(GradientBackwardDifference(GRAD_Y, i, j, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
          break;
        case NS_BC_PERIODIC:
          PetscCall(GradientCentralDifference(GRAD_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
        }
      } else {
        PetscCall(GradientCentralDifference(GRAD_Y, i, j, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
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
