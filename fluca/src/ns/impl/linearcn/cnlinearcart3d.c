#include <fluca/private/nslinearcnimpl.h>
#include "../src/ns/utils/cartdiscret.h"

static PetscErrorCode ComputePressureGradientOperator_Private(DM sdm, DM vdm, const NSBoundaryCondition *bcs, Mat G)
{
  PetscInt            M, N, P, x, y, z, m, n, p;
  DMStagStencil       row, col[3];
  PetscInt            ncols, ir, ic[3];
  PetscScalar         v[3];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j, k;
  const PetscInt      dim = 3;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(sdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(sdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(G, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  for (i = 0; i < 3; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  /* Compute x-gradient operator */
  row.c = 0;
  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;

        if (i == 0) {
          /* Left boundary */
          switch (bcs[0].type) {
          case NS_BC_VELOCITY:
            PetscCall(NSComputeFirstDerivForwardDiffNoCond_Cart(DIR_X, i, j, k, arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], &ncols, col, v));
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
          }
        } else if (i == M - 1) {
          /* Right boundary */
          switch (bcs[1].type) {
          case NS_BC_VELOCITY:
            PetscCall(NSComputeFirstDerivBackwardDiffNoCond_Cart(DIR_X, i, j, k, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
          }
        } else {
          PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i + 1][ielemc], &ncols, col, v));
        }

        PetscCall(DMStagStencilToIndexLocal(vdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(G, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }

  /* Compute y-gradient operator */
  row.c = 1;
  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;

        if (j == 0) {
          /* Down boundary */
          switch (bcs[2].type) {
          case NS_BC_VELOCITY:
            PetscCall(NSComputeFirstDerivForwardDiffNoCond_Cart(DIR_Y, i, j, k, arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], &ncols, col, v));
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for down boundary");
          }
        } else if (j == N - 1) {
          /* Up boundary */
          switch (bcs[3].type) {
          case NS_BC_VELOCITY:
            PetscCall(NSComputeFirstDerivBackwardDiffNoCond_Cart(DIR_Y, i, j, k, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
          }
        } else {
          PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j + 1][ielemc], &ncols, col, v));
        }

        PetscCall(DMStagStencilToIndexLocal(vdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(G, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }

  /* Compute z-gradient operator */
  row.c = 2;
  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;

        if (k == 0) {
          /* Back boundary */
          switch (bcs[4].type) {
          case NS_BC_VELOCITY:
            PetscCall(NSComputeFirstDerivForwardDiffNoCond_Cart(DIR_Z, i, j, k, arrcz[k][ielemc], arrcz[k + 1][ielemc], arrcz[k + 2][ielemc], &ncols, col, v));
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k + 1][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
          }
        } else if (k == P - 1) {
          /* Front boundary */
          switch (bcs[5].type) {
          case NS_BC_VELOCITY:
            PetscCall(NSComputeFirstDerivBackwardDiffNoCond_Cart(DIR_Z, i, j, k, arrcz[k - 2][ielemc], arrcz[k - 1][ielemc], arrcz[k][ielemc], &ncols, col, v));
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k + 1][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
          }
        } else {
          PetscCall(NSComputeFirstDerivCentralDiff_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k + 1][ielemc], &ncols, col, v));
        }

        PetscCall(DMStagStencilToIndexLocal(vdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(G, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityLaplacianOperator_Private(DM dm, const NSBoundaryCondition *bcs, Mat L)
{
  PetscInt            M, N, P, x, y, z, m, n, p, dim;
  DMStagStencil       row, col[7];
  PetscInt            ncols;
  PetscScalar         v[7];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j, k, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, &P));
  PetscCall(DMStagGetCorners(dm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(MatSetOption(L, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  for (c = 0; c < dim; ++c) {
    row.loc = DMSTAG_ELEMENT;
    row.c   = c;
    for (i = 0; i < 7; ++i) {
      col[i].loc = DMSTAG_ELEMENT;
      col[i].c   = c;
    }

    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          row.i    = i;
          row.j    = j;
          row.k    = k;
          col[0].i = i;
          col[0].j = j;
          col[0].k = k;
          v[0]     = 0.;
          ncols    = 1;

          if (i == 0) {
            /* Left boundary */
            switch (bcs[0].type) {
            case NS_BC_VELOCITY:
              PetscCall(NSComputeSecondDerivForwardDiffDirichletCond_Cart(DIR_X, i, j, k, arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i + 1][ielemc], arrcx[i + 2][ielemc], &ncols, col, v));
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary");
            }
          } else if (i == M - 1) {
            /* Right boundary */
            switch (bcs[1].type) {
            case NS_BC_VELOCITY:
              PetscCall(NSComputeSecondDerivBackwardDiffDirichletCond_Cart(DIR_X, i, j, k, arrcx[i - 2][ielemc], arrcx[i - 1][ielemc], arrcx[i][ielemc], arrcx[i][inextc], &ncols, col, v));
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary");
            }
          } else {
            PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], arrcx[i][inextc], arrcx[i + 1][ielemc], &ncols, col, v));
          }

          if (j == 0) {
            /* Down boundary */
            switch (bcs[2].type) {
            case NS_BC_VELOCITY:
              PetscCall(NSComputeSecondDerivForwardDiffDirichletCond_Cart(DIR_Y, i, j, k, arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j + 1][ielemc], arrcy[j + 2][ielemc], &ncols, col, v));
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary");
            }
          } else if (j == N - 1) {
            /* Up boundary */
            switch (bcs[3].type) {
            case NS_BC_VELOCITY:
              PetscCall(NSComputeSecondDerivBackwardDiffDirichletCond_Cart(DIR_Y, i, j, k, arrcy[j - 2][ielemc], arrcy[j - 1][ielemc], arrcy[j][ielemc], arrcy[j][inextc], &ncols, col, v));
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
            }
          } else {
            PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], arrcy[j][inextc], arrcy[j + 1][ielemc], &ncols, col, v));
          }

          /* Z-direction second derivative */
          if (k == 0) {
            /* Back boundary */
            switch (bcs[4].type) {
            case NS_BC_VELOCITY:
              PetscCall(NSComputeSecondDerivForwardDiffDirichletCond_Cart(DIR_Z, i, j, k, arrcz[k][iprevc], arrcz[k][ielemc], arrcz[k + 1][ielemc], arrcz[k + 2][ielemc], &ncols, col, v));
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], arrcz[k][inextc], arrcz[k + 1][ielemc], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for back boundary");
            }
          } else if (k == P - 1) {
            /* Front boundary */
            switch (bcs[5].type) {
            case NS_BC_VELOCITY:
              PetscCall(NSComputeSecondDerivBackwardDiffDirichletCond_Cart(DIR_Z, i, j, k, arrcz[k - 2][ielemc], arrcz[k - 1][ielemc], arrcz[k][ielemc], arrcz[k][inextc], &ncols, col, v));
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], arrcz[k][inextc], arrcz[k + 1][ielemc], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for front boundary");
            }
          } else {
            PetscCall(NSComputeSecondDerivCentralDiff_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], arrcz[k][inextc], arrcz[k + 1][ielemc], &ncols, col, v));
          }

          PetscCall(DMStagMatSetValuesStencil(dm, L, 1, &row, ncols, col, v, INSERT_VALUES));
        }
  }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeVelocityLaplacianBoundaryConditionVector_Private(DM vdm, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, P, x, y, z, m, n, p, dim;
  PetscBool           isFirstRankx, isFirstRanky, isFirstRankz, isLastRankx, isLastRanky, isLastRankz;
  DMStagStencil       row[3];
  PetscScalar         v[3];
  PetscReal           xb[3];
  PetscScalar         vb[3];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, inextc, ielemc;
  PetscReal           h1, h2, h3;
  PetscInt            i, j, k, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(vdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(vdm, &isFirstRankx, &isFirstRanky, &isFirstRankz));
  PetscCall(DMStagGetIsLastRank(vdm, &isLastRankx, &isLastRanky, &isLastRankz));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
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
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          for (c = 0; c < dim; ++c) {
            row[c].i = 0;
            row[c].j = j;
            row[c].k = k;
          }
          xb[0] = arrcx[0][iprevc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[0].velocity(dim, t, xb, vb, bcs[0].ctx_velocity));

          h1 = arrcx[0][ielemc] - arrcx[0][iprevc];
          h2 = arrcx[1][ielemc] - arrcx[0][ielemc];
          h3 = arrcx[2][ielemc] - arrcx[0][ielemc];

          for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
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
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          for (c = 0; c < dim; ++c) {
            row[c].i = M - 1;
            row[c].j = j;
            row[c].k = k;
          }
          xb[0] = arrcx[M - 1][inextc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[1].velocity(dim, t, xb, vb, bcs[1].ctx_velocity));

          h1 = arrcx[M - 1][inextc] - arrcx[M - 1][ielemc];
          h2 = arrcx[M - 1][ielemc] - arrcx[M - 2][ielemc];
          h3 = arrcx[M - 1][ielemc] - arrcx[M - 3][ielemc];

          for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = 0;
            row[c].k = k;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[0][iprevc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[2].velocity(dim, t, xb, vb, bcs[2].ctx_velocity));

          h1 = arrcy[0][ielemc] - arrcy[0][iprevc];
          h2 = arrcy[1][ielemc] - arrcy[0][ielemc];
          h3 = arrcy[2][ielemc] - arrcy[0][ielemc];

          for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = N - 1;
            row[c].k = k;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[N - 1][inextc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[3].velocity(dim, t, xb, vb, bcs[3].ctx_velocity));

          h1 = arrcy[N - 1][inextc] - arrcy[N - 1][ielemc];
          h2 = arrcy[N - 1][ielemc] - arrcy[N - 2][ielemc];
          h3 = arrcy[N - 1][ielemc] - arrcy[N - 3][ielemc];

          for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary");
    }

  /* Back boundary */
  if (isFirstRankz) switch (bcs[4].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = j;
            row[c].k = 0;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[0][iprevc];
          PetscCall(bcs[4].velocity(dim, t, xb, vb, bcs[4].ctx_velocity));

          h1 = arrcz[0][ielemc] - arrcz[0][iprevc];
          h2 = arrcz[1][ielemc] - arrcz[0][ielemc];
          h3 = arrcz[2][ielemc] - arrcz[0][ielemc];

          for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for back boundary");
    }

  /* Front boundary */
  if (isLastRankz) switch (bcs[5].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = j;
            row[c].k = P - 1;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[P - 1][inextc];
          PetscCall(bcs[5].velocity(dim, t, xb, vb, bcs[5].ctx_velocity));

          h1 = arrcz[P - 1][inextc] - arrcz[P - 1][ielemc];
          h2 = arrcz[P - 1][ielemc] - arrcz[P - 2][ielemc];
          h3 = arrcz[P - 1][ielemc] - arrcz[P - 3][ielemc];

          for (c = 0; c < dim; ++c) v[c] = 2. * (h2 + h3) / (h1 * (h1 + h2) * (h1 + h3)) * vb[c];
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for front boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvectionOperator_Private(DM vdm, DM Sdm, DM Vdm, Vec V0, Vec v0interp, const NSBoundaryCondition *bcs, Mat C)
{
  PetscInt              M, N, P, x, y, z, m, n, p;
  DMStagStencil         row, col[2];
  PetscInt              ncols;
  PetscScalar           v[2];
  Vec                   V0local, v0interplocal;
  const PetscScalar ****arrV0, ****arrv0interp;
  const PetscScalar   **arrcx, **arrcy, **arrcz;
  PetscReal             hx, hy, hz;
  PetscInt              iV0[3], iv0interp[3][3], iprevc, ielemc;
  PetscInt              i, j, k, l, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(vdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));

  PetscCall(DMGetLocalVector(Sdm, &V0local));
  PetscCall(DMGetLocalVector(Vdm, &v0interplocal));

  PetscCall(DMGlobalToLocal(Sdm, V0, INSERT_VALUES, V0local));
  PetscCall(DMGlobalToLocal(Vdm, v0interp, INSERT_VALUES, v0interplocal));
  PetscCall(DMStagVecGetArrayRead(Sdm, V0local, &arrV0));
  PetscCall(DMStagGetLocationSlot(Sdm, DMSTAG_LEFT, 0, &iV0[0]));
  PetscCall(DMStagGetLocationSlot(Sdm, DMSTAG_DOWN, 0, &iV0[1]));
  PetscCall(DMStagGetLocationSlot(Sdm, DMSTAG_BACK, 0, &iV0[2]));
  PetscCall(DMStagVecGetArrayRead(Vdm, v0interplocal, &arrv0interp));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_LEFT, 0, &iv0interp[0][0]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_DOWN, 0, &iv0interp[0][1]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_BACK, 0, &iv0interp[0][2]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_LEFT, 1, &iv0interp[1][0]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_DOWN, 1, &iv0interp[1][1]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_BACK, 1, &iv0interp[1][2]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_LEFT, 2, &iv0interp[2][0]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_DOWN, 2, &iv0interp[2][1]));
  PetscCall(DMStagGetLocationSlot(Vdm, DMSTAG_BACK, 2, &iv0interp[2][2]));

  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  for (i = 0; i < 2; ++i) col[i].loc = DMSTAG_ELEMENT;

  /* (Cv)_i = (1/2) * d/dx_j (v_i * V0_j + v0interp_i * v_j) */
  for (c = 0; c < 3; ++c) {
    row.c = c;

    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = j;
          row.k = k;
          hx    = arrcx[i + 1][iprevc] - arrcx[i][iprevc];
          hy    = arrcy[j + 1][iprevc] - arrcy[j][iprevc];
          hz    = arrcz[k + 1][iprevc] - arrcz[k][iprevc];

          /* Left cell face */
          for (l = 0; l < 2; ++l) col[l].c = c;
          ncols = 0;
          if (i == 0) {
            /* Left boundary */
            switch (bcs[0].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrV0[k][j][i][iV0[0]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrV0[k][j][i][iV0[0]], &ncols, col, v));
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
              PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrv0interp[k][j][i][iv0interp[c][0]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for left boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], hx, arrv0interp[k][j][i][iv0interp[c][0]], &ncols, col, v));
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
              PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_X, i, j, k, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrV0[k][j][i + 1][iV0[0]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_X, i, j, k, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrV0[k][j][i + 1][iV0[0]], &ncols, col, v));
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
              PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_X, i, j, k, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrv0interp[k][j][i + 1][iv0interp[c][0]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for right boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_X, i, j, k, arrcx[i][ielemc], arrcx[i + 1][iprevc], arrcx[i + 1][ielemc], hx, arrv0interp[k][j][i + 1][iv0interp[c][0]], &ncols, col, v));
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
              PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrV0[k][j][i][iV0[1]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for bottom boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrV0[k][j][i][iV0[1]], &ncols, col, v));
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
              PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrv0interp[k][j][i][iv0interp[c][1]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for bottom boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], hy, arrv0interp[k][j][i][iv0interp[c][1]], &ncols, col, v));
          }
          if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

          /* Top cell face */
          for (l = 0; l < 2; ++l) col[l].c = c;
          ncols = 0;
          if (j == N - 1) {
            /* Top boundary */
            switch (bcs[3].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Y, i, j, k, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrV0[k][j + 1][i][iV0[1]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for top boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Y, i, j, k, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrV0[k][j + 1][i][iV0[1]], &ncols, col, v));
          }
          if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

          for (l = 0; l < 2; ++l) col[l].c = 1;
          ncols = 0;
          if (j == N - 1) {
            /* Top boundary */
            switch (bcs[3].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Y, i, j, k, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrv0interp[k][j + 1][i][iv0interp[c][1]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for top boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Y, i, j, k, arrcy[j][ielemc], arrcy[j + 1][iprevc], arrcy[j + 1][ielemc], hy, arrv0interp[k][j + 1][i][iv0interp[c][1]], &ncols, col, v));
          }
          if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

          /* Back cell face */
          for (l = 0; l < 2; ++l) col[l].c = c;
          ncols = 0;
          if (k == 0) {
            /* Back boundary */
            switch (bcs[4].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], hz, arrV0[k][j][i][iV0[2]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], hz, arrV0[k][j][i][iV0[2]], &ncols, col, v));
          }
          if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

          for (l = 0; l < 2; ++l) col[l].c = 2;
          ncols = 0;
          if (k == 0) {
            /* Back boundary */
            switch (bcs[4].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], hz, arrv0interp[k][j][i][iv0interp[c][2]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationPrev_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], hz, arrv0interp[k][j][i][iv0interp[c][2]], &ncols, col, v));
          }
          if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

          /* Front cell face */
          for (l = 0; l < 2; ++l) col[l].c = c;
          ncols = 0;
          if (k == P - 1) {
            /* Front boundary */
            switch (bcs[5].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Z, i, j, k, arrcz[k][ielemc], arrcz[k + 1][iprevc], arrcz[k + 1][ielemc], hz, arrV0[k + 1][j][i][iV0[2]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Z, i, j, k, arrcz[k][ielemc], arrcz[k + 1][iprevc], arrcz[k + 1][ielemc], hz, arrV0[k + 1][j][i][iV0[2]], &ncols, col, v));
          }
          if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));

          for (l = 0; l < 2; ++l) col[l].c = 2;
          ncols = 0;
          if (k == P - 1) {
            /* Front boundary */
            switch (bcs[5].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Z, i, j, k, arrcz[k][ielemc], arrcz[k + 1][iprevc], arrcz[k + 1][ielemc], hz, arrv0interp[k + 1][j][i][iv0interp[c][2]], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
            }
          } else {
            PetscCall(NSComputeConvectionLinearInterpolationNext_Cart(DIR_Z, i, j, k, arrcz[k][ielemc], arrcz[k + 1][iprevc], arrcz[k + 1][ielemc], hz, arrv0interp[k + 1][j][i][iv0interp[c][2]], &ncols, col, v));
          }
          if (ncols > 0) PetscCall(DMStagMatSetValuesStencil(vdm, C, 1, &row, ncols, col, v, ADD_VALUES));
        }
  }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagVecRestoreArrayRead(Sdm, V0local, &arrV0));
  PetscCall(DMStagVecRestoreArrayRead(Vdm, v0interplocal, &arrv0interp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));

  PetscCall(DMRestoreLocalVector(Sdm, &V0local));
  PetscCall(DMRestoreLocalVector(Vdm, &v0interplocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeConvectionBoundaryConditionVector_Private(DM vdm, const NSBoundaryCondition *bcs, PetscReal t0, PetscReal t, Vec vbc)
{
  PetscInt            M, N, P, x, y, z, m, n, p;
  PetscBool           isFirstRankx, isFirstRanky, isFirstRankz, isLastRankx, isLastRanky, isLastRankz;
  DMStagStencil       row[3];
  PetscScalar         v[3];
  PetscReal           xb[3];
  PetscScalar         vb0[3], vb[3];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscReal           hx, hy, hz;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j, k, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(vdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(vdm, &isFirstRankx, &isFirstRanky, &isFirstRankz));
  PetscCall(DMStagGetIsLastRank(vdm, &isLastRankx, &isLastRanky, &isLastRankz));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  for (c = 0; c < 3; ++c) {
    row[c].loc = DMSTAG_ELEMENT;
    row[c].c   = c;
  }

  /* Left boundary */
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          for (c = 0; c < 3; ++c) {
            row[c].i = 0;
            row[c].j = j;
            row[c].k = k;
          }
          xb[0] = arrcx[0][iprevc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[0].velocity(3, t0, xb, vb0, bcs[0].ctx_velocity));
          PetscCall(bcs[0].velocity(3, t, xb, vb, bcs[0].ctx_velocity));

          hx = arrcx[0][inextc] - arrcx[0][iprevc];

          for (c = 0; c < 3; ++c) v[c] = -0.5 * (vb[c] * vb0[0] + vb0[c] * vb[0]) / hx;
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
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
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          for (c = 0; c < 3; ++c) {
            row[c].i = M - 1;
            row[c].j = j;
            row[c].k = k;
          }
          xb[0] = arrcx[M - 1][inextc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[1].velocity(3, t0, xb, vb0, bcs[1].ctx_velocity));
          PetscCall(bcs[1].velocity(3, t, xb, vb, bcs[1].ctx_velocity));

          hx = arrcx[M - 1][inextc] - arrcx[M - 1][iprevc];

          for (c = 0; c < 3; ++c) v[c] = 0.5 * (vb[c] * vb0[0] + vb0[c] * vb[0]) / hx;
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < 3; ++c) {
            row[c].i = i;
            row[c].j = 0;
            row[c].k = k;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[0][iprevc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[2].velocity(3, t0, xb, vb0, bcs[2].ctx_velocity));
          PetscCall(bcs[2].velocity(3, t, xb, vb, bcs[2].ctx_velocity));

          hy = arrcy[0][inextc] - arrcy[0][iprevc];

          for (c = 0; c < 3; ++c) v[c] = -0.5 * (vb[c] * vb0[1] + vb0[c] * vb[1]) / hy;
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < 3; ++c) {
            row[c].i = i;
            row[c].j = N - 1;
            row[c].k = k;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[N - 1][inextc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[3].velocity(3, t0, xb, vb0, bcs[3].ctx_velocity));
          PetscCall(bcs[3].velocity(3, t, xb, vb, bcs[3].ctx_velocity));

          hy = arrcy[N - 1][inextc] - arrcy[N - 1][iprevc];

          for (c = 0; c < 3; ++c) v[c] = 0.5 * (vb[c] * vb0[1] + vb0[c] * vb[1]) / hy;
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for bottom boundary");
    }

  /* Back boundary */
  if (isFirstRankz) switch (bcs[4].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < 3; ++c) {
            row[c].i = i;
            row[c].j = j;
            row[c].k = 0;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[0][iprevc];
          PetscCall(bcs[4].velocity(3, t0, xb, vb0, bcs[4].ctx_velocity));
          PetscCall(bcs[4].velocity(3, t, xb, vb, bcs[4].ctx_velocity));

          hz = arrcz[0][inextc] - arrcz[0][iprevc];

          for (c = 0; c < 3; ++c) v[c] = -0.5 * (vb[c] * vb0[2] + vb0[c] * vb[2]) / hz;
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
    }

  /* Front boundary */
  if (isLastRankz) switch (bcs[5].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < 3; ++c) {
            row[c].i = i;
            row[c].j = j;
            row[c].k = P - 1;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[P - 1][inextc];
          PetscCall(bcs[5].velocity(3, t0, xb, vb0, bcs[5].ctx_velocity));
          PetscCall(bcs[5].velocity(3, t, xb, vb, bcs[5].ctx_velocity));

          hz = arrcz[P - 1][inextc] - arrcz[P - 1][iprevc];

          for (c = 0; c < 3; ++c) v[c] = 0.5 * (vb[c] * vb0[2] + vb0[c] * vb[2]) / hz;
          PetscCall(DMStagVecSetValuesStencil(vdm, vbc, 3, row, v, ADD_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceVelocityInterpolationOperator_Private(DM vdm, DM Vdm, const NSBoundaryCondition *bcs, Mat B)
{
  PetscInt            M, N, P, x, y, z, m, n, p, nExtrax, nExtray, nExtraz;
  DMStagStencil       row, col[2];
  PetscInt            ncols, ir, ic[2];
  PetscScalar         v[2];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j, k, c;
  const PetscInt      dim = 3;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(vdm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  for (i = 0; i < 2; ++i) col[i].loc = DMSTAG_ELEMENT;

  for (c = 0; c < dim; ++c) {
    row.c = c;
    for (i = 0; i < 2; ++i) col[i].c = c;

    /* Interpolation in x-direction */
    row.loc = DMSTAG_LEFT;
    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m + nExtrax; ++i) {
          row.i = i;
          row.j = j;
          row.k = k;
          ncols = 0;

          if (i == 0) {
            /* Left boundary */
            switch (bcs[0].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeLinearInterpolation_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], &ncols, col, v));
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
            PetscCall(NSComputeLinearInterpolation_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], &ncols, col, v));
          }

          if (ncols > 0) {
            PetscCall(DMStagStencilToIndexLocal(Vdm, dim, 1, &row, &ir));
            PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
            PetscCall(MatSetValuesLocal(B, 1, &ir, ncols, ic, v, INSERT_VALUES));
          }
        }

    /* Interpolation in y-direction */
    row.loc = DMSTAG_DOWN;
    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n + nExtray; ++j)
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = j;
          row.k = k;
          ncols = 0;

          if (j == 0) {
            /* Down boundary */
            switch (bcs[2].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeLinearInterpolation_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], &ncols, col, v));
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
            PetscCall(NSComputeLinearInterpolation_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], &ncols, col, v));
          }

          if (ncols > 0) {
            PetscCall(DMStagStencilToIndexLocal(Vdm, dim, 1, &row, &ir));
            PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
            PetscCall(MatSetValuesLocal(B, 1, &ir, ncols, ic, v, INSERT_VALUES));
          }
        }

    /* Interpolation in z-direction */
    row.loc = DMSTAG_BACK;
    for (k = z; k < z + p + nExtraz; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = j;
          row.k = k;
          ncols = 0;

          if (k == 0) {
            /* Back boundary */
            switch (bcs[4].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC:
              PetscCall(NSComputeLinearInterpolation_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], &ncols, col, v));
              break;
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
            }
          } else if (k == P) {
            /* Front boundary */
            switch (bcs[5].type) {
            case NS_BC_VELOCITY:
              break;
            case NS_BC_PERIODIC: /* Cannot happen */
            default:
              SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
            }
          } else {
            PetscCall(NSComputeLinearInterpolation_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], &ncols, col, v));
          }

          if (ncols > 0) {
            PetscCall(DMStagStencilToIndexLocal(Vdm, dim, 1, &row, &ir));
            PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
            PetscCall(MatSetValuesLocal(B, 1, &ir, ncols, ic, v, INSERT_VALUES));
          }
        }
  }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceVelocityInterpolationBoundaryConditionVector_Private(DM Vdm, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, P, x, y, z, m, n, p, dim;
  PetscBool           isFirstRankx, isFirstRanky, isFirstRankz, isLastRankx, isLastRanky, isLastRankz;
  DMStagStencil       row[3];
  PetscReal           xb[3];
  PetscScalar         vb[3];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j, k, c;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(Vdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(Vdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(Vdm, &isFirstRankx, &isFirstRanky, &isFirstRankz));
  PetscCall(DMStagGetIsLastRank(Vdm, &isLastRankx, &isLastRanky, &isLastRankz));
  PetscCall(DMStagGetProductCoordinateArraysRead(Vdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Vdm, DMSTAG_ELEMENT, &ielemc));
  PetscCall(DMGetDimension(Vdm, &dim));

  PetscCall(VecSet(vbc, 0.));

  for (c = 0; c < dim; ++c) row[c].c = c;

  /* Left boundary */
  for (c = 0; c < dim; ++c) row[c].loc = DMSTAG_LEFT;
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          for (c = 0; c < dim; ++c) {
            row[c].i = 0;
            row[c].j = j;
            row[c].k = k;
          }
          xb[0] = arrcx[0][iprevc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[0].velocity(3, t, xb, vb, bcs[0].ctx_velocity));
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
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          for (c = 0; c < dim; ++c) {
            row[c].i = M;
            row[c].j = j;
            row[c].k = k;
          }
          xb[0] = arrcx[M][iprevc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[1].velocity(3, t, xb, vb, bcs[1].ctx_velocity));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = 0;
            row[c].k = k;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[0][iprevc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[2].velocity(3, t, xb, vb, bcs[2].ctx_velocity));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = N;
            row[c].k = k;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[N][iprevc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[3].velocity(3, t, xb, vb, bcs[3].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, dim, row, vb, INSERT_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
    }

  /* Back boundary */
  for (c = 0; c < dim; ++c) row[c].loc = DMSTAG_BACK;
  if (isFirstRankz) switch (bcs[4].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = j;
            row[c].k = 0;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[0][iprevc];
          PetscCall(bcs[4].velocity(3, t, xb, vb, bcs[4].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, dim, row, vb, INSERT_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
    }

  /* Front boundary */
  if (isLastRankz) switch (bcs[5].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          for (c = 0; c < dim; ++c) {
            row[c].i = i;
            row[c].j = j;
            row[c].k = P;
          }
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[P][iprevc];
          PetscCall(bcs[5].velocity(3, t, xb, vb, bcs[5].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(Vdm, vbc, dim, row, vb, INSERT_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(Vdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceNormalVelocityInterpolationOperator_Private(DM vdm, DM Sdm, const NSBoundaryCondition *bcs, Mat T)
{
  PetscInt            M, N, P, x, y, z, m, n, p, nExtrax, nExtray, nExtraz;
  DMStagStencil       row, col[2];
  PetscInt            ncols, ir, ic[2];
  PetscScalar         v[2];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j, k;
  const PetscInt      dim = 3;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(vdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(vdm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz));
  PetscCall(DMStagGetProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(vdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(T, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.c = 0;
  for (i = 0; i < 2; ++i) col[i].loc = DMSTAG_ELEMENT;

  /* Interpolation in x-direction */
  row.loc = DMSTAG_LEFT;
  for (i = 0; i < 2; ++i) col[i].c = 0;

  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m + nExtrax; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        ncols = 0;

        if (i == 0) {
          /* Left boundary */
          switch (bcs[0].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeLinearInterpolation_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], &ncols, col, v));
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
          PetscCall(NSComputeLinearInterpolation_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][iprevc], arrcx[i][ielemc], &ncols, col, v));
        }

        if (ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(T, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }

  /* Interpolation in y-direction */
  row.loc = DMSTAG_DOWN;
  for (i = 0; i < 2; ++i) col[i].c = 1;

  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        ncols = 0;

        if (j == 0) {
          /* Down boundary */
          switch (bcs[2].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeLinearInterpolation_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], &ncols, col, v));
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
          PetscCall(NSComputeLinearInterpolation_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][iprevc], arrcy[j][ielemc], &ncols, col, v));
        }

        if (ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(T, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }

  /* Interpolation in z-direction */
  row.loc = DMSTAG_BACK;
  for (i = 0; i < 2; ++i) col[i].c = 2;

  for (k = z; k < z + p + nExtraz; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        ncols = 0;

        if (k == 0) {
          /* Back boundary */
          switch (bcs[4].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeLinearInterpolation_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
          }
        } else if (k == P) {
          /* Front boundary */
          switch (bcs[5].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC: /* Cannot happen */
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
          }
        } else {
          PetscCall(NSComputeLinearInterpolation_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][iprevc], arrcz[k][ielemc], &ncols, col, v));
        }

        if (ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(vdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(T, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(T, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(T, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(vdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceNormalVelocityInterpolationBoundaryConditionVector_Private(DM Sdm, const NSBoundaryCondition *bcs, PetscReal t, Vec vbc)
{
  PetscInt            M, N, P, x, y, z, m, n, p;
  PetscBool           isFirstRankx, isFirstRanky, isFirstRankz, isLastRankx, isLastRanky, isLastRankz;
  DMStagStencil       row;
  PetscReal           xb[3];
  PetscScalar         vb[3];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, ielemc;
  PetscInt            i, j, k;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(Sdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(Sdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(Sdm, &isFirstRankx, &isFirstRanky, &isFirstRankz));
  PetscCall(DMStagGetIsLastRank(Sdm, &isLastRankx, &isLastRanky, &isLastRankz));
  PetscCall(DMStagGetProductCoordinateArraysRead(Sdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Sdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(Sdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(VecSet(vbc, 0.));

  row.c = 0;

  /* Left boundary */
  row.loc = DMSTAG_LEFT;
  if (isFirstRankx) switch (bcs[0].type) {
    case NS_BC_VELOCITY:
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          row.i = 0;
          row.j = j;
          row.k = k;
          xb[0] = arrcx[0][iprevc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[0].velocity(3, t, xb, vb, bcs[0].ctx_velocity));
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
      for (k = z; k < z + p; ++k)
        for (j = y; j < y + n; ++j) {
          row.i = M;
          row.j = j;
          row.k = k;
          xb[0] = arrcx[M][iprevc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[1].velocity(3, t, xb, vb, bcs[1].ctx_velocity));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = 0;
          row.k = k;
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[0][iprevc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[2].velocity(3, t, xb, vb, bcs[2].ctx_velocity));
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
      for (k = z; k < z + p; ++k)
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = N;
          row.k = k;
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[N][iprevc];
          xb[2] = arrcz[k][ielemc];
          PetscCall(bcs[3].velocity(3, t, xb, vb, bcs[3].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(Sdm, vbc, 1, &row, &vb[1], INSERT_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for up boundary");
    }

  /* Back boundary */
  row.loc = DMSTAG_BACK;
  if (isFirstRankz) switch (bcs[4].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = j;
          row.k = 0;
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[0][iprevc];
          PetscCall(bcs[4].velocity(3, t, xb, vb, bcs[4].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(Sdm, vbc, 1, &row, &vb[2], INSERT_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
    }

  /* Front boundary */
  if (isLastRankz) switch (bcs[5].type) {
    case NS_BC_VELOCITY:
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) {
          row.i = i;
          row.j = j;
          row.k = P;
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][ielemc];
          xb[2] = arrcz[P][iprevc];
          PetscCall(bcs[5].velocity(3, t, xb, vb, bcs[5].ctx_velocity));
          PetscCall(DMStagVecSetValuesStencil(Sdm, vbc, 1, &row, &vb[2], INSERT_VALUES));
        }
      break;
    case NS_BC_PERIODIC:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
    }

  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(VecAssemblyEnd(vbc));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(Sdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeStaggeredVelocityDivergenceOperator_Private(DM Sdm, DM sdm, const NSBoundaryCondition *bcs, Mat D)
{
  PetscInt            M, N, P, x, y, z, m, n, p;
  DMStagStencil       row, col[6];
  PetscInt            ncols, ir, ic[6];
  PetscScalar         v[6];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            iprevc, inextc;
  PetscScalar         dx, dy, dz;
  PetscInt            i, j, k;
  const PetscInt      dim = 3;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(sdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(sdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_RIGHT, &inextc));

  PetscCall(MatSetOption(D, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (i = 0; i < 6; ++i) col[i].c = 0;

  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        ncols = 0;

        /* x-gradient */
        dx = arrcx[i][inextc] - arrcx[i][iprevc];

        col[ncols].i   = i;
        col[ncols].j   = j;
        col[ncols].k   = k;
        col[ncols].loc = DMSTAG_LEFT;
        v[ncols]       = -1. / dx;
        ncols++;

        col[ncols].i   = i;
        col[ncols].j   = j;
        col[ncols].k   = k;
        col[ncols].loc = DMSTAG_RIGHT;
        v[ncols]       = 1. / dx;
        ncols++;

        /* y-gradient */
        dy = arrcy[j][inextc] - arrcy[j][iprevc];

        col[ncols].i   = i;
        col[ncols].j   = j;
        col[ncols].k   = k;
        col[ncols].loc = DMSTAG_DOWN;
        v[ncols]       = -1. / dy;
        ncols++;

        col[ncols].i   = i;
        col[ncols].j   = j;
        col[ncols].k   = k;
        col[ncols].loc = DMSTAG_UP;
        v[ncols]       = 1. / dy;
        ncols++;

        /* z-gradient */
        dz = arrcz[k][inextc] - arrcz[k][iprevc];

        col[ncols].i   = i;
        col[ncols].j   = j;
        col[ncols].k   = k;
        col[ncols].loc = DMSTAG_BACK;
        v[ncols]       = -1. / dz;
        ncols++;

        col[ncols].i   = i;
        col[ncols].j   = j;
        col[ncols].k   = k;
        col[ncols].loc = DMSTAG_FRONT;
        v[ncols]       = 1. / dz;
        ncols++;

        PetscCall(DMStagStencilToIndexLocal(sdm, dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(Sdm, dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(D, 1, &ir, ncols, ic, v, INSERT_VALUES));
      }

  PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeStaggeredPressureGradientOperators_Private(DM sdm, DM Sdm, const NSBoundaryCondition *bcs, Mat Gst)
{
  PetscInt            M, N, P, x, y, z, m, n, p, nExtrax, nExtray, nExtraz;
  DMStagStencil       row, col[2];
  PetscInt            ncols, ir, ic[2];
  PetscScalar         v[2];
  const PetscScalar **arrcx, **arrcy, **arrcz;
  PetscInt            ielemc;
  PetscInt            i, j, k;
  const PetscInt      dim = 3;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(sdm, &M, &N, &P));
  PetscCall(DMStagGetCorners(sdm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz));
  PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, &arrcz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(MatSetOption(Gst, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  row.c = 0;
  for (i = 0; i < 2; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  /* x-gradient */
  row.loc = DMSTAG_LEFT;
  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m + nExtrax; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        ncols = 0;

        if (i == 0) {
          /* Left boundary */
          switch (bcs[0].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFaceNormalDerivative_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
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
          PetscCall(NSComputeFaceNormalDerivative_Cart(DIR_X, i, j, k, arrcx[i - 1][ielemc], arrcx[i][ielemc], &ncols, col, v));
        }

        if (ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(Gst, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }

  /* y-gradient */
  row.loc = DMSTAG_DOWN;
  for (k = z; k < z + p; ++k)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        ncols = 0;

        if (j == 0) {
          /* Down boundary */
          switch (bcs[2].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFaceNormalDerivative_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
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
          PetscCall(NSComputeFaceNormalDerivative_Cart(DIR_Y, i, j, k, arrcy[j - 1][ielemc], arrcy[j][ielemc], &ncols, col, v));
        }

        if (ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(Gst, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }

  /* z-gradient */
  row.loc = DMSTAG_BACK;
  for (k = z; k < z + p + nExtraz; ++k)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        ncols = 0;

        if (k == 0) {
          /* Back boundary */
          switch (bcs[4].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC:
            PetscCall(NSComputeFaceNormalDerivative_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][ielemc], &ncols, col, v));
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for back boundary");
          }
        } else if (k == P) {
          /* Front boundary */
          switch (bcs[5].type) {
          case NS_BC_VELOCITY:
            break;
          case NS_BC_PERIODIC: /* Cannot happen */
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type for front boundary");
          }
        } else {
          PetscCall(NSComputeFaceNormalDerivative_Cart(DIR_Z, i, j, k, arrcz[k - 1][ielemc], arrcz[k][ielemc], &ncols, col, v));
        }

        if (ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(Sdm, dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(sdm, dim, ncols, col, ic));
          PetscCall(MatSetValuesLocal(Gst, 1, &ir, ncols, ic, v, INSERT_VALUES));
        }
      }

  PetscCall(MatAssemblyBegin(Gst, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gst, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, &arrcz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSStep_CNLinear_Cart3d_Internal(NS ns)
{
  NS_CNLinear *cnl = (NS_CNLinear *)ns->data;
  DM           vdm, Vdm;
  IS           vis, Vis, pis;
  Vec          v0, p0, v, V, dp, solv, solV, solp, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_STAG_VECTOR, &Vdm));
  PetscCall(NSGetField(ns, NS_FIELD_VELOCITY, NULL, NULL, &vis));
  PetscCall(NSGetField(ns, NS_FIELD_FACE_NORMAL_VELOCITY, NULL, NULL, &Vis));
  PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, NULL, &pis));

  if (!cnl->Bcomputed) {
    PetscCall(ComputeFaceVelocityInterpolationOperator_Private(vdm, Vdm, ns->bcs, cnl->B));
    cnl->Bcomputed = PETSC_TRUE;
  }
  PetscCall(DMGetGlobalVector(Vdm, &vbc));
  PetscCall(VecGetSubVector(ns->sol0, vis, &v0));
  PetscCall(MatMult(cnl->B, v0, cnl->v0interp));
  PetscCall(ComputeFaceVelocityInterpolationBoundaryConditionVector_Private(Vdm, ns->bcs, ns->t, vbc));
  PetscCall(VecAXPY(cnl->v0interp, 1., vbc));
  PetscCall(DMRestoreGlobalVector(Vdm, &vbc));
  PetscCall(VecRestoreSubVector(ns->sol0, vis, &v0));

  PetscCall(SNESSolve(ns->snes, NULL, ns->x));
  PetscCall(NSCheckDiverged(ns));

  PetscCall(VecGetSubVector(ns->x, vis, &v));
  PetscCall(VecGetSubVector(ns->x, Vis, &V));
  PetscCall(VecGetSubVector(ns->x, pis, &dp));
  PetscCall(VecGetSubVector(ns->sol, vis, &solv));
  PetscCall(VecGetSubVector(ns->sol, Vis, &solV));
  PetscCall(VecGetSubVector(ns->sol, pis, &solp));

  PetscCall(VecCopy(v, solv));
  PetscCall(VecCopy(V, solV));

  if (ns->step == 0) {
    PetscCall(VecGetSubVector(ns->sol0, pis, &p0));
    PetscCall(VecWAXPY(solp, 2., dp, p0));
    PetscCall(VecWAXPY(cnl->phalf, 1., dp, p0));
    PetscCall(VecRestoreSubVector(ns->sol0, pis, &p0));
  } else {
    PetscCall(VecWAXPY(solp, 1.5, dp, cnl->phalf));
    PetscCall(VecAXPY(cnl->phalf, 1., dp));
  }

  PetscCall(VecRestoreSubVector(ns->x, vis, &v));
  PetscCall(VecRestoreSubVector(ns->x, Vis, &V));
  PetscCall(VecRestoreSubVector(ns->x, pis, &dp));
  PetscCall(VecRestoreSubVector(ns->sol, vis, &solv));
  PetscCall(VecRestoreSubVector(ns->sol, Vis, &solV));
  PetscCall(VecRestoreSubVector(ns->sol, pis, &solp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFormJacobian_CNLinear_Cart3d_Internal(NS ns, Vec x, Mat J, NSFormJacobianType type)
{
  NS_CNLinear *cnl = (NS_CNLinear *)ns->data;
  DM           sdm, vdm, Sdm, Vdm;
  PetscInt     vidx, Vidx, pidx, entries;
  MeshDMType   vdmtype, Vdmtype, pdmtype;
  IS           vis, Vis, pis;
  Mat          A, G, negT, Identity, negR, D, L, Gst;
  Vec          V0;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_SCALAR, &sdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_STAG_SCALAR, &Sdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_STAG_VECTOR, &Vdm));
  PetscCall(NSGetField(ns, NS_FIELD_VELOCITY, &vidx, &vdmtype, &vis));
  PetscCall(NSGetField(ns, NS_FIELD_FACE_NORMAL_VELOCITY, &Vidx, &Vdmtype, &Vis));
  PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, &pidx, &pdmtype, &pis));

  if (type == NS_INIT_JACOBIAN) {
    PetscCall(MeshCreateMatrix(ns->mesh, MESH_DM_VECTOR, MESH_DM_VECTOR, &A));
    PetscCall(MatNestSetSubMat(J, vidx, vidx, A));

    PetscCall(MeshCreateMatrix(ns->mesh, MESH_DM_VECTOR, MESH_DM_SCALAR, &G));
    PetscCall(ComputePressureGradientOperator_Private(sdm, vdm, ns->bcs, G));
    PetscCall(MatScale(G, ns->dt / ns->rho));
    PetscCall(MatNestSetSubMat(J, vidx, pidx, G));

    PetscCall(MeshCreateMatrix(ns->mesh, MESH_DM_STAG_SCALAR, MESH_DM_VECTOR, &negT));
    PetscCall(ComputeFaceNormalVelocityInterpolationOperator_Private(vdm, Sdm, ns->bcs, negT));
    PetscCall(MatScale(negT, -1.));
    PetscCall(MatNestSetSubMat(J, Vidx, vidx, negT));

    PetscCall(DMStagGetEntries(Sdm, &entries));
    PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)J), entries, entries, PETSC_DETERMINE, PETSC_DETERMINE, 1., &Identity));
    PetscCall(MatNestSetSubMat(J, Vidx, Vidx, Identity));

    PetscCall(MeshCreateMatrix(ns->mesh, MESH_DM_VECTOR, MESH_DM_VECTOR, &L));
    PetscCall(ComputeVelocityLaplacianOperator_Private(vdm, ns->bcs, L));

    PetscCall(MeshCreateMatrix(ns->mesh, MESH_DM_STAG_SCALAR, MESH_DM_SCALAR, &Gst));
    PetscCall(ComputeStaggeredPressureGradientOperators_Private(sdm, Sdm, ns->bcs, Gst));
    PetscCall(MatScale(Gst, ns->dt / ns->rho));

    PetscCall(MatMatMult(negT, G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &negR));
    PetscCall(MatAXPY(negR, 1., Gst, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNestSetSubMat(J, Vidx, pidx, negR));

    PetscCall(MeshCreateMatrix(ns->mesh, MESH_DM_SCALAR, MESH_DM_STAG_SCALAR, &D));
    PetscCall(ComputeStaggeredVelocityDivergenceOperator_Private(Sdm, sdm, ns->bcs, D));
    PetscCall(MatNestSetSubMat(J, pidx, Vidx, D));

    PetscCall(PetscObjectCompose((PetscObject)J, "Laplacian", (PetscObject)L));
    PetscCall(PetscObjectCompose((PetscObject)J, "StaggeredGradient", (PetscObject)Gst));

    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&G));
    PetscCall(MatDestroy(&negT));
    PetscCall(MatDestroy(&Identity));
    PetscCall(MatDestroy(&negR));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&L));
    PetscCall(MatDestroy(&Gst));
  }

  if (ns->sol0) {
    PetscCall(MatNestGetSubMat(J, vidx, vidx, &A));
    /* A = I + dt * C - (nu*dt/2) * L */
    PetscCall(MatZeroEntries(A));
    PetscCall(VecGetSubVector(ns->sol0, Vis, &V0));
    PetscCall(ComputeConvectionOperator_Private(vdm, Sdm, Vdm, V0, cnl->v0interp, ns->bcs, A));
    PetscCall(VecRestoreSubVector(ns->sol0, Vis, &V0));
    PetscCall(MatScale(A, ns->dt));
    PetscCall(PetscObjectQuery((PetscObject)J, "Laplacian", (PetscObject *)&L));
    PetscCall(MatAXPY(A, -0.5 * ns->mu * ns->dt / ns->rho, L, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatShift(A, 1.));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFormFunction_CNLinear_Cart3d_Internal(NS ns, Vec x, Vec f)
{
  NS_CNLinear *cnl = (NS_CNLinear *)ns->data;
  DM           sdm, vdm, Sdm;
  IS           vis, Vis, pis;
  Vec          v0, V0, p0, momrhs, interprhs, contrhs;
  Mat          G, L;
  Vec          Gp, Lv, vbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_SCALAR, &sdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_STAG_SCALAR, &Sdm));
  PetscCall(NSGetField(ns, NS_FIELD_VELOCITY, NULL, NULL, &vis));
  PetscCall(NSGetField(ns, NS_FIELD_FACE_NORMAL_VELOCITY, NULL, NULL, &Vis));
  PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, NULL, &pis));
  PetscCall(VecGetSubVector(ns->sol0, vis, &v0));
  PetscCall(VecGetSubVector(ns->sol0, Vis, &V0));
  PetscCall(VecGetSubVector(f, vis, &momrhs));
  PetscCall(VecGetSubVector(f, Vis, &interprhs));
  PetscCall(VecGetSubVector(f, pis, &contrhs));

  PetscCall(MatCreateSubMatrix(ns->J, vis, pis, MAT_INITIAL_MATRIX, &G));
  PetscCall(PetscObjectQuery((PetscObject)ns->J, "Laplacian", (PetscObject *)&L));
  PetscCall(DMGetGlobalVector(vdm, &Gp));
  PetscCall(DMGetGlobalVector(vdm, &Lv));
  PetscCall(DMGetGlobalVector(vdm, &vbc));
  if (ns->step == 0) {
    PetscCall(VecGetSubVector(ns->sol0, pis, &p0));
    PetscCall(MatMult(G, p0, Gp));
    PetscCall(VecRestoreSubVector(ns->sol0, pis, &p0));
  } else {
    PetscCall(MatMult(G, cnl->phalf, Gp));
  }
  PetscCall(MatMult(L, v0, Lv));
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(vdm, ns->bcs, ns->t, vbc));
  PetscCall(VecAXPY(Lv, 1., vbc));
  PetscCall(VecAXPBYPCZ(momrhs, 1., 0.5 * ns->mu * ns->dt / ns->rho, 0., v0, Lv));
  PetscCall(ComputeConvectionBoundaryConditionVector_Private(vdm, ns->bcs, ns->t, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(momrhs, -ns->dt, vbc));
  PetscCall(VecAXPY(momrhs, -1., Gp));
  PetscCall(ComputeVelocityLaplacianBoundaryConditionVector_Private(vdm, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecAXPY(momrhs, 0.5 * ns->mu * ns->dt / ns->rho, vbc));
  PetscCall(MatDestroy(&G));
  PetscCall(DMRestoreGlobalVector(vdm, &Gp));
  PetscCall(DMRestoreGlobalVector(vdm, &Lv));
  PetscCall(DMRestoreGlobalVector(vdm, &vbc));

  /* interprhs is not a vector on Sdm so setting values directly on it is not safe */
  PetscCall(DMGetGlobalVector(Sdm, &vbc));
  PetscCall(ComputeFaceNormalVelocityInterpolationBoundaryConditionVector_Private(Sdm, ns->bcs, ns->t + ns->dt, vbc));
  PetscCall(VecCopy(vbc, interprhs));
  PetscCall(DMRestoreGlobalVector(Sdm, &vbc));

  PetscCall(VecSet(contrhs, 0.));

  PetscCall(VecRestoreSubVector(ns->sol0, vis, &v0));
  PetscCall(VecRestoreSubVector(ns->sol0, Vis, &V0));
  PetscCall(VecRestoreSubVector(f, vis, &momrhs));
  PetscCall(VecRestoreSubVector(f, Vis, &interprhs));
  PetscCall(VecRestoreSubVector(f, pis, &contrhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
