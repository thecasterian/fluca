#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "Test IJacobian by comparing with finite-difference Jacobian\n"
                           "IFunction is linear in (y, y_dot), so they must match exactly\n";

static PetscErrorCode FillExactSolution(DM sol_dm, Vec Y)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem;
  PetscInt            i, j, c, dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil stencils[3];

      vals[0] = -PetscCosReal(x) * PetscSinReal(y);
      vals[1] = PetscSinReal(x) * PetscCosReal(y);
      vals[2] = -0.25 * (PetscCosReal(2 * x) + PetscCosReal(2 * y));

      for (c = 0; c < dim + 1; c++) {
        stencils[c].i   = i;
        stencils[c].j   = j;
        stencils[c].k   = 0;
        stencils[c].loc = DMSTAG_ELEMENT;
        stencils[c].c   = c;
      }
      PetscCall(DMStagVecSetValuesStencil(sol_dm, Y, dim + 1, stencils, vals, INSERT_VALUES));
    }
  }

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                 dm, sol_dm;
  FlucaIB            ib;
  Phys               phys;
  Mat                J, Jfd;
  Vec                Y, Ydot, F0, F1, F2, e_k, y_pert, ydot_pert;
  const PetscScalar *f1_arr;
  PetscReal          shift = 1., t = 0., norm_J, norm_diff;
  PetscInt           N = 4, n, col, row, rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-shift", &shift, NULL));

  /* Create 2D periodic base DMStag */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, N, N, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2 * PETSC_PI, 0., 2 * PETSC_PI, 0., 0.));

  /* Create and set up Phys INS */
  PetscCall(FlucaIBCreateNone(PETSC_COMM_WORLD, dm, &ib));
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetIB(phys, ib));
  PetscCall(FlucaIBDestroy(&ib));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));

  /* Create vectors and fill with TGV */
  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(DMCreateGlobalVector(sol_dm, &Ydot));
  PetscCall(DMCreateGlobalVector(sol_dm, &F0));
  PetscCall(DMCreateGlobalVector(sol_dm, &F1));
  PetscCall(DMCreateGlobalVector(sol_dm, &F2));
  PetscCall(VecDuplicate(Y, &e_k));
  PetscCall(VecDuplicate(Y, &y_pert));
  PetscCall(VecDuplicate(Ydot, &ydot_pert));
  PetscCall(FillExactSolution(sol_dm, Y));
  PetscCall(FillExactSolution(sol_dm, Ydot));

  /* Compute analytic IJacobian: J = dF/dy + shift * dF/dy_dot */
  PetscCall(DMCreateMatrix(sol_dm, &J));
  PetscCall(MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(PhysComputeIJacobian(phys, t, Y, Ydot, shift, J, J));

  /* Build FD Jacobian column by column.
     Since F is linear: J_fd[:, k] = [F(y + e_k, y_dot) - F(y, y_dot)] + shift * [F(y, y_dot + e_k) - F(y, y_dot)]
     No FD epsilon needed — the result is exact for linear operators. */
  PetscCall(VecGetSize(Y, &n));
  PetscCall(VecGetOwnershipRange(Y, &rstart, &rend));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, &Jfd));

  PetscCall(VecZeroEntries(F0));
  PetscCall(PhysComputeIFunction(phys, t, Y, Ydot, F0));

  for (col = 0; col < n; col++) {
    PetscCall(VecZeroEntries(e_k));
    PetscCall(VecSetValue(e_k, col, 1., INSERT_VALUES));
    PetscCall(VecAssemblyBegin(e_k));
    PetscCall(VecAssemblyEnd(e_k));

    /* A[:, k] = F(y + e_k, y_dot) - F(y, y_dot) */
    PetscCall(VecWAXPY(y_pert, 1., e_k, Y));
    PetscCall(VecZeroEntries(F1));
    PetscCall(PhysComputeIFunction(phys, t, y_pert, Ydot, F1));
    PetscCall(VecAXPY(F1, -1., F0));

    /* B[:, k] = F(y, y_dot + e_k) - F(y, y_dot) */
    PetscCall(VecWAXPY(ydot_pert, 1., e_k, Ydot));
    PetscCall(VecZeroEntries(F2));
    PetscCall(PhysComputeIFunction(phys, t, Y, ydot_pert, F2));
    PetscCall(VecAXPY(F2, -1., F0));

    /* J_fd[:, k] = A[:, k] + shift * B[:, k] */
    PetscCall(VecAXPY(F1, shift, F2));

    PetscCall(VecGetArrayRead(F1, &f1_arr));
    for (row = rstart; row < rend; row++) PetscCall(MatSetValue(Jfd, row, col, f1_arr[row - rstart], INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(F1, &f1_arr));
  }

  PetscCall(MatAssemblyBegin(Jfd, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jfd, MAT_FINAL_ASSEMBLY));

  /* Compare: ||J - Jfd|| / ||J|| */
  PetscCall(MatNorm(J, NORM_FROBENIUS, &norm_J));
  PetscCall(MatAXPY(Jfd, -1., J, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(Jfd, NORM_FROBENIUS, &norm_diff));
  PetscCheck(norm_diff / norm_J < 1e-8, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "IJacobian error ||J - Jfd||/||J|| = %g exceeds tolerance", (double)(norm_diff / norm_J));

  PetscCall(VecDestroy(&ydot_pert));
  PetscCall(VecDestroy(&y_pert));
  PetscCall(VecDestroy(&e_k));
  PetscCall(MatDestroy(&Jfd));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&F2));
  PetscCall(VecDestroy(&F1));
  PetscCall(VecDestroy(&F0));
  PetscCall(VecDestroy(&Ydot));
  PetscCall(VecDestroy(&Y));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: shift_0
    nsize: 1
    args: -shift 0
    output_file: output/empty.out

  test:
    suffix: shift_1
    nsize: 1
    args: -shift 1
    output_file: output/empty.out

  test:
    suffix: shift_100
    nsize: 1
    args: -shift 100
    output_file: output/empty.out

TEST*/
