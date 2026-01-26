#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscksp.h>

static const char help[] = "Solve 2D steady heat equation (Laplacian u = 0)\n"
                           "Domain: [0,1] x [0,1]\n"
                           "BC: u=1 on top (y=1), u=0 on other boundaries\n";

static PetscErrorCode CreateLaplacianOperator(DM dm, FlucaFD *laplacian)
{
  DM      cdm;
  FlucaFD fd_d2[2];

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));

  /* Create d²/dx² operator */
  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_X, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_d2[0]));
  PetscCall(FlucaFDSetUp(fd_d2[0]));

  /* Create d²/dy² operator */
  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_Y, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_d2[1]));
  PetscCall(FlucaFDSetUp(fd_d2[1]));

  /* Create Laplacian = d²/dx² + d²/dy² */
  PetscCall(FlucaFDSumCreate(2, fd_d2, laplacian));

  /* Set boundary conditions: all Dirichlet */
  {
    FlucaFDBoundaryCondition bcs[4];

    /* Left (x=0): u = 0 */
    bcs[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs[0].value = 0.;
    /* Right (x=1): u = 0 */
    bcs[1].type  = FLUCAFD_BC_DIRICHLET;
    bcs[1].value = 0.;
    /* Down (y=0): u = 0 */
    bcs[2].type  = FLUCAFD_BC_DIRICHLET;
    bcs[2].value = 0.;
    /* Up (y=1): u = 1 */
    bcs[3].type  = FLUCAFD_BC_DIRICHLET;
    bcs[3].value = 1.;
    PetscCall(FlucaFDSetBoundaryConditions(*laplacian, bcs));
  }
  PetscCall(FlucaFDSetUp(*laplacian));

  PetscCall(FlucaFDDestroy(&fd_d2[1]));
  PetscCall(FlucaFDDestroy(&fd_d2[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar ExactSolution(PetscScalar x, PetscScalar y)
{
  PetscScalar    u;
  PetscInt       n;
  const PetscInt nterms = 200;

  /*
    Compute exact solution using Fourier series:
    u(x,y) = (2/pi) * sum_{n=1}^{N} [(1 - cos(n*pi)) / (n * sinh(n*pi))] * sin(n*pi*x) * sinh(n*pi*y)
  */
  u = 0.;
  for (n = 1; n <= nterms; ++n) u += (1. - PetscCosScalar(n * PETSC_PI)) / (n * PetscSinhScalar(n * PETSC_PI)) * PetscSinScalar(n * PETSC_PI * x) * PetscSinhScalar(n * PETSC_PI * y);
  return (2. / PETSC_PI) * u;
}

static PetscErrorCode FillExactSolution(DM dm, Vec u_exact)
{
  Vec                 u_local;
  PetscInt            xs, ys, m, n, i, j, slot, slot_coord_x, slot_coord_y;
  PetscScalar      ***arr;
  const PetscScalar **arr_coord_x, **arr_coord_y;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &u_local));
  PetscCall(VecZeroEntries(u_local));

  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord_x, &arr_coord_y, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord_x));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord_y));

  PetscCall(DMStagVecGetArray(dm, u_local, &arr));
  for (j = ys; j < ys + n; ++j)
    for (i = xs; i < xs + m; ++i) arr[j][i][slot] = ExactSolution(arr_coord_x[i][slot_coord_x], arr_coord_y[j][slot_coord_y]);
  PetscCall(DMStagVecRestoreArray(dm, u_local, &arr));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord_x, &arr_coord_y, NULL));
  PetscCall(DMLocalToGlobal(dm, u_local, INSERT_VALUES, u_exact));
  PetscCall(DMRestoreLocalVector(dm, &u_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ValidateSolution(DM dm, Vec u)
{
  Vec       u_exact;
  PetscReal max_err;

  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(dm, &u_exact));
  PetscCall(FillExactSolution(dm, u_exact));
  PetscCall(VecAXPY(u_exact, -1., u));
  PetscCall(VecNorm(u_exact, NORM_INFINITY, &max_err));
  PetscCall(VecDestroy(&u_exact));

  PetscCheck(max_err < 0.05, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Solution validation failed: max error %g exceeds tolerance", max_err);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM      dm;
  FlucaFD laplacian;
  Mat     A;
  Vec     b, u, vbc;
  KSP     ksp;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 16, 16, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  PetscCall(CreateLaplacianOperator(dm, &laplacian));

  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &vbc));

  PetscCall(MatZeroEntries(A));
  PetscCall(VecZeroEntries(vbc));
  PetscCall(FlucaFDApply(laplacian, dm, dm, A, vbc));

  /* RHS = -vbc (since we solve A*u + vbc = 0) */
  PetscCall(VecCopy(vbc, b));
  PetscCall(VecScale(b, -1.));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, u));
  PetscCall(VecViewFromOptions(u, NULL, "-u_view"));

  PetscCall(ValidateSolution(dm, u));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&vbc));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(FlucaFDDestroy(&laplacian));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: basic
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16
    output_file: output/empty.out

  test:
    suffix: fine_grid
    nsize: 1
    args: -stag_grid_x 32 -stag_grid_y 32
    output_file: output/empty.out

TEST*/
