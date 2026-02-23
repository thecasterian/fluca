#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaFDApply with spatially varying boundary condition callback\n"
                           "Domain: [0,1] x [0,1], f(x,y) = x*y, df/dx = y\n";

/* Boundary condition callback: returns f(x,y) = x*y at boundary face center */
static PetscErrorCode BCCallback(PetscInt dim, const PetscReal x[], PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = x[0] * x[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                  dm;
  FlucaFD             fd;
  Vec                 input, output, expected;
  Vec                 input_local, expected_local;
  PetscInt            M = 8, N = 8, xs, ys, xm, ym, i, j, slot, slot_elem;
  PetscScalar      ***arr, ***arr_exp;
  const PetscScalar **arr_coord_x, **arr_coord_y;
  PetscReal           max_error;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create 2D DMStag with element DOFs */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, M, N, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  /* Create first derivative df/dx operator */
  PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd));

  /* Set BCs with callback: f(x,y) = x*y */
  {
    FlucaFDBoundaryCondition bcs[4] = {{0}};

    /* Left (x=0): Dirichlet, f(0,y) = 0 */
    bcs[0].type = FLUCAFD_BC_DIRICHLET;
    bcs[0].fn   = BCCallback;
    /* Right (x=1): Dirichlet, f(1,y) = y */
    bcs[1].type = FLUCAFD_BC_DIRICHLET;
    bcs[1].fn   = BCCallback;
    /* Down (y=0): not used by df/dx */
    bcs[2].type = FLUCAFD_BC_NONE;
    /* Up (y=1): not used by df/dx */
    bcs[3].type = FLUCAFD_BC_NONE;
    PetscCall(FlucaFDSetBoundaryConditions(fd, bcs));
  }
  PetscCall(FlucaFDSetUp(fd));

  /* Fill input vector with f(x,y) = x * y */
  PetscCall(DMCreateGlobalVector(dm, &input));
  PetscCall(DMGetLocalVector(dm, &input_local));
  PetscCall(VecZeroEntries(input_local));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord_x, &arr_coord_y, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagVecGetArray(dm, input_local, &arr));
  for (j = ys; j < ys + ym; j++)
    for (i = xs; i < xs + xm; i++) arr[j][i][slot] = arr_coord_x[i][slot_elem] * arr_coord_y[j][slot_elem];
  PetscCall(DMStagVecRestoreArray(dm, input_local, &arr));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord_x, &arr_coord_y, NULL));
  PetscCall(DMLocalToGlobal(dm, input_local, INSERT_VALUES, input));
  PetscCall(DMRestoreLocalVector(dm, &input_local));

  /* Apply df/dx operator */
  PetscCall(DMCreateGlobalVector(dm, &output));
  PetscCall(FlucaFDApply(fd, dm, dm, input, output));

  /* Build expected result: df/dx of x*y = y */
  PetscCall(VecDuplicate(output, &expected));
  PetscCall(DMGetLocalVector(dm, &expected_local));
  PetscCall(VecZeroEntries(expected_local));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, NULL, &arr_coord_y, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagVecGetArray(dm, expected_local, &arr_exp));
  for (j = ys; j < ys + ym; j++)
    for (i = xs; i < xs + xm; i++) arr_exp[j][i][slot] = arr_coord_y[j][slot_elem];
  PetscCall(DMStagVecRestoreArray(dm, expected_local, &arr_exp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, NULL, &arr_coord_y, NULL));
  PetscCall(DMLocalToGlobal(dm, expected_local, INSERT_VALUES, expected));
  PetscCall(DMRestoreLocalVector(dm, &expected_local));

  /* Compare */
  PetscCall(VecAXPY(expected, -1., output));
  PetscCall(VecNorm(expected, NORM_INFINITY, &max_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "max error = %g\n", (double)max_error));
  PetscCheck(max_error < 1e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Validation failed: max error %g exceeds tolerance", (double)max_error);

  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&output));
  PetscCall(VecDestroy(&input));
  PetscCall(FlucaFDDestroy(&fd));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: bc_callback_2d
    nsize: 1
    output_file: output/ex8_bc_callback_2d.out

TEST*/
