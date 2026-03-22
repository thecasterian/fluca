#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaFD function-based boundary condition values\n"
                           "Options:\n"
                           "  -left_val <real>  : BC value on the left boundary\n"
                           "  -right_val <real> : BC value on the right boundary\n";

/* BC function: returns constant value from ctx */
static PetscErrorCode ConstValBCFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PetscFunctionBeginUser;
  *value = *(PetscScalar *)ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fill input vector with u(x) = x^2 */
static PetscErrorCode FillInputVector(DM dm, Vec u)
{
  Vec                 u_local;
  PetscInt            x, m, i, slot_elem;
  PetscScalar       **arr;
  const PetscScalar **arr_coord;
  PetscInt            slot_coord_elem;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord_elem));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot_elem));
  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, NULL, NULL, NULL));

  PetscCall(DMGetLocalVector(dm, &u_local));
  PetscCall(VecZeroEntries(u_local));
  PetscCall(DMStagVecGetArray(dm, u_local, &arr));
  for (i = x; i < x + m; ++i) {
    PetscReal xi      = PetscRealPart(arr_coord[i][slot_coord_elem]);
    arr[i][slot_elem] = xi * xi;
  }
  PetscCall(DMStagVecRestoreArray(dm, u_local, &arr));
  PetscCall(DMLocalToGlobal(dm, u_local, INSERT_VALUES, u));
  PetscCall(DMRestoreLocalVector(dm, &u_local));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                       dm;
  FlucaFD                  fd_const, fd_fn;
  FlucaFDBoundaryCondition bcs_const[2] = {{0}}, bcs_fn[2] = {{0}};
  Vec                      u, y_const, y_fn;
  PetscReal                norm_diff;
  PetscScalar              left_val = 0., right_val = 0.;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-left_val", &left_val, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-right_val", &right_val, NULL));

  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 0., 0., 0.));

  /* Reference operator with constant BC values */
  PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_const));
  bcs_const[0].value = left_val;
  bcs_const[1].value = right_val;
  PetscCall(FlucaFDSetBoundaryConditions(fd_const, 0, bcs_const));
  PetscCall(FlucaFDSetFromOptions(fd_const));
  PetscCall(FlucaFDSetUp(fd_const));

  /* Function-based operator: ConstValBCFn returns the same values from ctx.
     Set value=99 to verify fn takes priority over value. */
  PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_fn));
  bcs_fn[0].value  = 99.;
  bcs_fn[0].fn     = ConstValBCFn;
  bcs_fn[0].fn_ctx = &left_val;
  bcs_fn[1].value  = 99.;
  bcs_fn[1].fn     = ConstValBCFn;
  bcs_fn[1].fn_ctx = &right_val;
  PetscCall(FlucaFDSetBoundaryConditions(fd_fn, 0, bcs_fn));
  PetscCall(FlucaFDSetFromOptions(fd_fn));
  PetscCall(FlucaFDSetUp(fd_fn));

  /* Apply both and assert results match */
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(FillInputVector(dm, u));

  PetscCall(DMCreateGlobalVector(dm, &y_const));
  PetscCall(DMCreateGlobalVector(dm, &y_fn));
  PetscCall(VecZeroEntries(y_const));
  PetscCall(FlucaFDApply(fd_const, 0., dm, dm, u, y_const));
  PetscCall(VecZeroEntries(y_fn));
  PetscCall(FlucaFDApply(fd_fn, 0., dm, dm, u, y_fn));

  PetscCall(VecAXPY(y_fn, -1.0, y_const));
  PetscCall(VecNorm(y_fn, NORM_INFINITY, &norm_diff));
  PetscCheck(norm_diff < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||y_fn - y_const||_inf = %g exceeds tolerance", (double)norm_diff);

  PetscCall(VecDestroy(&y_fn));
  PetscCall(VecDestroy(&y_const));
  PetscCall(VecDestroy(&u));
  PetscCall(FlucaFDDestroy(&fd_fn));
  PetscCall(FlucaFDDestroy(&fd_const));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: dirichlet_fn
    nsize: 1
    args: -flucafd_0_left_bc_type dirichlet -flucafd_0_right_bc_type dirichlet -left_val 0 -right_val 1
    output_file: output/empty.out

  test:
    suffix: neumann_fn
    nsize: 1
    args: -flucafd_0_left_bc_type neumann -flucafd_0_right_bc_type neumann -left_val 0 -right_val 2
    output_file: output/empty.out

TEST*/
