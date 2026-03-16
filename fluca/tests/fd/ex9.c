#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "Test FlucaFD function-based boundary condition values\n"
                           "Options:\n"
                           "  -M <int>  : Number of elements (default 8)\n";

/* Constant-equivalent BC function: returns 2.0 always */
static PetscErrorCode ConstantBCFn(PetscInt dim, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PetscFunctionBeginUser;
  *value = 2.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Position-dependent Dirichlet BC function: value = x[0]^2 */
static PetscErrorCode PositionBCFn(PetscInt dim, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PetscFunctionBeginUser;
  *value = x[0] * x[0];
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
  DM       dm;
  PetscInt M;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  M = 8;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));

  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, M, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 0., 0., 0.));

  /* Test 1: constant-equivalent function BC applied via FlucaFDApply
     Compare result of constant BC value=2.0 vs function BC that returns 2.0.
     Both should give the same derivative values for du/dx with left Dirichlet BC.
     u(x) = x^2, du/dx = 2x, left BC = 0.0 (or 0.0^2=0.0 from function) */
  {
    FlucaFDBoundaryCondition bcs_const[2] = {{0}}, bcs_fn[2] = {{0}};
    FlucaFD                  fd_const, fd_fn;
    Vec                      u, y_const, y_fn;
    PetscScalar              norm_diff;

    /* Create constant-BC derivative operator */
    PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_const));
    bcs_const[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs_const[0].value = 0.; /* u(0) = 0 */
    bcs_const[1].type  = FLUCAFD_BC_DIRICHLET;
    bcs_const[1].value = 1.; /* u(1) = 1 */
    PetscCall(FlucaFDSetBoundaryConditions(fd_const, bcs_const));
    PetscCall(FlucaFDSetUp(fd_const));

    /* Create function-BC derivative operator */
    PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_fn));
    bcs_fn[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs_fn[0].value = 0.;           /* fallback constant, overridden by fn */
    bcs_fn[0].fn    = PositionBCFn; /* PositionBCFn(x=0)=0, matches constant */
    bcs_fn[1].type  = FLUCAFD_BC_DIRICHLET;
    bcs_fn[1].value = 1.;           /* fallback constant, overridden by fn */
    bcs_fn[1].fn    = PositionBCFn; /* PositionBCFn(x=1)=1, matches constant */
    PetscCall(FlucaFDSetBoundaryConditions(fd_fn, bcs_fn));
    PetscCall(FlucaFDSetUp(fd_fn));

    /* Fill input vector u(x) = x^2 */
    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(FillInputVector(dm, u));

    /* Apply both and compare */
    PetscCall(DMCreateGlobalVector(dm, &y_const));
    PetscCall(DMCreateGlobalVector(dm, &y_fn));
    PetscCall(FlucaFDApply(fd_const, dm, dm, u, y_const));
    PetscCall(FlucaFDApply(fd_fn, dm, dm, u, y_fn));

    PetscCall(VecAXPY(y_fn, -1.0, y_const));
    PetscCall(VecNorm(y_fn, NORM_INFINITY, &norm_diff));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test 1 (const-equiv function BC): ||y_fn - y_const||_inf = %g\n", (double)PetscRealPart(norm_diff)));

    PetscCall(VecDestroy(&y_fn));
    PetscCall(VecDestroy(&y_const));
    PetscCall(VecDestroy(&u));
    PetscCall(FlucaFDDestroy(&fd_fn));
    PetscCall(FlucaFDDestroy(&fd_const));
  }

  /* Test 2: Neumann function BC - apply derivative with Neumann BC function
     u(x) = x^2, du/dx|_{x=0} = 0, du/dx|_{x=1} = 2
     For second-order derivative with Neumann left BC = 0, right BC = 2 */
  {
    FlucaFD                  fd_const, fd_fn;
    FlucaFDBoundaryCondition bcs_const[2] = {{0}}, bcs_fn[2] = {{0}};
    Vec                      u, y_const, y_fn;
    PetscScalar              norm_diff;

    /* Constant Neumann BC operator */
    PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_const));
    bcs_const[0].type  = FLUCAFD_BC_NEUMANN;
    bcs_const[0].value = 0.; /* du/dx = 0 at x=0 */
    bcs_const[1].type  = FLUCAFD_BC_NEUMANN;
    bcs_const[1].value = 2.; /* du/dx = 2 at x=1 */
    PetscCall(FlucaFDSetBoundaryConditions(fd_const, bcs_const));
    PetscCall(FlucaFDSetUp(fd_const));

    /* Function Neumann BC operator: constant fn returning 0 and 2 */
    PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_fn));
    bcs_fn[0].type  = FLUCAFD_BC_NEUMANN;
    bcs_fn[0].value = 99.;          /* should be overridden by fn */
    bcs_fn[0].fn    = ConstantBCFn; /* returns 2.0 - different from bcs_const[0].value=0 intentionally */
    bcs_fn[1].type  = FLUCAFD_BC_NEUMANN;
    bcs_fn[1].value = 99.; /* should be overridden by fn */
    PetscCall(FlucaFDSetBoundaryConditions(fd_fn, bcs_fn));
    PetscCall(FlucaFDSetUp(fd_fn));

    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(FillInputVector(dm, u));

    PetscCall(DMCreateGlobalVector(dm, &y_const));
    PetscCall(DMCreateGlobalVector(dm, &y_fn));
    PetscCall(FlucaFDApply(fd_const, dm, dm, u, y_const));
    PetscCall(FlucaFDApply(fd_fn, dm, dm, u, y_fn));

    /* The results should differ because ConstantBCFn returns 2.0 not 0.0 for left BC */
    PetscCall(VecAXPY(y_fn, -1.0, y_const));
    PetscCall(VecNorm(y_fn, NORM_INFINITY, &norm_diff));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test 2 (Neumann function BC differs from const): ||y_fn - y_const||_inf = %g\n", (double)PetscRealPart(norm_diff)));

    PetscCall(VecDestroy(&y_fn));
    PetscCall(VecDestroy(&y_const));
    PetscCall(VecDestroy(&u));
    PetscCall(FlucaFDDestroy(&fd_fn));
    PetscCall(FlucaFDDestroy(&fd_const));
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: bc_fn_const_equiv
    nsize: 1
    args: -M 8

  test:
    suffix: bc_fn_neumann_differs
    nsize: 1
    args: -M 8

TEST*/
