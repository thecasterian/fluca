#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaFDApplyDot: time derivative of BC values\n"
                           "Options:\n"
                           "  -case <int> : 1=constant_bc, 2=steady_fn_bc, 3=timedep_fn_fd_approx, 4=explicit_fn_dot\n";

/* BC functions */
static PetscErrorCode ConstantBCFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PetscFunctionBeginUser;
  *value = 3.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SinTimeBCFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PetscFunctionBeginUser;
  *value = PetscSinReal(t) * 2.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CosDotBCFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PetscFunctionBeginUser;
  *value = PetscCosReal(t) * 2.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fill input vector with u(x) = x */
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
  for (i = x; i < x + m; ++i) arr[i][slot_elem] = PetscRealPart(arr_coord[i][slot_coord_elem]);
  PetscCall(DMStagVecRestoreArray(dm, u_local, &arr));
  PetscCall(DMLocalToGlobal(dm, u_local, INSERT_VALUES, u));
  PetscCall(DMRestoreLocalVector(dm, &u_local));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                       dm;
  FlucaFD                  fd_test, fd_ref;
  FlucaFDBoundaryCondition bcs_test[2], bcs_ref[2];
  Vec                      u, y_test, y_ref;
  PetscReal                norm_diff, tol, t;
  PetscInt                 cas;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  cas = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-case", &cas, NULL));

  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 0., 0., 0.));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(FillInputVector(dm, u));
  PetscCall(DMCreateGlobalVector(dm, &y_test));
  PetscCall(DMCreateGlobalVector(dm, &y_ref));

  PetscCall(PetscMemzero(bcs_test, sizeof(bcs_test)));
  PetscCall(PetscMemzero(bcs_ref, sizeof(bcs_ref)));

  t = 1.;

  switch (cas) {
  case 1:
    /* constant_bc: fn=NULL, fn_dot=NULL, value=5. ApplyDot -> 0. */
    bcs_test[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs_test[0].value = 5.;
    bcs_test[1].type  = FLUCAFD_BC_DIRICHLET;
    bcs_test[1].value = 5.;
    bcs_ref[0].type   = FLUCAFD_BC_DIRICHLET;
    bcs_ref[0].value  = 0.;
    bcs_ref[1].type   = FLUCAFD_BC_DIRICHLET;
    bcs_ref[1].value  = 0.;
    tol               = 1e-10;
    break;
  case 2:
    /* steady_fn_bc: fn returns constant 3, fn_dot=NULL. ApplyDot FD approx ~ 0. */
    bcs_test[0].type = FLUCAFD_BC_DIRICHLET;
    bcs_test[0].fn   = ConstantBCFn;
    bcs_test[1].type = FLUCAFD_BC_DIRICHLET;
    bcs_test[1].fn   = ConstantBCFn;
    bcs_ref[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs_ref[0].value = 0.;
    bcs_ref[1].type  = FLUCAFD_BC_DIRICHLET;
    bcs_ref[1].value = 0.;
    tol              = 1e-4;
    break;
  case 3:
    /* timedep_fn_fd: fn=sin(t)*2, fn_dot=NULL. ApplyDot FD approx ~ cos(1)*2. */
    bcs_test[0].type = FLUCAFD_BC_DIRICHLET;
    bcs_test[0].fn   = SinTimeBCFn;
    bcs_test[1].type = FLUCAFD_BC_DIRICHLET;
    bcs_test[1].fn   = SinTimeBCFn;
    bcs_ref[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs_ref[0].value = PetscCosReal(1.) * 2.;
    bcs_ref[1].type  = FLUCAFD_BC_DIRICHLET;
    bcs_ref[1].value = PetscCosReal(1.) * 2.;
    tol              = 1e-4;
    break;
  case 4:
    /* explicit_fn_dot: fn=sin(t)*2, fn_dot=cos(t)*2. ApplyDot uses fn_dot exactly. */
    bcs_test[0].type   = FLUCAFD_BC_DIRICHLET;
    bcs_test[0].fn     = SinTimeBCFn;
    bcs_test[0].fn_dot = CosDotBCFn;
    bcs_test[1].type   = FLUCAFD_BC_DIRICHLET;
    bcs_test[1].fn     = SinTimeBCFn;
    bcs_test[1].fn_dot = CosDotBCFn;
    bcs_ref[0].type    = FLUCAFD_BC_DIRICHLET;
    bcs_ref[0].value   = PetscCosReal(1.) * 2.;
    bcs_ref[1].type    = FLUCAFD_BC_DIRICHLET;
    bcs_ref[1].value   = PetscCosReal(1.) * 2.;
    tol                = 1e-10;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid -case %" PetscInt_FMT "; must be 1-4", cas);
  }

  PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_test));
  PetscCall(FlucaFDSetBoundaryConditions(fd_test, 0, bcs_test));
  PetscCall(FlucaFDSetUp(fd_test));

  PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_ref));
  PetscCall(FlucaFDSetBoundaryConditions(fd_ref, 0, bcs_ref));
  PetscCall(FlucaFDSetUp(fd_ref));

  PetscCall(FlucaFDApplyDot(fd_test, t, dm, dm, u, y_test));
  PetscCall(FlucaFDApply(fd_ref, t, dm, dm, u, y_ref));

  PetscCall(VecAXPY(y_test, -1., y_ref));
  PetscCall(VecNorm(y_test, NORM_INFINITY, &norm_diff));
  PetscCheck(norm_diff < tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "case %" PetscInt_FMT ": ||ApplyDot - ref||_inf = %g exceeds tol %g", cas, (double)norm_diff, (double)tol);

  PetscCall(FlucaFDDestroy(&fd_ref));
  PetscCall(FlucaFDDestroy(&fd_test));
  PetscCall(VecDestroy(&y_ref));
  PetscCall(VecDestroy(&y_test));
  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: constant_bc
    nsize: 1
    args: -case 1
    output_file: output/empty.out

  test:
    suffix: steady_fn_bc
    nsize: 1
    args: -case 2
    output_file: output/empty.out

  test:
    suffix: timedep_fn_fd_approx
    nsize: 1
    args: -case 3
    output_file: output/empty.out

  test:
    suffix: explicit_fn_dot
    nsize: 1
    args: -case 4
    output_file: output/empty.out

TEST*/
