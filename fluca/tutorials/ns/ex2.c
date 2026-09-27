#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "2D Taylor-Green vortex with NS\n"
                           "Exact solution, nu = mu/rho:\n"
                           "  u =  sin(x) cos(y) exp(-2 nu t)\n"
                           "  v = -cos(x) sin(y) exp(-2 nu t)\n"
                           "  p = rho/4 (cos(2x) + cos(2y)) exp(-4 nu t)\n"
                           "The solution holds the extrapolated p^{n+1}, compared with the exact pressure at the final time.\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : grid cells per direction (default: 32)\n"
                           "  -periodic : periodic [0, 2 pi]^2 instead of walled [0, pi]^2 (default: false)\n";

typedef struct {
  PetscReal rho, nu;
} AppCtx;

static PetscErrorCode ExactVelocity_Private(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  AppCtx   *app   = (AppCtx *)ctx;
  PetscReal decay = PetscExpReal(-2. * app->nu * t);

  PetscFunctionBeginUser;
  *val = comp == 0 ? PetscSinReal(x[0]) * PetscCosReal(x[1]) * decay : -PetscCosReal(x[0]) * PetscSinReal(x[1]) * decay;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExactVelocityDot_Private(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  AppCtx *app = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(ExactVelocity_Private(dim, t, x, comp, val, ctx));
  *val *= -2. * app->nu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Cell velocity, face-normal velocity and cell pressure of the exact solution at time t */
static PetscErrorCode FillExact_Private(Phys phys, AppCtx *app, PetscReal t, Vec X)
{
  DM                  dm;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            c_vel, c_U, c_p, xs, ys, xm, ym, nx, ny, slot_elem, slot_prev, i, j;
  PetscReal           decay_v = PetscExpReal(-2. * app->nu * t), decay_p = PetscExpReal(-4. * app->nu * t);

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(VecZeroEntries(X));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, &nx, &ny, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &slot_prev));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  for (j = ys; j < ys + ym + ny; ++j) {
    for (i = xs; i < xs + xm + nx; ++i) {
      PetscReal     xe = PetscRealPart(arrc[0][i][slot_elem]), ye = PetscRealPart(arrc[1][j][slot_elem]);
      PetscReal     xf = PetscRealPart(arrc[0][i][slot_prev]), yf = PetscRealPart(arrc[1][j][slot_prev]);
      DMStagStencil st;
      PetscScalar   v;

      st.i = i;
      st.j = j;
      st.k = 0;
      if (i < xs + xm && j < ys + ym) {
        st.loc = DMSTAG_ELEMENT;
        st.c   = c_vel;
        v      = PetscSinReal(xe) * PetscCosReal(ye) * decay_v;
        PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
        st.c = c_vel + 1;
        v    = -PetscCosReal(xe) * PetscSinReal(ye) * decay_v;
        PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
        st.c = c_p;
        v    = app->rho / 4. * (PetscCosReal(2. * xe) + PetscCosReal(2. * ye)) * decay_p;
        PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
      }
      st.c = c_U;
      if (j < ys + ym) {
        st.loc = DMSTAG_LEFT;
        v      = PetscSinReal(xf) * PetscCosReal(ye) * decay_v;
        PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
      }
      if (i < xs + xm) {
        st.loc = DMSTAG_DOWN;
        v      = -PetscCosReal(xe) * PetscSinReal(yf) * decay_v;
        PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Discrete L2 error of one field; with remove_mean, the mean of the computed field is removed first */
static PetscErrorCode FieldError_Private(NS ns, const char name[], Vec exact, PetscReal cell_volume, PetscBool remove_mean, PetscReal *err)
{
  IS          is;
  Vec         sol, s, e, diff;
  PetscScalar sum;
  PetscInt    n;

  PetscFunctionBeginUser;
  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(NSGetField(ns, name, &is));
  PetscCall(VecGetSubVector(sol, is, &s));
  PetscCall(VecGetSubVector(exact, is, &e));
  PetscCall(VecDuplicate(s, &diff));
  PetscCall(VecCopy(s, diff));
  if (remove_mean) {
    PetscCall(VecSum(diff, &sum));
    PetscCall(VecGetSize(diff, &n));
    PetscCall(VecShift(diff, -sum / n));
  }
  PetscCall(VecAXPY(diff, -1., e));
  PetscCall(VecNorm(diff, NORM_2, err));
  *err *= PetscSqrtReal(cell_volume);
  PetscCall(VecDestroy(&diff));
  PetscCall(VecRestoreSubVector(exact, is, &e));
  PetscCall(VecRestoreSubVector(sol, is, &s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  Phys      phys;
  NS        ns;
  Vec       sol, exact;
  AppCtx    app;
  PhysBC    bc;
  PetscBool periodic = PETSC_FALSE;
  PetscReal L, mu, t, err_u, err_p;
  PetscInt  Nx, Ny, f;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-periodic", &periodic, NULL));
  L = periodic ? 2. * PETSC_PI : PETSC_PI;

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, periodic ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, periodic ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, 32, 32, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., L, 0., L, 0., 0.));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, NULL));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(phys, dm));
  if (!periodic) {
    bc.type       = PHYS_BC_VELOCITY;
    bc.fn         = ExactVelocity_Private;
    bc.ctx        = &app;
    bc.fn_dot     = ExactVelocityDot_Private;
    bc.fn_dot_ctx = &app;
    for (f = 0; f < 4; f++) PetscCall(PhysSetBoundaryCondition(phys, f, bc));
  }
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysGetDensity(phys, &app.rho));
  PetscCall(PhysGetViscosity(phys, &mu));
  app.nu = mu / app.rho;

  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSCNLINEAR));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(FillExact_Private(phys, &app, 0., sol));
  PetscCall(NSSolve(ns));

  PetscCall(NSGetTime(ns, &t));
  PetscCall(VecDuplicate(sol, &exact));
  PetscCall(FillExact_Private(phys, &app, t, exact));
  PetscCall(FieldError_Private(ns, PHYS_FIELD_VELOCITY, exact, (L / Nx) * (L / Ny), PETSC_FALSE, &err_u));
  PetscCall(FieldError_Private(ns, PHYS_FIELD_PRESSURE, exact, (L / Nx) * (L / Ny), PETSC_TRUE, &err_p));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time %g: velocity L2 error %g, pressure L2 error %g\n", (double)t, (double)err_u, (double)err_p));

  PetscCall(VecDestroy(&exact));
  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: periodic
    nsize: 1
    args: -periodic -stag_grid_x 16 -stag_grid_y 16 -phys_viscosity 0.1 -ns_time_step_size 0.01 -ns_max_steps 10

  test:
    suffix: walls
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -phys_viscosity 0.1 -ns_time_step_size 0.01 -ns_max_steps 10

TEST*/
