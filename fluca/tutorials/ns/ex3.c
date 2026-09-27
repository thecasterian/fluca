#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "3D Ethier-Steinman flow with NS\n"
                           "Exact solution on [-1, 1]^3, a = pi/4, d = pi/2, nu = mu/rho, E = exp(-nu d^2 t):\n"
                           "  u = -a [e^{ax} sin(ay+dz) + e^{az} cos(ax+dy)] E\n"
                           "  v = -a [e^{ay} sin(az+dx) + e^{ax} cos(ay+dz)] E\n"
                           "  w = -a [e^{az} sin(ax+dy) + e^{ay} cos(az+dx)] E\n"
                           "  p = -rho a^2/2 [e^{2ax}+e^{2ay}+e^{2az}\n"
                           "                  + 2 sin(ax+dy) cos(az+dx) e^{a(y+z)}\n"
                           "                  + 2 sin(ay+dz) cos(ax+dy) e^{a(z+x)}\n"
                           "                  + 2 sin(az+dx) cos(ay+dz) e^{a(x+y)}] E^2\n"
                           "The solution holds the extrapolated p^{n+1}, compared with the exact pressure at the final time.\n"
                           "The exact pressure does not have zero mean on this domain: the mean is removed from both the\n"
                           "computed and the exact pressure before differencing.\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int>, -stag_grid_z <int> : grid cells per direction (default: 16)\n";

#define EX3_A (PETSC_PI / 4.)
#define EX3_D (PETSC_PI / 2.)

typedef struct {
  PetscReal rho, nu;
} AppCtx;

static PetscErrorCode ExactVelocity_Private(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  AppCtx   *app   = (AppCtx *)ctx;
  PetscReal decay = PetscExpReal(-app->nu * EX3_D * EX3_D * t);
  PetscReal xx = x[0], yy = x[1], zz = x[2];

  PetscFunctionBeginUser;
  if (comp == 0) *val = -EX3_A * (PetscExpReal(EX3_A * xx) * PetscSinReal(EX3_A * yy + EX3_D * zz) + PetscExpReal(EX3_A * zz) * PetscCosReal(EX3_A * xx + EX3_D * yy)) * decay;
  else if (comp == 1) *val = -EX3_A * (PetscExpReal(EX3_A * yy) * PetscSinReal(EX3_A * zz + EX3_D * xx) + PetscExpReal(EX3_A * xx) * PetscCosReal(EX3_A * yy + EX3_D * zz)) * decay;
  else *val = -EX3_A * (PetscExpReal(EX3_A * zz) * PetscSinReal(EX3_A * xx + EX3_D * yy) + PetscExpReal(EX3_A * yy) * PetscCosReal(EX3_A * zz + EX3_D * xx)) * decay;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExactVelocityDot_Private(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  AppCtx *app = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(ExactVelocity_Private(dim, t, x, comp, val, ctx));
  *val *= -app->nu * EX3_D * EX3_D;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscScalar ExactPressure_Private(AppCtx *app, PetscReal t, PetscReal xx, PetscReal yy, PetscReal zz)
{
  PetscReal decay = PetscExpReal(-2. * app->nu * EX3_D * EX3_D * t);
  PetscReal sum;

  sum = PetscExpReal(2. * EX3_A * xx) + PetscExpReal(2. * EX3_A * yy) + PetscExpReal(2. * EX3_A * zz);
  sum += 2. * PetscSinReal(EX3_A * xx + EX3_D * yy) * PetscCosReal(EX3_A * zz + EX3_D * xx) * PetscExpReal(EX3_A * (yy + zz));
  sum += 2. * PetscSinReal(EX3_A * yy + EX3_D * zz) * PetscCosReal(EX3_A * xx + EX3_D * yy) * PetscExpReal(EX3_A * (zz + xx));
  sum += 2. * PetscSinReal(EX3_A * zz + EX3_D * xx) * PetscCosReal(EX3_A * yy + EX3_D * zz) * PetscExpReal(EX3_A * (xx + yy));
  return -app->rho * EX3_A * EX3_A / 2. * sum * decay;
}

/* Cell velocity, face-normal velocity and cell pressure of the exact solution at time t */
static PetscErrorCode FillExact_Private(Phys phys, AppCtx *app, PetscReal t, Vec X)
{
  DM                  dm;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            c_vel, c_U, c_p, xs, ys, zs, xm, ym, zm, nx, ny, nz, slot_elem, slot_prev, i, j, k;
  PetscReal           px[3];

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(VecZeroEntries(X));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm, &nx, &ny, &nz));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &slot_prev));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  for (k = zs; k < zs + zm + nz; ++k) {
    for (j = ys; j < ys + ym + ny; ++j) {
      for (i = xs; i < xs + xm + nx; ++i) {
        PetscReal     xe = PetscRealPart(arrc[0][i][slot_elem]), ye = PetscRealPart(arrc[1][j][slot_elem]), ze = PetscRealPart(arrc[2][k][slot_elem]);
        PetscReal     xf = PetscRealPart(arrc[0][i][slot_prev]), yf = PetscRealPart(arrc[1][j][slot_prev]), zf = PetscRealPart(arrc[2][k][slot_prev]);
        DMStagStencil st;
        PetscScalar   v;

        st.i = i;
        st.j = j;
        st.k = k;
        if (i < xs + xm && j < ys + ym && k < zs + zm) {
          st.loc = DMSTAG_ELEMENT;
          px[0]  = xe;
          px[1]  = ye;
          px[2]  = ze;
          st.c   = c_vel;
          PetscCall(ExactVelocity_Private(3, t, px, 0, &v, app));
          PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
          st.c = c_vel + 1;
          PetscCall(ExactVelocity_Private(3, t, px, 1, &v, app));
          PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
          st.c = c_vel + 2;
          PetscCall(ExactVelocity_Private(3, t, px, 2, &v, app));
          PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
          st.c = c_p;
          v    = ExactPressure_Private(app, t, xe, ye, ze);
          PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
        }
        st.c = c_U;
        if (j < ys + ym && k < zs + zm) {
          st.loc = DMSTAG_LEFT;
          px[0]  = xf;
          px[1]  = ye;
          px[2]  = ze;
          PetscCall(ExactVelocity_Private(3, t, px, 0, &v, app));
          PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
        }
        if (i < xs + xm && k < zs + zm) {
          st.loc = DMSTAG_DOWN;
          px[0]  = xe;
          px[1]  = yf;
          px[2]  = ze;
          PetscCall(ExactVelocity_Private(3, t, px, 1, &v, app));
          PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
        }
        if (i < xs + xm && j < ys + ym) {
          st.loc = DMSTAG_BACK;
          px[0]  = xe;
          px[1]  = ye;
          px[2]  = zf;
          PetscCall(ExactVelocity_Private(3, t, px, 2, &v, app));
          PetscCall(DMStagVecSetValuesStencil(dm, X, 1, &st, &v, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Discrete L2 error of one field; the mean is removed from both fields before differencing when remove_mean */
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
    Vec e_shifted;

    PetscCall(VecSum(diff, &sum));
    PetscCall(VecGetSize(diff, &n));
    PetscCall(VecShift(diff, -sum / n));
    PetscCall(VecDuplicate(e, &e_shifted));
    PetscCall(VecCopy(e, e_shifted));
    PetscCall(VecSum(e_shifted, &sum));
    PetscCall(VecShift(e_shifted, -sum / n));
    PetscCall(VecAXPY(diff, -1., e_shifted));
    PetscCall(VecDestroy(&e_shifted));
  } else PetscCall(VecAXPY(diff, -1., e));
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
  PetscReal mu, t, err_u, err_p;
  PetscInt  Nx, Ny, Nz, f;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 16, 16, 16, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, -1., 1., -1., 1., -1., 1.));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, &Nz));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(phys, dm));
  bc.type       = PHYS_BC_VELOCITY;
  bc.fn         = ExactVelocity_Private;
  bc.ctx        = &app;
  bc.fn_dot     = ExactVelocityDot_Private;
  bc.fn_dot_ctx = &app;
  for (f = 0; f < 6; f++) PetscCall(PhysSetBoundaryCondition(phys, f, bc));
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
  PetscCall(FieldError_Private(ns, PHYS_FIELD_VELOCITY, exact, (2. / Nx) * (2. / Ny) * (2. / Nz), PETSC_FALSE, &err_u));
  PetscCall(FieldError_Private(ns, PHYS_FIELD_PRESSURE, exact, (2. / Nx) * (2. / Ny) * (2. / Nz), PETSC_TRUE, &err_p));
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
    suffix: ethier_steinman
    nsize: 1
    args: -stag_grid_x 8 -stag_grid_y 8 -stag_grid_z 8 -phys_viscosity 0.1 -ns_time_step_size 0.01 -ns_max_steps 10

TEST*/
