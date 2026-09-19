#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "Temporal self-convergence of TSFSM on the 2D Taylor-Green vortex\n"
                           "Solves to -ts_max_time with dt, dt/2, dt/4, ... and prints ||X_dt - X_{dt/2}|| for u and p.\n"
                           "The final pressure is extrapolated from the last two half-step values (Armfield & Street).\n"
                           "Options:\n"
                           "  -walled        : Unit square with the exact time-dependent wall velocity (default: periodic [0, 2*pi]^2)\n"
                           "  -mu <real>     : Dynamic viscosity with rho = 1 (default: 1.0)\n"
                           "  -dt <real>     : Largest time step (default: 0.02)\n"
                           "  -nlevels <int> : Number of time steps (default: 4)\n";

typedef struct {
  PetscReal nu;
  IS        is_p;
  Vec       p_prev; /* pressure before the current step, p^{n-1/2} */
} AppCtx;

static PetscScalar TGVVelocity(PetscInt comp, PetscReal nu, PetscReal t, PetscReal x, PetscReal y)
{
  PetscReal decay = PetscExpReal(-2. * nu * t);

  return comp == 0 ? -PetscCosReal(x) * PetscSinReal(y) * decay : PetscSinReal(x) * PetscCosReal(y) * decay;
}

static PetscErrorCode WallVelocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  *val = TGVVelocity(comp, user->nu, t, x[0], x[1]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SavePressure(TS ts)
{
  AppCtx *user;
  Vec     X, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &user));
  PetscCall(TSGetSolution(ts, &X));
  PetscCall(VecGetSubVector(X, user->is_p, &p));
  PetscCall(VecCopy(p, user->p_prev));
  PetscCall(VecRestoreSubVector(X, user->is_p, &p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillInitialCondition(Phys phys, AppCtx *user, Vec Y)
{
  DM                  sol_dm;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            c_vel, c_p, xs, ys, xm, ym, slot_elem, i, j;

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil st[3];
      PetscInt      c;

      vals[0] = TGVVelocity(0, user->nu, 0., x, y);
      vals[1] = TGVVelocity(1, user->nu, 0., x, y);
      vals[2] = -0.25 * (PetscCosReal(2. * x) + PetscCosReal(2. * y));
      for (c = 0; c < 3; c++) {
        st[c].i   = i;
        st[c].j   = j;
        st[c].k   = 0;
        st[c].loc = DMSTAG_ELEMENT;
        st[c].c   = c < 2 ? c_vel + c : c_p;
      }
      PetscCall(DMStagVecSetValuesStencil(sol_dm, Y, 3, st, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solve with time step dt; on return the pressure field of Y holds the mean-free extrapolated p^N */
static PetscErrorCode Solve(Phys phys, AppCtx *user, PetscReal dt, PetscReal tmax, Vec Y)
{
  TS          ts;
  Vec         p;
  PetscScalar mean;
  PetscInt    np;

  PetscFunctionBeginUser;
  PetscCall(FillInitialCondition(phys, user, Y));
  PetscCall(TSCreate(PetscObjectComm((PetscObject)phys), &ts));
  PetscCall(PhysSetUpTS(phys, ts));
  PetscCall(TSSetApplicationContext(ts, user));
  PetscCall(TSSetPreStep(ts, SavePressure));
  PetscCall(TSSetMaxTime(ts, tmax));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, Y));
  PetscCall(TSDestroy(&ts));

  PetscCall(VecGetSubVector(Y, user->is_p, &p));
  PetscCall(VecAXPBY(p, -0.5, 1.5, user->p_prev));
  PetscCall(VecGetSize(p, &np));
  PetscCall(VecSum(p, &mean));
  PetscCall(VecShift(p, -mean / np));
  PetscCall(VecRestoreSubVector(Y, user->is_p, &p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ||a - b||_2 over one field, scaled by the cell width */
static PetscErrorCode FieldDifference(IS is, Vec a, Vec b, PetscReal h, PetscReal *e)
{
  Vec sa, sb, d;

  PetscFunctionBeginUser;
  PetscCall(VecGetSubVector(a, is, &sa));
  PetscCall(VecGetSubVector(b, is, &sb));
  PetscCall(VecDuplicate(sa, &d));
  PetscCall(VecWAXPY(d, -1., sb, sa));
  PetscCall(VecNorm(d, NORM_2, e));
  *e *= h;
  PetscCall(VecDestroy(&d));
  PetscCall(VecRestoreSubVector(b, is, &sb));
  PetscCall(VecRestoreSubVector(a, is, &sa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  PhysINSBC bc;
  AppCtx    user;
  IS        is_v;
  Vec       Y[2], sub;
  PetscBool walled = PETSC_FALSE;
  PetscReal mu = 1., dt = 0.02, tmax = 0.1, L, h, e_u, e_p, e_u_prev = 0., e_p_prev = 0.;
  PetscInt  nlevels = 4, N, l, f;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-walled", &walled, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &mu, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ts_max_time", &tmax, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nlevels", &nlevels, NULL));
  user.nu = mu;
  L       = walled ? 1. : 2. * PETSC_PI;

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, walled ? DM_BOUNDARY_NONE : DM_BOUNDARY_PERIODIC, walled ? DM_BOUNDARY_NONE : DM_BOUNDARY_PERIODIC, 32, 32, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., L, 0., L, 0., 0.));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysINSSetDensity(phys, 1.));
  PetscCall(PhysINSSetViscosity(phys, mu));
  if (walled) {
    bc.type       = PHYS_INS_BC_VELOCITY;
    bc.fn         = WallVelocity;
    bc.ctx        = &user;
    bc.fn_dot     = NULL;
    bc.fn_dot_ctx = NULL;
    for (f = 0; f < 4; f++) PetscCall(PhysINSSetBoundaryCondition(phys, f, bc));
  }
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMStagGetGlobalSizes(sol_dm, &N, NULL, NULL));
  h = L / N;

  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_VELOCITY, &is_v));
  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_PRESSURE, &user.is_p));
  PetscCall(DMCreateGlobalVector(sol_dm, &Y[0]));
  PetscCall(DMCreateGlobalVector(sol_dm, &Y[1]));
  PetscCall(VecGetSubVector(Y[0], user.is_p, &sub));
  PetscCall(VecDuplicate(sub, &user.p_prev));
  PetscCall(VecRestoreSubVector(Y[0], user.is_p, &sub));

  for (l = 0; l < nlevels; ++l) {
    PetscReal dtl = dt / PetscPowInt(2, l);

    PetscCall(Solve(phys, &user, dtl, tmax, Y[l % 2]));
    if (l == 0) continue;
    PetscCall(FieldDifference(is_v, Y[0], Y[1], h, &e_u));
    PetscCall(FieldDifference(user.is_p, Y[0], Y[1], h, &e_p));
    if (l == 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dt = %.5f: |u_dt - u_dt/2| = %.4e, |p_dt - p_dt/2| = %.4e\n", (double)dtl, (double)e_u, (double)e_p));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dt = %.5f: |u_dt - u_dt/2| = %.4e (ratio %.2f), |p_dt - p_dt/2| = %.4e (ratio %.2f)\n", (double)dtl, (double)e_u, (double)(e_u_prev / e_u), (double)e_p, (double)(e_p_prev / e_p)));
    e_u_prev = e_u;
    e_p_prev = e_p;
  }

  PetscCall(VecDestroy(&user.p_prev));
  PetscCall(VecDestroy(&Y[1]));
  PetscCall(VecDestroy(&Y[0]));
  PetscCall(ISDestroy(&user.is_p));
  PetscCall(ISDestroy(&is_v));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: periodic
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -nlevels 3 -ts_max_time 0.04 -ts_fsm_ksp_rtol 1e-12 -ts_fsm_abf_momentum_ksp_type preonly -ts_fsm_abf_momentum_pc_type lu -ts_fsm_abf_schur_ksp_type preonly -ts_fsm_abf_schur_pc_type lu -ts_fsm_abf_schur_pc_factor_shift_type nonzero

  test:
    suffix: walled
    nsize: 1
    args: -walled -stag_grid_x 16 -stag_grid_y 16 -nlevels 3 -ts_max_time 0.04 -ts_fsm_ksp_rtol 1e-12 -ts_fsm_abf_momentum_ksp_type preonly -ts_fsm_abf_momentum_pc_type lu -ts_fsm_abf_schur_ksp_type preonly -ts_fsm_abf_schur_pc_type lu -ts_fsm_abf_schur_pc_factor_shift_type nonzero

TEST*/
