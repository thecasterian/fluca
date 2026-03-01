#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

static const char help[] = "Taylor-Green vortex decay on [0, 2*pi]^2 (rho=mu=1)\n"
                           "  u =  sin(x)*cos(y)*exp(-2t)\n"
                           "  v = -cos(x)*sin(y)*exp(-2t)\n"
                           "  p = (cos(2x) + cos(2y))/4 * exp(-4t)\n"
                           "Options:\n"
                           "  -stag_grid_x <int>  : Grid size in x (default: 16)\n"
                           "  -stag_grid_y <int>  : Grid size in y (default: 16)\n"
                           "  -ts_max_time <real> : Final time (default: 0.1)\n";

typedef struct {
  PetscReal t;
} UnsteadyBCCtx;

/* Exact solution at time t */
static PetscErrorCode ExactSolution(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar u[])
{
  PetscReal e2t = PetscExpReal(-2.0 * t);

  PetscFunctionBeginUser;
  u[0] = PetscSinReal(x[0]) * PetscCosReal(x[1]) * e2t;
  u[1] = -PetscCosReal(x[0]) * PetscSinReal(x[1]) * e2t;
  u[2] = (PetscCosReal(2.0 * x[0]) + PetscCosReal(2.0 * x[1])) / 4.0 * e2t * e2t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* BC callback: exact velocity at boundary using current time from context */
static PetscErrorCode BCVelocity(PetscInt dim, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  UnsteadyBCCtx *bc_ctx = (UnsteadyBCCtx *)ctx;
  PetscScalar    u[3];

  PetscFunctionBeginUser;
  PetscCall(ExactSolution(dim, bc_ctx->t, x, u));
  *val = u[comp];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Update BC time before each TS stage */
static PetscErrorCode PreStage(TS ts, PetscReal stagetime)
{
  UnsteadyBCCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &ctx));
  ctx->t = stagetime;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set initial condition from exact solution at t=0 */
static PetscErrorCode SetInitialCondition(DM sol_dm, Vec x)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem;
  PetscInt            i, j, d;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     coords[2];
      PetscScalar   exact[3];
      DMStagStencil stencil;

      coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
      coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
      PetscCall(ExactSolution(2, 0.0, coords, exact));

      stencil.i   = i;
      stencil.j   = j;
      stencil.k   = 0;
      stencil.loc = DMSTAG_ELEMENT;
      for (d = 0; d < 3; d++) {
        stencil.c = d;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, x, 1, &stencil, &exact[d], INSERT_VALUES));
      }
    }
  }

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM            dm, sol_dm;
  Phys          phys;
  TS            ts;
  Vec           x;
  PetscInt      Nx, Ny, f;
  PetscReal     T;
  PhysINSBC     bc;
  UnsteadyBCCtx bc_ctx;
  PetscReal     l2_vel_err, l2_p_err;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create 2D base DMStag */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 16, 16, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, NULL));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0.0, 2.0 * PETSC_PI, 0.0, 2.0 * PETSC_PI, 0.0, 0.0));

  /* Create and configure Phys */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));

  /* Set velocity Dirichlet BCs on all 4 faces with time-dependent context */
  bc_ctx.t = 0.0;
  bc.type  = PHYS_INS_BC_VELOCITY;
  bc.fn    = BCVelocity;
  bc.ctx   = &bc_ctx;
  for (f = 0; f < 4; f++) PetscCall(PhysINSSetBoundaryCondition(phys, f, bc));

  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  /* Create solution vector and set initial condition */
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMCreateGlobalVector(sol_dm, &x));
  PetscCall(SetInitialCondition(sol_dm, x));

  /* Create TS and set up solver */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(PhysSetUpTS(phys, ts));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetApplicationContext(ts, &bc_ctx));
  PetscCall(TSSetPreStage(ts, PreStage));
  PetscCall(TSSetFromOptions(ts));

  /* Solve */
  PetscCall(TSSolve(ts, x));
  PetscCall(TSGetTime(ts, &T));

  /* Compute L2 errors at final time */
  {
    Vec                 x_exact, err, sub;
    IS                  is_vel, is_p;
    DMStagStencil       vel_stencils[2], p_stencil;
    const PetscScalar **arrc[3] = {NULL, NULL, NULL};
    PetscInt            xs, ys, xm, ym, slot_elem;
    PetscInt            i, j, d;
    PetscScalar         sum_h, sum_exact;
    PetscInt            N_cells;

    N_cells = Nx * Ny;

    /* Build exact solution vector at final time */
    PetscCall(DMCreateGlobalVector(sol_dm, &x_exact));
    PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
    PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], NULL));
    PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal     coords[2];
        PetscScalar   exact[3];
        DMStagStencil st;

        coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
        coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
        PetscCall(ExactSolution(2, T, coords, exact));

        st.i   = i;
        st.j   = j;
        st.k   = 0;
        st.loc = DMSTAG_ELEMENT;
        for (d = 0; d < 3; d++) {
          st.c = d;
          PetscCall(DMStagVecSetValuesStencil(sol_dm, x_exact, 1, &st, &exact[d], INSERT_VALUES));
        }
      }
    }
    PetscCall(VecAssemblyBegin(x_exact));
    PetscCall(VecAssemblyEnd(x_exact));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], NULL));

    /* Create IS for velocity (components 0,1) and pressure (component 2) */
    for (d = 0; d < 2; d++) {
      vel_stencils[d].i   = 0;
      vel_stencils[d].j   = 0;
      vel_stencils[d].k   = 0;
      vel_stencils[d].loc = DMSTAG_ELEMENT;
      vel_stencils[d].c   = d;
    }
    p_stencil.i   = 0;
    p_stencil.j   = 0;
    p_stencil.k   = 0;
    p_stencil.loc = DMSTAG_ELEMENT;
    p_stencil.c   = 2;
    PetscCall(DMStagCreateISFromStencils(sol_dm, 2, vel_stencils, &is_vel));
    PetscCall(DMStagCreateISFromStencils(sol_dm, 1, &p_stencil, &is_p));

    /* Shift exact pressure to match computed mean (pressure defined up to a constant) */
    PetscCall(VecGetSubVector(x, is_p, &sub));
    PetscCall(VecSum(sub, &sum_h));
    PetscCall(VecRestoreSubVector(x, is_p, &sub));
    PetscCall(VecGetSubVector(x_exact, is_p, &sub));
    PetscCall(VecSum(sub, &sum_exact));
    PetscCall(VecShift(sub, PetscRealPart(sum_h - sum_exact) / N_cells));
    PetscCall(VecRestoreSubVector(x_exact, is_p, &sub));

    /* err = x - x_exact */
    PetscCall(VecDuplicate(x, &err));
    PetscCall(VecCopy(x, err));
    PetscCall(VecAXPY(err, -1.0, x_exact));

    /* Velocity RMS error */
    PetscCall(VecGetSubVector(err, is_vel, &sub));
    PetscCall(VecNorm(sub, NORM_2, &l2_vel_err));
    l2_vel_err /= PetscSqrtReal((PetscReal)N_cells);
    PetscCall(VecRestoreSubVector(err, is_vel, &sub));

    /* Pressure RMS error */
    PetscCall(VecGetSubVector(err, is_p, &sub));
    PetscCall(VecNorm(sub, NORM_2, &l2_p_err));
    l2_p_err /= PetscSqrtReal((PetscReal)N_cells);
    PetscCall(VecRestoreSubVector(err, is_p, &sub));

    PetscCall(VecDestroy(&err));
    PetscCall(VecDestroy(&x_exact));
    PetscCall(ISDestroy(&is_vel));
    PetscCall(ISDestroy(&is_p));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Grid: %" PetscInt_FMT " x %" PetscInt_FMT "\n", Nx, Ny));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final time: %g\n", (double)T));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Velocity L2 error: %.4e\n", (double)l2_vel_err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Pressure L2 error: %.4e\n", (double)l2_p_err));

  /* Cleanup */
  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  test:
    suffix: beuler_default
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -ts_max_time 0.1 -ts_type beuler -ts_dt 0.01 -snes_type newtonls -pc_type lu -pc_factor_shift_type nonzero -phys_ins_flucafd_limiter superbee

  test:
    suffix: beuler_refined
    nsize: 1
    args: -stag_grid_x 32 -stag_grid_y 32 -ts_max_time 0.1 -ts_type beuler -ts_dt 0.005 -snes_type newtonls -pc_type lu -pc_factor_shift_type nonzero -phys_ins_flucafd_limiter superbee

TEST*/
