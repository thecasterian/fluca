#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

static const char help[] = "Manufactured solution test for unsteady Stokes solver\n"
                           "Decaying Taylor-Green vortex:\n"
                           "  u = -cos(pi*x)*sin(pi*y)*exp(-t)\n"
                           "  v =  sin(pi*x)*cos(pi*y)*exp(-t)\n"
                           "  p = -(cos(2*pi*x) + cos(2*pi*y))/4 * exp(-t)\n"
                           "Options:\n"
                           "  -N <int>  : Grid size in each direction (default: 16)\n"
                           "  -T <real> : Final time (default: 0.1)\n";

typedef struct {
  PetscReal t;
} UnsteadyBCCtx;

/* Exact solution at time t */
static PetscErrorCode ExactSolution(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar u[])
{
  PetscReal pi = PETSC_PI;
  PetscReal et = PetscExpReal(-t);

  PetscFunctionBeginUser;
  u[0] = -PetscCosReal(pi * x[0]) * PetscSinReal(pi * x[1]) * et;
  u[1] = PetscSinReal(pi * x[0]) * PetscCosReal(pi * x[1]) * et;
  u[2] = -(PetscCosReal(2.0 * pi * x[0]) + PetscCosReal(2.0 * pi * x[1])) / 4.0 * et;
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

/* Body force for decaying Taylor-Green (rho=mu=1):
   f = rho * du/dt - mu * nabla^2(u) + (1/rho) * grad(p)
   f_x = e^{-t} * [(1 - 2*pi^2)*cos(pi*x)*sin(pi*y) + pi*sin(pi*x)*cos(pi*x)]
   f_y = e^{-t} * [(-1 + 2*pi^2)*sin(pi*x)*cos(pi*y) + pi*sin(pi*y)*cos(pi*y)] */
static PetscErrorCode BodyForce(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx)
{
  PetscReal pi = PETSC_PI;
  PetscReal cx = PetscCosReal(pi * x[0]), sx = PetscSinReal(pi * x[0]);
  PetscReal cy = PetscCosReal(pi * x[1]), sy = PetscSinReal(pi * x[1]);
  PetscReal et = PetscExpReal(-t);

  PetscFunctionBeginUser;
  f[0] = et * ((1.0 - 2.0 * pi * pi) * cx * sy + pi * sx * cx);
  f[1] = et * ((-1.0 + 2.0 * pi * pi) * sx * cy + pi * sy * cy);
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
  DM                  dm, sol_dm;
  Phys                phys;
  TS                  ts;
  Vec                 x;
  PetscInt            N = 16, f;
  PetscReal           T = 0.1;
  PhysINSBC           bc;
  UnsteadyBCCtx       bc_ctx;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem, slot[3];
  PetscInt            i, j, d;
  PetscReal           l2_vel_err2 = 0.0, l2_p_err2 = 0.0;
  PetscReal           p_mean_h = 0.0, p_mean_exact = 0.0;
  PetscReal           l2_vel_err, l2_p_err;
  PetscInt            N_cells;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-T", &T, NULL));

  /* Create 2D base DMStag */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, N, N, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));

  /* Create and configure Phys */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysSetBodyForce(phys, BodyForce, NULL));

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
  PetscCall(TSSetMaxTime(ts, T));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetApplicationContext(ts, &bc_ctx));
  PetscCall(TSSetPreStage(ts, PreStage));
  PetscCall(TSSetFromOptions(ts));

  /* Solve */
  PetscCall(TSSolve(ts, x));

  /* Compute L2 errors at final time */
  {
    Vec                  x_local;
    const PetscScalar ***x_arr;

    PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
    PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
    PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
    for (d = 0; d < 3; d++) PetscCall(DMStagGetLocationSlot(sol_dm, DMSTAG_ELEMENT, d, &slot[d]));

    PetscCall(DMGetLocalVector(sol_dm, &x_local));
    PetscCall(DMGlobalToLocal(sol_dm, x, INSERT_VALUES, x_local));
    PetscCall(DMStagVecGetArrayRead(sol_dm, x_local, (void *)&x_arr));

    /* First pass: compute pressure means */
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal   coords[2];
        PetscScalar exact[3];

        coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
        coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
        PetscCall(ExactSolution(2, T, coords, exact));

        p_mean_h += PetscRealPart(x_arr[j][i][slot[2]]);
        p_mean_exact += PetscRealPart(exact[2]);
      }
    }

    N_cells = N * N;
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &p_mean_h, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &p_mean_exact, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    p_mean_h /= N_cells;
    p_mean_exact /= N_cells;

    /* Second pass: compute L2 errors */
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal   coords[2];
        PetscScalar exact[3];
        PetscReal   diff;

        coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
        coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
        PetscCall(ExactSolution(2, T, coords, exact));

        for (d = 0; d < 2; d++) {
          diff = PetscRealPart(x_arr[j][i][slot[d]]) - PetscRealPart(exact[d]);
          l2_vel_err2 += diff * diff;
        }
        diff = (PetscRealPart(x_arr[j][i][slot[2]]) - p_mean_h) - (PetscRealPart(exact[2]) - p_mean_exact);
        l2_p_err2 += diff * diff;
      }
    }

    PetscCall(DMStagVecRestoreArrayRead(sol_dm, x_local, (void *)&x_arr));
    PetscCall(DMRestoreLocalVector(sol_dm, &x_local));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  }

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &l2_vel_err2, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &l2_p_err2, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));

  l2_vel_err = PetscSqrtReal(l2_vel_err2 / N_cells);
  l2_p_err   = PetscSqrtReal(l2_p_err2 / N_cells);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Grid: %" PetscInt_FMT " x %" PetscInt_FMT "\n", N, N));
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
    args: -N 16 -T 0.1 -ts_type beuler -ts_dt 0.01 -snes_type ksponly -pc_type lu -pc_factor_shift_type nonzero

  test:
    suffix: beuler_refined
    nsize: 1
    args: -N 32 -T 0.1 -ts_type beuler -ts_dt 0.005 -snes_type ksponly -pc_type lu -pc_factor_shift_type nonzero

TEST*/
