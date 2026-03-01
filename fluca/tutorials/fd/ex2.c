#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

static const char help[] = "Solve 1D unsteady convection equation using TVD scheme\n"
                           "Equation: (d/dt) phi = - (d/dx)(rho*u*phi)\n"
                           "Domain: [0,1] with periodic BC, rho = u = 1\n"
                           "Options:\n"
                           "  -flucafd_limiter <type> : Flux limiter (minmod/vanleer/superbee/mc/vanalbada)\n";

typedef struct {
  DM      dm, dm_mass_flux;
  Vec     mass_flux;
  FlucaFD fd_tvd, fd;
} AppCtx;

static PetscErrorCode CreateMassFluxVector(DM dm, DM *dm_mass_flux, Vec *mass_flux)
{
  Vec           mass_flux_local;
  PetscInt      x, m, nExtra, slot, i;
  PetscScalar **arr;

  PetscFunctionBegin;
  PetscCall(DMStagCreateCompatibleDMStag(dm, 1, 0, 0, 0, dm_mass_flux));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm_mass_flux, 0., 1., 0., 0., 0., 0.));

  PetscCall(DMCreateGlobalVector(*dm_mass_flux, mass_flux));
  PetscCall(DMGetLocalVector(*dm_mass_flux, &mass_flux_local));

  PetscCall(DMStagGetCorners(*dm_mass_flux, &x, NULL, NULL, &m, NULL, NULL, &nExtra, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(*dm_mass_flux, DMSTAG_LEFT, 0, &slot));
  PetscCall(DMStagVecGetArray(*dm_mass_flux, mass_flux_local, &arr));

  /* Set F = rho*u = 1 everywhere (rho=1, u=1) */
  for (i = x; i < x + m + nExtra; ++i) arr[i][slot] = 1.;

  PetscCall(DMStagVecRestoreArray(*dm_mass_flux, mass_flux_local, &arr));
  PetscCall(DMLocalToGlobal(*dm_mass_flux, mass_flux_local, INSERT_VALUES, *mass_flux));
  PetscCall(DMRestoreLocalVector(*dm_mass_flux, &mass_flux_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateConvectionOperator(AppCtx *ctx, FlucaFD *fd_conv)
{
  FlucaFD fd_conv_deriv;

  PetscFunctionBegin;

  /* TVD interpolation operator (rho = 1, so no scaling needed) */
  PetscCall(FlucaFDSecondOrderTVDCreate(ctx->dm, FLUCAFD_X, 0, 0, &ctx->fd_tvd));
  PetscCall(FlucaFDSetFromOptions(ctx->fd_tvd));
  PetscCall(FlucaFDSetUp(ctx->fd_tvd));

  /* Derivative operator: d/dx */
  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_conv_deriv));
  PetscCall(FlucaFDSetUp(fd_conv_deriv));

  /* Compose: d/dx(u * phi) */
  PetscCall(FlucaFDCompositionCreate(ctx->fd_tvd, fd_conv_deriv, fd_conv));
  PetscCall(FlucaFDSetUp(*fd_conv));

  /* Cleanup intermediate operators */
  PetscCall(FlucaFDDestroy(&fd_conv_deriv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSFunction(TS ts, PetscReal t, Vec u, Vec F, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;

  PetscFunctionBegin;
  /* Update TVD operator with current solution and velocity */
  PetscCall(FlucaFDSecondOrderTVDSetMassFlux(ctx->fd_tvd, ctx->mass_flux, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd, u));

  PetscCall(FlucaFDApply(ctx->fd, ctx->dm, ctx->dm, u, F));
  /* RHS is negative of convection term */
  PetscCall(VecScale(F, -1.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSJacobian(TS ts, PetscReal t, Vec u, Mat A, Mat P, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(FlucaFDSecondOrderTVDSetMassFlux(ctx->fd_tvd, ctx->mass_flux, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd, u));

  PetscCall(MatZeroEntries(A));
  PetscCall(FlucaFDGetOperator(ctx->fd, ctx->dm, ctx->dm, A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(A, -1.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInitialCondition(DM dm, Vec u)
{
  Vec           u_local;
  PetscInt      x, m, i, slot, slot_coord;
  PetscScalar **arr;
  PetscScalar **arr_coord;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &u_local));
  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot));
  PetscCall(DMStagVecGetArray(dm, u_local, &arr));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));

  /* Square wave: u = 1 for 0.25 <= x <= 0.5, else u = 0 */
  for (i = x; i < x + m; ++i) {
    if (0.25 <= arr_coord[i][slot_coord] && arr_coord[i][slot_coord] <= 0.5) {
      arr[i][slot] = 1.;
    } else {
      arr[i][slot] = 0.;
    }
  }

  PetscCall(DMStagVecRestoreArray(dm, u_local, &arr));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscCall(DMLocalToGlobal(dm, u_local, INSERT_VALUES, u));
  PetscCall(DMRestoreLocalVector(dm, &u_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx ctx;
  TS     ts;
  Mat    A;
  Vec    u;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create DM with periodic boundary */
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 64, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, &ctx.dm));
  PetscCall(DMSetFromOptions(ctx.dm));
  PetscCall(DMSetUp(ctx.dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(ctx.dm, 0., 1., 0., 0., 0., 0.));

  /* Create solution vector and matrix */
  PetscCall(DMCreateGlobalVector(ctx.dm, &u));
  PetscCall(DMCreateMatrix(ctx.dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  /* Create velocity and convection operator */
  PetscCall(CreateMassFluxVector(ctx.dm, &ctx.dm_mass_flux, &ctx.mass_flux));
  PetscCall(CreateConvectionOperator(&ctx, &ctx.fd));

  /* Set initial condition */
  PetscCall(SetInitialCondition(ctx.dm, u));

  /* Create TS */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, ctx.dm));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, ComputeRHSFunction, &ctx));
  PetscCall(TSSetRHSJacobian(ts, A, A, ComputeRHSJacobian, &ctx));
  PetscCall(TSSetMaxTime(ts, 1.));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, 0.001));
  PetscCall(TSSetFromOptions(ts));

  /* Solve */
  PetscCall(TSSolve(ts, u));

  /* Cleanup */
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(FlucaFDDestroy(&ctx.fd));
  PetscCall(FlucaFDDestroy(&ctx.fd_tvd));
  PetscCall(VecDestroy(&ctx.mass_flux));
  PetscCall(DMDestroy(&ctx.dm_mass_flux));
  PetscCall(DMDestroy(&ctx.dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: superbee
    nsize: 1
    args: -flucafd_limiter superbee -stag_grid_x 512 -ts_type ssp -ts_dt 0.002

  test:
    suffix: upwind
    nsize: 1
    args: -flucafd_limiter upwind -stag_grid_x 512 -ts_type ssp -ts_dt 0.002

  test:
    suffix: quick
    nsize: 1
    args: -flucafd_limiter quick -stag_grid_x 512 -ts_type ssp -ts_dt 0.002

TEST*/
