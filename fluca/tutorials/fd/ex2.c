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
  DM      dm, dm_vel;
  Vec     vel;
  FlucaFD fd_tvd, fd;
} AppCtx;

static PetscErrorCode CreateVelocityVector(DM dm, DM *dm_vel, Vec *vel)
{
  Vec           vel_local;
  PetscInt      x, m, nExtra, slot, i;
  PetscScalar **arr;

  PetscFunctionBegin;
  PetscCall(DMStagCreateCompatibleDMStag(dm, 1, 0, 0, 0, dm_vel));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm_vel, 0., 1., 0., 0., 0., 0.));

  PetscCall(DMCreateGlobalVector(*dm_vel, vel));
  PetscCall(DMGetLocalVector(*dm_vel, &vel_local));

  PetscCall(DMStagGetCorners(*dm_vel, &x, NULL, NULL, &m, NULL, NULL, &nExtra, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(*dm_vel, DMSTAG_LEFT, 0, &slot));
  PetscCall(DMStagVecGetArray(*dm_vel, vel_local, &arr));

  /* Set u = 1 everywhere */
  for (i = x; i < x + m + nExtra; ++i) arr[i][slot] = 1.;

  PetscCall(DMStagVecRestoreArray(*dm_vel, vel_local, &arr));
  PetscCall(DMLocalToGlobal(*dm_vel, vel_local, INSERT_VALUES, *vel));
  PetscCall(DMRestoreLocalVector(*dm_vel, &vel_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateConvectionOperator(AppCtx *ctx, FlucaFD *fd_conv)
{
  DM                       cdm;
  FlucaFD                  fd_conv_deriv;
  FlucaFDBoundaryCondition bcs[2];

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(ctx->dm, &cdm));

  /* Periodic boundary conditions */
  bcs[0].type  = FLUCAFD_BC_PERIODIC;
  bcs[0].value = 0.;
  bcs[1].type  = FLUCAFD_BC_PERIODIC;
  bcs[1].value = 0.;

  /* TVD interpolation operator (rho = 1, so no scaling needed) */
  PetscCall(FlucaFDSecondOrderTVDCreate(cdm, FLUCAFD_X, 0, 0, &ctx->fd_tvd));
  PetscCall(FlucaFDSetBoundaryConditions(ctx->fd_tvd, bcs));
  PetscCall(FlucaFDSetFromOptions(ctx->fd_tvd));
  PetscCall(FlucaFDSetUp(ctx->fd_tvd));

  /* Derivative operator: d/dx */
  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_conv_deriv));
  PetscCall(FlucaFDSetBoundaryConditions(fd_conv_deriv, bcs));
  PetscCall(FlucaFDSetUp(fd_conv_deriv));

  /* Compose: d/dx(u * phi) */
  PetscCall(FlucaFDCompositionCreate(ctx->fd_tvd, fd_conv_deriv, fd_conv));
  PetscCall(FlucaFDSetBoundaryConditions(*fd_conv, bcs));
  PetscCall(FlucaFDSetUp(*fd_conv));

  /* Cleanup intermediate operators */
  PetscCall(FlucaFDDestroy(&fd_conv_deriv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSFunction(TS ts, PetscReal t, Vec u, Vec F, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;
  Mat     A;

  PetscFunctionBegin;
  PetscCall(TSGetRHSJacobian(ts, &A, NULL, NULL, NULL));

  /* Update TVD operator with current solution and velocity */
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd, u));

  /* Build the operator matrix */
  PetscCall(MatZeroEntries(A));
  PetscCall(VecZeroEntries(F));
  PetscCall(FlucaFDApply(ctx->fd, ctx->dm, ctx->dm, A, F));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(F));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyEnd(F));

  PetscCall(MatMultAdd(A, u, F, F));

  /* RHS is negative of convection term */
  PetscCall(MatScale(A, -1.));
  PetscCall(VecScale(F, -1.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSJacobian(TS ts, PetscReal t, Vec u, Mat A, Mat P, void *ptr)
{
  PetscFunctionBegin;
  /* Jacobian is computed in ComputeRHSFunction */
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
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 64, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, &ctx.dm));
  PetscCall(DMSetFromOptions(ctx.dm));
  PetscCall(DMSetUp(ctx.dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(ctx.dm, 0., 1., 0., 0., 0., 0.));

  /* Create solution vector and matrix */
  PetscCall(DMCreateGlobalVector(ctx.dm, &u));
  PetscCall(DMCreateMatrix(ctx.dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  /* Create velocity and convection operator */
  PetscCall(CreateVelocityVector(ctx.dm, &ctx.dm_vel, &ctx.vel));
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
  PetscCall(VecViewFromOptions(u, NULL, "-phi_view"));

  /* Cleanup */
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(FlucaFDDestroy(&ctx.fd));
  PetscCall(FlucaFDDestroy(&ctx.fd_tvd));
  PetscCall(VecDestroy(&ctx.vel));
  PetscCall(DMDestroy(&ctx.dm_vel));
  PetscCall(DMDestroy(&ctx.dm));

  PetscCall(FlucaFinalize());
}
