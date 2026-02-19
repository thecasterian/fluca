#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

static const char help[] = "Solve 2D unsteady convection-diffusion equation using TVD scheme\n"
                           "Equation: (d/dt)phi = -(d/dx)(rho*u*phi) - (d/dy)(rho*v*phi) + (d/dx)(mu*dphi/dx) + (d/dy)(mu*dphi/dy)\n"
                           "Domain: [0,1]x[0,1] with periodic BC, rho = u = v = 1\n"
                           "Options:\n"
                           "  -mu <real>                : Diffusion coefficient (default: 0.01)\n"
                           "  -flucafd_limiter <type>   : Flux limiter (minmod/vanleer/superbee/mc/vanalbada)\n"
                           "  -ts_monitor_dmda <viewer> : View solution on DMDA at each time step\n";

typedef struct {
  DM          dm, dm_vel;
  Vec         vel;
  PetscScalar mu;
  FlucaFD     fd_tvd_x, fd_tvd_y, fd;
} AppCtx;

static PetscErrorCode CreateVelocityVector(DM dm, DM *dm_vel, Vec *vel)
{
  Vec            vel_local;
  PetscInt       xs, ys, m, n, nExtrax, nExtray, slot_left, slot_down, i, j;
  PetscScalar ***arr;

  PetscFunctionBegin;
  PetscCall(DMStagCreateCompatibleDMStag(dm, 0, 1, 0, 0, dm_vel));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm_vel, 0., 1., 0., 1., 0., 0.));

  PetscCall(DMCreateGlobalVector(*dm_vel, vel));
  PetscCall(DMGetLocalVector(*dm_vel, &vel_local));

  PetscCall(DMStagGetCorners(*dm_vel, &xs, &ys, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetLocationSlot(*dm_vel, DMSTAG_LEFT, 0, &slot_left));
  PetscCall(DMStagGetLocationSlot(*dm_vel, DMSTAG_DOWN, 0, &slot_down));
  PetscCall(DMStagVecGetArray(*dm_vel, vel_local, &arr));

  /* Set u = 1 on LEFT faces, v = 1 on DOWN faces */
  for (j = ys; j < ys + n + nExtray; ++j)
    for (i = xs; i < xs + m + nExtrax; ++i) {
      arr[j][i][slot_left] = 1.;
      arr[j][i][slot_down] = 1.;
    }

  PetscCall(DMStagVecRestoreArray(*dm_vel, vel_local, &arr));
  PetscCall(DMLocalToGlobal(*dm_vel, vel_local, INSERT_VALUES, *vel));
  PetscCall(DMRestoreLocalVector(*dm_vel, &vel_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperator(AppCtx *ctx, FlucaFD *fd_rhs)
{
  FlucaFD fd_conv_deriv_x, fd_conv_x, fd_neg_conv_x;
  FlucaFD fd_conv_deriv_y, fd_conv_y, fd_neg_conv_y;
  FlucaFD fd_diff_inner_x, fd_diff_scaled_x, fd_diff_outer_x, fd_diff_x;
  FlucaFD fd_diff_inner_y, fd_diff_scaled_y, fd_diff_outer_y, fd_diff_y;
  FlucaFD operands[4];

  PetscFunctionBegin;

  /* Convection x: -d/dx(u*phi) */
  PetscCall(FlucaFDSecondOrderTVDCreate(ctx->dm, FLUCAFD_X, 0, 0, &ctx->fd_tvd_x));
  PetscCall(FlucaFDSetFromOptions(ctx->fd_tvd_x));
  PetscCall(FlucaFDSetUp(ctx->fd_tvd_x));

  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_conv_deriv_x));
  PetscCall(FlucaFDSetUp(fd_conv_deriv_x));

  PetscCall(FlucaFDCompositionCreate(ctx->fd_tvd_x, fd_conv_deriv_x, &fd_conv_x));
  PetscCall(FlucaFDSetUp(fd_conv_x));

  PetscCall(FlucaFDScaleCreateConstant(fd_conv_x, -1., &fd_neg_conv_x));
  PetscCall(FlucaFDSetUp(fd_neg_conv_x));

  /* Convection y: -d/dy(v*phi) */
  PetscCall(FlucaFDSecondOrderTVDCreate(ctx->dm, FLUCAFD_Y, 0, 0, &ctx->fd_tvd_y));
  PetscCall(FlucaFDSetFromOptions(ctx->fd_tvd_y));
  PetscCall(FlucaFDSetUp(ctx->fd_tvd_y));

  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_Y, 1, 2, DMSTAG_DOWN, 0, DMSTAG_ELEMENT, 0, &fd_conv_deriv_y));
  PetscCall(FlucaFDSetUp(fd_conv_deriv_y));

  PetscCall(FlucaFDCompositionCreate(ctx->fd_tvd_y, fd_conv_deriv_y, &fd_conv_y));
  PetscCall(FlucaFDSetUp(fd_conv_y));

  PetscCall(FlucaFDScaleCreateConstant(fd_conv_y, -1., &fd_neg_conv_y));
  PetscCall(FlucaFDSetUp(fd_neg_conv_y));

  /* Diffusion x: d/dx(mu*dphi/dx) */
  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 1, 2, DMSTAG_ELEMENT, 0, DMSTAG_LEFT, 0, &fd_diff_inner_x));
  PetscCall(FlucaFDSetUp(fd_diff_inner_x));

  PetscCall(FlucaFDScaleCreateConstant(fd_diff_inner_x, ctx->mu, &fd_diff_scaled_x));
  PetscCall(FlucaFDSetUp(fd_diff_scaled_x));

  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_diff_outer_x));
  PetscCall(FlucaFDSetUp(fd_diff_outer_x));

  PetscCall(FlucaFDCompositionCreate(fd_diff_scaled_x, fd_diff_outer_x, &fd_diff_x));
  PetscCall(FlucaFDSetUp(fd_diff_x));

  /* Diffusion y: d/dy(mu*dphi/dy) */
  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_Y, 1, 2, DMSTAG_ELEMENT, 0, DMSTAG_DOWN, 0, &fd_diff_inner_y));
  PetscCall(FlucaFDSetUp(fd_diff_inner_y));

  PetscCall(FlucaFDScaleCreateConstant(fd_diff_inner_y, ctx->mu, &fd_diff_scaled_y));
  PetscCall(FlucaFDSetUp(fd_diff_scaled_y));

  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_Y, 1, 2, DMSTAG_DOWN, 0, DMSTAG_ELEMENT, 0, &fd_diff_outer_y));
  PetscCall(FlucaFDSetUp(fd_diff_outer_y));

  PetscCall(FlucaFDCompositionCreate(fd_diff_scaled_y, fd_diff_outer_y, &fd_diff_y));
  PetscCall(FlucaFDSetUp(fd_diff_y));

  /* Combined RHS: -conv_x - conv_y + diff_x + diff_y */
  operands[0] = fd_neg_conv_x;
  operands[1] = fd_neg_conv_y;
  operands[2] = fd_diff_x;
  operands[3] = fd_diff_y;
  PetscCall(FlucaFDSumCreate(4, operands, fd_rhs));
  PetscCall(FlucaFDSetUp(*fd_rhs));

  /* Cleanup intermediate operators */
  PetscCall(FlucaFDDestroy(&fd_diff_y));
  PetscCall(FlucaFDDestroy(&fd_diff_outer_y));
  PetscCall(FlucaFDDestroy(&fd_diff_scaled_y));
  PetscCall(FlucaFDDestroy(&fd_diff_inner_y));
  PetscCall(FlucaFDDestroy(&fd_diff_x));
  PetscCall(FlucaFDDestroy(&fd_diff_outer_x));
  PetscCall(FlucaFDDestroy(&fd_diff_scaled_x));
  PetscCall(FlucaFDDestroy(&fd_diff_inner_x));
  PetscCall(FlucaFDDestroy(&fd_neg_conv_y));
  PetscCall(FlucaFDDestroy(&fd_conv_y));
  PetscCall(FlucaFDDestroy(&fd_conv_deriv_y));
  PetscCall(FlucaFDDestroy(&fd_neg_conv_x));
  PetscCall(FlucaFDDestroy(&fd_conv_x));
  PetscCall(FlucaFDDestroy(&fd_conv_deriv_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSFunction(TS ts, PetscReal t, Vec u, Vec F, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;

  PetscFunctionBegin;
  /* Update TVD operators with current solution and velocity */
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd_x, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd_x, u));
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd_y, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd_y, u));

  /* RHS = operator(u) (operator already has correct signs) */
  PetscCall(FlucaFDApply(ctx->fd, ctx->dm, ctx->dm, u, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSJacobian(TS ts, PetscReal t, Vec u, Mat A, Mat P, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd_x, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd_x, u));
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd_y, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd_y, u));

  PetscCall(MatZeroEntries(A));
  PetscCall(FlucaFDGetOperator(ctx->fd, ctx->dm, ctx->dm, A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInitialCondition(DM dm, Vec u)
{
  Vec                 u_local;
  PetscInt            xs, ys, m, n, i, j, slot, slot_coord;
  PetscScalar      ***arr;
  PetscScalar         xc, yc;
  const PetscScalar **arr_coord_x, **arr_coord_y;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &u_local));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot));
  PetscCall(DMStagVecGetArray(dm, u_local, &arr));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord_x, &arr_coord_y, NULL));

  /* Square wave: phi = 1 inside [0.25,0.5]x[0.25,0.5], else phi = 0 */
  for (j = ys; j < ys + n; ++j) {
    for (i = xs; i < xs + m; ++i) {
      xc = arr_coord_x[i][slot_coord];
      yc = arr_coord_y[j][slot_coord];
      if (0.25 <= xc && xc <= 0.5 && 0.25 <= yc && yc <= 0.5) {
        arr[j][i][slot] = 1.;
      } else {
        arr[j][i][slot] = 0.;
      }
    }
  }

  PetscCall(DMStagVecRestoreArray(dm, u_local, &arr));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord_x, &arr_coord_y, NULL));
  PetscCall(DMLocalToGlobal(dm, u_local, INSERT_VALUES, u));
  PetscCall(DMRestoreLocalVector(dm, &u_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorSplitToDMDA(TS ts, PetscInt step, PetscReal time, Vec u, PetscViewerAndFormat *vf)
{
  DM  dm, da;
  Vec davec;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (vf->view_interval > 0 && step % vf->view_interval) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMStagVecSplitToDMDA(dm, u, DMSTAG_ELEMENT, 0, &da, &davec));
  PetscCall(PetscViewerPushFormat(vf->viewer, vf->format));
  PetscCall(VecView(davec, vf->viewer));
  PetscCall(PetscViewerPopFormat(vf->viewer));
  PetscCall(VecDestroy(&davec));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx ctx;
  TS     ts;
  Mat    A;
  Vec    u;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Options */
  ctx.mu = 0.01;
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-mu", &ctx.mu, NULL));

  /* Create DM with periodic boundary */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, 64, 64, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &ctx.dm));
  PetscCall(DMSetFromOptions(ctx.dm));
  PetscCall(DMSetUp(ctx.dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(ctx.dm, 0., 1., 0., 1., 0., 0.));

  /* Create solution vector and matrix */
  PetscCall(DMCreateGlobalVector(ctx.dm, &u));
  PetscCall(DMCreateMatrix(ctx.dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  /* Create velocity and operators */
  PetscCall(CreateVelocityVector(ctx.dm, &ctx.dm_vel, &ctx.vel));
  PetscCall(CreateOperator(&ctx, &ctx.fd));

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
  PetscCall(TSMonitorSetFromOptions(ts, "-ts_monitor_dmda", "View solution on DMDA", "ex3.c", MonitorSplitToDMDA, NULL));
  PetscCall(TSSetFromOptions(ts));

  /* Solve */
  PetscCall(TSSolve(ts, u));

  /* Cleanup */
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(FlucaFDDestroy(&ctx.fd));
  PetscCall(FlucaFDDestroy(&ctx.fd_tvd_y));
  PetscCall(FlucaFDDestroy(&ctx.fd_tvd_x));
  PetscCall(VecDestroy(&ctx.vel));
  PetscCall(DMDestroy(&ctx.dm_vel));
  PetscCall(DMDestroy(&ctx.dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: pure_convection_superbee
    nsize: 1
    args: -mu 0 -flucafd_limiter superbee -ts_type ssp -ts_dt 0.01

  test:
    suffix: pure_convection_upwind
    nsize: 1
    args: -mu 0 -flucafd_limiter upwind -ts_type ssp -ts_dt 0.01

  test:
    suffix: pure_convection_quick
    nsize: 1
    args: -mu 0 -flucafd_limiter quick -ts_type ssp -ts_dt 0.01

  test:
    suffix: superbee
    nsize: 1
    args: -mu 0.01 -flucafd_limiter superbee -ts_type ssp -ts_dt 0.01

  test:
    suffix: upwind
    nsize: 1
    args: -mu 0.01 -flucafd_limiter upwind -ts_type ssp -ts_dt 0.01

  test:
    suffix: quick
    nsize: 1
    args: -mu 0.01 -flucafd_limiter quick -ts_type ssp -ts_dt 0.01

TEST*/
