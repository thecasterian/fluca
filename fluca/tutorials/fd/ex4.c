#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

static const char help[] = "Solve 1D viscous Burger's equation using TVD scheme\n"
                           "Equation: (d/dt)phi + (d/dx)(u*phi) = (d/dx)(nu*(d/dx)phi), where u = phi/2\n"
                           "Domain: [0,1] with periodic BC\n"
                           "IC: phi(x,0) = sin^2(4*pi*x) for 0.25 <= x <= 0.5, else 0\n"
                           "Options:\n"
                           "  -flucafd_limiter <type> : Flux limiter (minmod/vanleer/superbee/mc/vanalbada)\n"
                           "  -nu <real>              : Kinematic viscosity (default: 0)\n";

typedef struct {
  DM        dm, dm_vel;
  Vec       vel;
  FlucaFD   fd_vel, fd_tvd, fd_scale_vel, fd;
  PetscReal nu;
} AppCtx;

static PetscErrorCode CreateVelocityDM(DM dm, DM *dm_vel, Vec *vel)
{
  PetscFunctionBegin;
  PetscCall(DMStagCreateCompatibleDMStag(dm, 1, 0, 0, 0, dm_vel));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm_vel, 0., 1., 0., 0., 0., 0.));
  PetscCall(DMCreateGlobalVector(*dm_vel, vel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateVelocityOperator(AppCtx *ctx)
{
  FlucaFD fd_interp;

  PetscFunctionBegin;
  /* 0th-order derivative = interpolation from cell centers to faces */
  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 0, 2, DMSTAG_ELEMENT, 0, DMSTAG_LEFT, 0, &fd_interp));
  PetscCall(FlucaFDSetUp(fd_interp));

  /* Scale by 0.5 to get u = phi/2 */
  PetscCall(FlucaFDScaleCreateConstant(fd_interp, 0.5, &ctx->fd_vel));
  PetscCall(FlucaFDSetUp(ctx->fd_vel));

  PetscCall(FlucaFDDestroy(&fd_interp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateConvectionOperator(AppCtx *ctx, FlucaFD *fd_conv)
{
  FlucaFD fd_conv_deriv;

  PetscFunctionBegin;

  /* TVD interpolation operator: phi(ELEMENT) -> phi_TVD(LEFT) */
  PetscCall(FlucaFDSecondOrderTVDCreate(ctx->dm, FLUCAFD_X, 0, 0, &ctx->fd_tvd));
  PetscCall(FlucaFDSetFromOptions(ctx->fd_tvd));
  PetscCall(FlucaFDSetUp(ctx->fd_tvd));

  /* Scale TVD output by velocity: phi_TVD -> u * phi_TVD at faces */
  PetscCall(FlucaFDScaleCreateVector(ctx->fd_tvd, ctx->vel, 0, &ctx->fd_scale_vel));
  PetscCall(FlucaFDSetUp(ctx->fd_scale_vel));

  /* Derivative operator: d/dx (LEFT -> ELEMENT) */
  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_conv_deriv));
  PetscCall(FlucaFDSetUp(fd_conv_deriv));

  /* Compose: d/dx(u * phi_TVD) */
  PetscCall(FlucaFDCompositionCreate(ctx->fd_scale_vel, fd_conv_deriv, fd_conv));
  PetscCall(FlucaFDSetUp(*fd_conv));

  PetscCall(FlucaFDDestroy(&fd_conv_deriv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDiffusionOperator(AppCtx *ctx, FlucaFD *fd_diff)
{
  FlucaFD fd_inner, fd_scaled, fd_outer;

  PetscFunctionBegin;
  /* Inner derivative: d/dx phi (ELEMENT -> LEFT) */
  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 1, 2, DMSTAG_ELEMENT, 0, DMSTAG_LEFT, 0, &fd_inner));
  PetscCall(FlucaFDSetUp(fd_inner));

  /* Scale by nu */
  PetscCall(FlucaFDScaleCreateConstant(fd_inner, ctx->nu, &fd_scaled));
  PetscCall(FlucaFDSetUp(fd_scaled));

  /* Outer derivative: d/dx (LEFT -> ELEMENT) */
  PetscCall(FlucaFDDerivativeCreate(ctx->dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_outer));
  PetscCall(FlucaFDSetUp(fd_outer));

  /* Compose: d/dx(nu * d/dx phi) */
  PetscCall(FlucaFDCompositionCreate(fd_scaled, fd_outer, fd_diff));
  PetscCall(FlucaFDSetUp(*fd_diff));

  PetscCall(FlucaFDDestroy(&fd_outer));
  PetscCall(FlucaFDDestroy(&fd_scaled));
  PetscCall(FlucaFDDestroy(&fd_inner));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSFunction(TS ts, PetscReal t, Vec x, Vec F, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;

  PetscFunctionBegin;
  /* Compute velocity u = phi/2 at faces */
  PetscCall(FlucaFDApply(ctx->fd_vel, ctx->dm, ctx->dm_vel, x, ctx->vel));

  /* Update operators with current velocity and solution */
  PetscCall(FlucaFDScaleSetVector(ctx->fd_scale_vel, ctx->vel, DMSTAG_LEFT, 0));
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd, x));

  PetscCall(FlucaFDApply(ctx->fd, ctx->dm, ctx->dm, x, F));
  /* RHS = -(convection) + diffusion */
  PetscCall(VecScale(F, -1.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRHSJacobian(TS ts, PetscReal t, Vec x, Mat A, Mat P, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;

  PetscFunctionBegin;
  /* Compute velocity u = phi/2 at faces */
  PetscCall(FlucaFDApply(ctx->fd_vel, ctx->dm, ctx->dm_vel, x, ctx->vel));

  /* Update operators with current velocity and solution */
  PetscCall(FlucaFDScaleSetVector(ctx->fd_scale_vel, ctx->vel, DMSTAG_LEFT, 0));
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd, x));

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
  PetscScalar   xc, s;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &u_local));
  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot));
  PetscCall(DMStagVecGetArray(dm, u_local, &arr));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));

  /* IC: phi = sin^2(4*pi*x) for 0.25 <= x <= 0.5, else 0 */
  for (i = x; i < x + m; ++i) {
    xc = arr_coord[i][slot_coord];
    if (0.25 <= xc && xc <= 0.5) {
      s            = PetscSinReal(4. * PETSC_PI * xc);
      arr[i][slot] = s * s;
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
  AppCtx  ctx;
  TS      ts;
  Mat     A;
  Vec     u;
  FlucaFD fd_conv, fd_diff, fd_neg_diff;
  FlucaFD operands[2];

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create DM with periodic boundary */
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 128, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, &ctx.dm));
  PetscCall(DMSetFromOptions(ctx.dm));
  PetscCall(DMSetUp(ctx.dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(ctx.dm, 0., 1., 0., 0., 0., 0.));

  /* Create solution vector and matrix */
  PetscCall(DMCreateGlobalVector(ctx.dm, &u));
  PetscCall(DMCreateMatrix(ctx.dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  /* Read viscosity */
  ctx.nu = 0.001;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-nu", &ctx.nu, NULL));

  /* Create velocity DM and operator */
  PetscCall(CreateVelocityDM(ctx.dm, &ctx.dm_vel, &ctx.vel));
  PetscCall(CreateVelocityOperator(&ctx));

  /* Create convection operator: d/dx(u * phi) */
  PetscCall(CreateConvectionOperator(&ctx, &fd_conv));

  /* Create diffusion operator: (d/dx)(nu * (d/dx)phi) */
  PetscCall(CreateDiffusionOperator(&ctx, &fd_diff));

  /* Negate diffusion for summation: operator computes conv - diff,
     then ComputeRHSFunction negates to get -(conv) + diff = RHS */
  PetscCall(FlucaFDScaleCreateConstant(fd_diff, -1., &fd_neg_diff));
  PetscCall(FlucaFDSetUp(fd_neg_diff));

  /* Sum: d/dx(u*phi) - d/dx(nu * d/dx phi) */
  operands[0] = fd_conv;
  operands[1] = fd_neg_diff;
  PetscCall(FlucaFDSumCreate(2, operands, &ctx.fd));
  PetscCall(FlucaFDSetUp(ctx.fd));

  PetscCall(FlucaFDDestroy(&fd_neg_diff));
  PetscCall(FlucaFDDestroy(&fd_diff));
  PetscCall(FlucaFDDestroy(&fd_conv));

  /* Set initial condition */
  PetscCall(SetInitialCondition(ctx.dm, u));

  /* Create TS */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, ctx.dm));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, ComputeRHSFunction, &ctx));
  PetscCall(TSSetRHSJacobian(ts, A, A, ComputeRHSJacobian, &ctx));
  PetscCall(TSSetMaxTime(ts, 0.5));
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
  PetscCall(FlucaFDDestroy(&ctx.fd_scale_vel));
  PetscCall(FlucaFDDestroy(&ctx.fd_tvd));
  PetscCall(FlucaFDDestroy(&ctx.fd_vel));
  PetscCall(VecDestroy(&ctx.vel));
  PetscCall(DMDestroy(&ctx.dm_vel));
  PetscCall(DMDestroy(&ctx.dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: superbee
    nsize: 1
    args: -flucafd_limiter superbee -stag_grid_x 256 -ts_type ssp -ts_dt 0.001 -ts_max_time 0.3

  test:
    suffix: minmod
    nsize: 1
    args: -flucafd_limiter minmod -stag_grid_x 256 -ts_type ssp -ts_dt 0.001 -ts_max_time 0.3

  test:
    suffix: vanleer
    nsize: 1
    args: -flucafd_limiter vanleer -stag_grid_x 256 -ts_type ssp -ts_dt 0.001 -ts_max_time 0.3

TEST*/
