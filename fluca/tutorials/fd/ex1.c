#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscsnes.h>

static const char help[] = "Solve 1D steady convection-diffusion equation using TVD scheme\n"
                           "Equation: (d/dx)(rho*u*phi) = (d/dx)(gamma*(d/dx)phi)\n"
                           "Domain: [0,1], BC: phi(0)=1, phi(1)=0\n"
                           "Options:\n"
                           "  -rho <real>             : Density (default: 1.)\n"
                           "  -gamma <real>           : Diffusivity (default: 1.)\n"
                           "  -flucafd_limiter <type> : Flux limiter (minmod/vanleer/superbee/mc/vanalbada)\n";

typedef struct {
  DM          dm, dm_vel;
  PetscScalar rho, gamma;
  PetscScalar phi_left, phi_right;
  Vec         vel;
  FlucaFD     fd_tvd, fd;
} AppCtx;

static PetscErrorCode CreateVelocityVector(DM dm, DM *dm_vel, Vec *vel)
{
  Vec           vel_local;
  PetscInt      x, m, nExtrax, slot_left, i;
  PetscScalar **arr;

  PetscFunctionBegin;
  PetscCall(DMStagCreateCompatibleDMStag(dm, 1, 0, 0, 0, dm_vel));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm_vel, 0., 1., 0., 0., 0., 0.));

  PetscCall(DMCreateGlobalVector(*dm_vel, vel));
  PetscCall(DMGetLocalVector(*dm_vel, &vel_local));

  PetscCall(DMStagGetCorners(*dm_vel, &x, NULL, NULL, &m, NULL, NULL, &nExtrax, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(*dm_vel, DMSTAG_LEFT, 0, &slot_left));
  PetscCall(DMStagVecGetArray(*dm_vel, vel_local, &arr));

  for (i = x; i < x + m + nExtrax; ++i) arr[i][slot_left] = 1.;

  PetscCall(DMStagVecRestoreArray(*dm_vel, vel_local, &arr));
  PetscCall(DMLocalToGlobal(*dm_vel, vel_local, INSERT_VALUES, *vel));
  PetscCall(DMRestoreLocalVector(*dm_vel, &vel_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateConvectionDiffusionOperator(AppCtx *ctx, FlucaFD *fd_convdiff)
{
  DM                       cdm;
  FlucaFD                  fd_scaled_tvd, fd_conv_deriv, fd_conv;
  FlucaFD                  fd_diff_inner, fd_diff_scaled, fd_diff_outer, fd_diff, fd_neg_diff;
  FlucaFD                  operands[2];
  FlucaFDBoundaryCondition bcs[2];

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(ctx->dm, &cdm));

  bcs[0].type  = FLUCAFD_BC_DIRICHLET;
  bcs[0].value = ctx->phi_left;
  bcs[1].type  = FLUCAFD_BC_DIRICHLET;
  bcs[1].value = ctx->phi_right;

  /* Convection operator: d/dx(rho * u * phi) */
  PetscCall(FlucaFDSecondOrderTVDCreate(cdm, FLUCAFD_X, 0, 0, &ctx->fd_tvd));
  PetscCall(FlucaFDSetBoundaryConditions(ctx->fd_tvd, bcs));
  PetscCall(FlucaFDSetFromOptions(ctx->fd_tvd));
  PetscCall(FlucaFDSetUp(ctx->fd_tvd));

  PetscCall(FlucaFDScaleCreateConstant(ctx->fd_tvd, ctx->rho, &fd_scaled_tvd));
  PetscCall(FlucaFDSetUp(fd_scaled_tvd));

  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_conv_deriv));
  PetscCall(FlucaFDSetUp(fd_conv_deriv));

  PetscCall(FlucaFDCompositionCreate(fd_scaled_tvd, fd_conv_deriv, &fd_conv));
  PetscCall(FlucaFDSetBoundaryConditions(fd_conv, bcs));
  PetscCall(FlucaFDSetUp(fd_conv));

  /* Negative diffusion operator: - d/dx(gamma * d/dx phi) */
  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_X, 1, 2, DMSTAG_ELEMENT, 0, DMSTAG_LEFT, 0, &fd_diff_inner));
  PetscCall(FlucaFDSetUp(fd_diff_inner));

  PetscCall(FlucaFDScaleCreateConstant(fd_diff_inner, ctx->gamma, &fd_diff_scaled));
  PetscCall(FlucaFDSetUp(fd_diff_scaled));

  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_diff_outer));
  PetscCall(FlucaFDSetUp(fd_diff_outer));

  PetscCall(FlucaFDCompositionCreate(fd_diff_scaled, fd_diff_outer, &fd_diff));
  PetscCall(FlucaFDSetUp(fd_diff));

  PetscCall(FlucaFDScaleCreateConstant(fd_diff, -1., &fd_neg_diff));
  PetscCall(FlucaFDSetBoundaryConditions(fd_neg_diff, bcs));
  PetscCall(FlucaFDSetUp(fd_neg_diff));

  /* Sum */
  operands[0] = fd_conv;
  operands[1] = fd_neg_diff;
  PetscCall(FlucaFDSumCreate(2, operands, fd_convdiff));
  PetscCall(FlucaFDSetBoundaryConditions(*fd_convdiff, bcs));
  PetscCall(FlucaFDSetUp(*fd_convdiff));

  /* Cleanup intermediate operators */
  PetscCall(FlucaFDDestroy(&fd_neg_diff));
  PetscCall(FlucaFDDestroy(&fd_diff));
  PetscCall(FlucaFDDestroy(&fd_diff_outer));
  PetscCall(FlucaFDDestroy(&fd_diff_scaled));
  PetscCall(FlucaFDDestroy(&fd_diff_inner));
  PetscCall(FlucaFDDestroy(&fd_conv));
  PetscCall(FlucaFDDestroy(&fd_conv_deriv));
  PetscCall(FlucaFDDestroy(&fd_scaled_tvd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFunction(SNES snes, Vec x, Vec b, void *ptr)
{
  AppCtx *ctx = (AppCtx *)ptr;
  Mat     A;

  PetscFunctionBegin;
  PetscCall(FlucaFDSecondOrderTVDSetVelocity(ctx->fd_tvd, ctx->vel, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ctx->fd_tvd, x));
  PetscCall(SNESGetJacobian(snes, &A, NULL, NULL, NULL));
  PetscCall(MatZeroEntries(A));
  PetscCall(VecZeroEntries(b));
  PetscCall(FlucaFDApply(ctx->fd, ctx->dm, ctx->dm, A, b));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(MatMultAdd(A, x, b, b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeJacobian(SNES snes, Vec x, Mat A, Mat P, void *ptr)
{
  PetscFunctionBegin;
  /* Jacobian is computed in ComputeFunction */
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx ctx;
  SNES   snes;
  Mat    A;
  Vec    x;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Options */
  ctx.rho   = 1.;
  ctx.gamma = 1.;
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-rho", &ctx.rho, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-gamma", &ctx.gamma, NULL));

  /* Create DM */
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 16, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, &ctx.dm));
  PetscCall(DMSetFromOptions(ctx.dm));
  PetscCall(DMSetUp(ctx.dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(ctx.dm, 0., 1., 0., 0., 0., 0.));

  /* Create solution vector and matrix */
  PetscCall(DMCreateGlobalVector(ctx.dm, &x));
  PetscCall(DMCreateMatrix(ctx.dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  /* Boundary conditions */
  ctx.phi_left  = 1.;
  ctx.phi_right = 0.;

  PetscCall(CreateVelocityVector(ctx.dm, &ctx.dm_vel, &ctx.vel));
  PetscCall(CreateConvectionDiffusionOperator(&ctx, &ctx.fd));

  /* Create SNES */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, ctx.dm));
  PetscCall(SNESSetFunction(snes, NULL, ComputeFunction, &ctx));
  PetscCall(SNESSetJacobian(snes, A, A, ComputeJacobian, &ctx));
  PetscCall(SNESSetFromOptions(snes));

  /* Set initial guess */
  PetscCall(VecZeroEntries(x));

  /* Solve */
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(VecViewFromOptions(x, NULL, "-phi_view"));

  /* Cleanup */
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(FlucaFDDestroy(&ctx.fd));
  PetscCall(FlucaFDDestroy(&ctx.fd_tvd));
  PetscCall(VecDestroy(&ctx.vel));
  PetscCall(DMDestroy(&ctx.dm_vel));
  PetscCall(DMDestroy(&ctx.dm));

  PetscCall(FlucaFinalize());
}
