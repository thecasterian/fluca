#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscsnes.h>

static const char help[] = "Manufactured solution test for steady Stokes solver\n"
                           "Exact solution: Taylor-Green-like\n"
                           "  u = -cos(pi*x)*sin(pi*y)\n"
                           "  v = sin(pi*x)*cos(pi*y)\n"
                           "  p = -(cos(2*pi*x) + cos(2*pi*y))/4\n"
                           "Options:\n"
                           "  -N <int> : Grid size in each direction (default: 16)\n";

/* Exact solution */
static PetscErrorCode ExactSolution(PetscInt dim, const PetscReal x[], PetscScalar u[])
{
  PetscReal pi = PETSC_PI;

  PetscFunctionBeginUser;
  u[0] = -PetscCosReal(pi * x[0]) * PetscSinReal(pi * x[1]);
  u[1] = PetscSinReal(pi * x[0]) * PetscCosReal(pi * x[1]);
  u[2] = -(PetscCosReal(2.0 * pi * x[0]) + PetscCosReal(2.0 * pi * x[1])) / 4.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* BC callback: exact velocity at boundary */
static PetscErrorCode BCVelocity(PetscInt dim, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscScalar u[3];

  PetscFunctionBeginUser;
  PetscCall(ExactSolution(dim, x, u));
  *val = u[comp];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Body force: f = -mu*nabla^2(u) + (1/rho)*grad(p) with mu=rho=1 */
static PetscErrorCode BodyForce(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx)
{
  PetscReal pi = PETSC_PI;
  PetscReal cx = PetscCosReal(pi * x[0]), sx = PetscSinReal(pi * x[0]);
  PetscReal cy = PetscCosReal(pi * x[1]), sy = PetscSinReal(pi * x[1]);

  PetscFunctionBeginUser;
  /* -nabla^2(u) = -2*pi^2*cos(pi*x)*sin(pi*y), dp/dx = pi*sin(pi*x)*cos(pi*x) */
  f[0] = -2.0 * pi * pi * cx * sy + pi * sx * cx;
  /* -nabla^2(v) = 2*pi^2*sin(pi*x)*cos(pi*y), dp/dy = pi*sin(pi*y)*cos(pi*y) */
  f[1] = 2.0 * pi * pi * sx * cy + pi * sy * cy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                  dm, sol_dm;
  Phys                phys;
  SNES                snes;
  Vec                 x;
  PetscInt            N = 16, d, f;
  PhysINSBC           bc;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem, slot[3];
  PetscInt            i, j;
  PetscReal           l2_vel_err2 = 0.0, l2_p_err2 = 0.0;
  PetscReal           p_mean_h = 0.0, p_mean_exact = 0.0;
  PetscReal           l2_vel_err, l2_p_err;
  PetscInt            N_cells;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));

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

  /* Set velocity Dirichlet BCs on all 4 faces */
  bc.type = PHYS_INS_BC_VELOCITY;
  bc.fn   = BCVelocity;
  bc.ctx  = NULL;
  for (f = 0; f < 4; f++) PetscCall(PhysINSSetBoundaryCondition(phys, f, bc));

  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  /* Create SNES and set up solver */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(PhysSetUpSNES(phys, snes));
  PetscCall(SNESSetFromOptions(snes));

  /* Create solution vector, zero initial guess */
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMCreateGlobalVector(sol_dm, &x));
  PetscCall(VecZeroEntries(x));

  /* Solve */
  PetscCall(SNESSolve(snes, NULL, x));

  /* Compute L2 errors using a local vector for array access */
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
        PetscCall(ExactSolution(2, coords, exact));

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
        PetscCall(ExactSolution(2, coords, exact));

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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Velocity L2 error: %.4e\n", (double)l2_vel_err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Pressure L2 error: %.4e\n", (double)l2_p_err));

  /* Cleanup */
  PetscCall(VecDestroy(&x));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  test:
    suffix: stokes_default
    nsize: 1
    args: -N 16 -snes_type ksponly -ksp_type preonly -pc_type lu -pc_factor_shift_type nonzero

  test:
    suffix: stokes_refined
    nsize: 1
    args: -N 32 -snes_type ksponly -ksp_type preonly -pc_type lu -pc_factor_shift_type nonzero

TEST*/
