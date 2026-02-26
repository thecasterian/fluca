#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

static const char help[] = "Manufactured solution test for steady INS solver\n"
                           "Exact solution: Taylor-Green-like\n"
                           "  u = -cos(pi*x)*sin(pi*y)\n"
                           "  v = sin(pi*x)*cos(pi*y)\n"
                           "  p = -(cos(2*pi*x) + cos(2*pi*y))/4\n"
                           "Stokes body force: f = -mu*nabla^2(u) + (1/rho)*grad(p) with mu=rho=1\n"
                           "  f_x = -2*pi^2*cos(pi*x)*sin(pi*y) + pi*sin(pi*x)*cos(pi*x)\n"
                           "  f_y =  2*pi^2*sin(pi*x)*cos(pi*y) + pi*sin(pi*y)*cos(pi*y)\n"
                           "Options:\n"
                           "  -stag_grid_x <int> : Grid size in x (default: 16)\n"
                           "  -stag_grid_y <int> : Grid size in y (default: 16)\n";

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

/* Stokes body force: f = -mu*nabla^2(u) + (1/rho)*grad(p) with mu=rho=1. */
static PetscErrorCode BodyForce(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx)
{
  PetscReal pi = PETSC_PI;
  PetscReal cx = PetscCosReal(pi * x[0]), sx = PetscSinReal(pi * x[0]);
  PetscReal cy = PetscCosReal(pi * x[1]), sy = PetscSinReal(pi * x[1]);

  PetscFunctionBeginUser;
  f[0] = -2.0 * pi * pi * cx * sy + pi * sx * cx;
  f[1] = 2.0 * pi * pi * sx * cy + pi * sy * cy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                  dm, sol_dm;
  Phys                phys;
  TS                  ts;
  Vec                 x;
  PetscInt            Nx, Ny, d, f;
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

  /* Create 2D base DMStag; grid size overridden via -stag_grid_x/-stag_grid_y */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 16, 16, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, NULL));
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

  /* Create solution vector, zero initial guess */
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMCreateGlobalVector(sol_dm, &x));
  PetscCall(VecZeroEntries(x));

  /* Solve via TSPSEUDO (steady state through pseudo-timestepping) */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(PhysSetUpTS(phys, ts));
  PetscCall(TSSetFromOptions(ts));

  /* Diagnostic: verify IFunction vanishes at the exact solution */
  {
    Vec                 x_exact, u_t, F_check;
    PetscReal           fnorm;
    const PetscScalar **arrc_e[2] = {NULL, NULL};
    PetscInt            slot_e, slot_dof[3];
    PetscInt            xs_e, ys_e, xm_e, ym_e;
    PetscInt            i_e, j_e, de;

    PetscCall(DMCreateGlobalVector(sol_dm, &x_exact));
    PetscCall(DMCreateGlobalVector(sol_dm, &u_t));
    PetscCall(DMCreateGlobalVector(sol_dm, &F_check));
    PetscCall(VecZeroEntries(u_t));

    PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_e));
    PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc_e[0], &arrc_e[1], NULL));
    PetscCall(DMStagGetCorners(sol_dm, &xs_e, &ys_e, NULL, &xm_e, &ym_e, NULL, NULL, NULL, NULL));
    for (de = 0; de < 3; de++) PetscCall(DMStagGetLocationSlot(sol_dm, DMSTAG_ELEMENT, de, &slot_dof[de]));

    for (j_e = ys_e; j_e < ys_e + ym_e; j_e++) {
      for (i_e = xs_e; i_e < xs_e + xm_e; i_e++) {
        PetscReal     coords_e[2];
        PetscScalar   exact_e[3];
        DMStagStencil st;

        coords_e[0] = PetscRealPart(arrc_e[0][i_e][slot_e]);
        coords_e[1] = PetscRealPart(arrc_e[1][j_e][slot_e]);
        PetscCall(ExactSolution(2, coords_e, exact_e));

        st.i   = i_e;
        st.j   = j_e;
        st.k   = 0;
        st.loc = DMSTAG_ELEMENT;
        for (de = 0; de < 3; de++) {
          st.c = de;
          PetscCall(DMStagVecSetValuesStencil(sol_dm, x_exact, 1, &st, &exact_e[de], INSERT_VALUES));
        }
      }
    }
    PetscCall(VecAssemblyBegin(x_exact));
    PetscCall(VecAssemblyEnd(x_exact));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc_e[0], &arrc_e[1], NULL));

    PetscCall(TSSetUp(ts));
    PetscCall(TSComputeIFunction(ts, 0.0, x_exact, u_t, F_check, PETSC_FALSE));
    PetscCall(VecNorm(F_check, NORM_INFINITY, &fnorm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "||IFunction(u_exact, 0)||_inf = %g\n", (double)fnorm));

    PetscCall(VecDestroy(&x_exact));
    PetscCall(VecDestroy(&u_t));
    PetscCall(VecDestroy(&F_check));
  }

  PetscCall(TSSolve(ts, x));

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

    N_cells = Nx * Ny;
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

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Grid: %" PetscInt_FMT " x %" PetscInt_FMT "\n", Nx, Ny));
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
    suffix: default
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -ts_type pseudo -ts_pseudo_fatol 1e-12 -ts_dt 1e-3 -ts_max_steps 50 -snes_type newtonls -ksp_type preonly -pc_type lu -pc_factor_shift_type nonzero -phys_ins_flucafd_limiter superbee

  test:
    suffix: refined
    nsize: 1
    args: -stag_grid_x 32 -stag_grid_y 32 -ts_type pseudo -ts_pseudo_fatol 1e-12 -ts_dt 1e-3 -ts_max_steps 50 -snes_type newtonls -ksp_type preonly -pc_type lu -pc_factor_shift_type nonzero -phys_ins_flucafd_limiter superbee

TEST*/
