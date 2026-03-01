#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscts.h>

static const char help[] = "Manufactured solution test for steady INS solver\n"
                           "Exact solution: Taylor-Green-like\n"
                           "  u = -cos(pi*x)*sin(pi*y)\n"
                           "  v = sin(pi*x)*cos(pi*y)\n"
                           "  p = -(cos(2*pi*x) + cos(2*pi*y))/4\n"
                           "Full NS body force: f = (u.grad)u - mu*nabla^2(u) + (1/rho)*grad(p)\n"
                           "  Convection and pressure gradient cancel for this solution.\n"
                           "  f_x = -2*pi^2*cos(pi*x)*sin(pi*y)\n"
                           "  f_y =  2*pi^2*sin(pi*x)*cos(pi*y)\n"
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

/* Full NS body force: f = (u.grad)u - mu*nabla^2(u) + (1/rho)*grad(p) with mu=rho=1.
   For this Taylor-Green solution, the convection and pressure gradient cancel exactly. */
static PetscErrorCode BodyForce(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx)
{
  PetscReal pi = PETSC_PI;
  PetscReal cx = PetscCosReal(pi * x[0]), sx = PetscSinReal(pi * x[0]);
  PetscReal cy = PetscCosReal(pi * x[1]), sy = PetscSinReal(pi * x[1]);

  PetscFunctionBeginUser;
  f[0] = -2.0 * pi * pi * cx * sy;
  f[1] = 2.0 * pi * pi * sx * cy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  TS        ts;
  Vec       x;
  PetscInt  Nx, Ny, f;
  PhysINSBC bc;
  PetscReal l2_vel_err, l2_p_err;

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

  /* Compute L2 errors against exact solution */
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

    /* Build exact solution vector */
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
        PetscCall(ExactSolution(2, coords, exact));

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
