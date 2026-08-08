#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "2D Taylor-Green vortex with TSARKIMEX\n"
                           "Exact solution on periodic [0, 2*pi]^2:\n"
                           "  u = -cos(x)*sin(y)*exp(-2*nu*t)\n"
                           "  v =  sin(x)*cos(y)*exp(-2*nu*t)\n"
                           "  p = -(cos(2x)+cos(2y))/4 * exp(-4*nu*t)\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : Grid cells per direction (default: 32)\n"
                           "  -rho <real> : Density (default: 1.0)\n"
                           "  -mu <real>  : Dynamic viscosity (default: 0.01)\n"
                           "\n"
                           "The pressure is an algebraic (DAE) variable, so use a stiffly-accurate\n"
                           "integrator (e.g. -ts_arkimex_type 3, the default) and a saddle-point solver.\n"
                           "Phys INS sets up a velocity/pressure PCFIELDSPLIT Schur complement with full\n"
                           "factorization; every sub-solve is left to the options database.\n"
                           "Recommended solvers:\n"
                           "  monolithic direct : -ksp_type preonly -pc_type lu -pc_factor_shift_type nonzero\n"
                           "  Schur fieldsplit  : -fieldsplit_velocity_pc_type lu\n"
                           "                      -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_ksp_rtol 1e-10\n"
                           "  SIMPLE (A1 = A2 = diag A) : -pc_fieldsplit_schur_precondition selfp\n"
                           "                      -fieldsplit_pressure_mat_schur_complement_ainv_type diag\n"
                           "                      -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi\n"
                           "                      -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi\n"
                           "See THEORY_GUIDE.md.\n";

/* Fill solution vector with exact TGV at time t */
static PetscErrorCode FillExactSolution(DM sol_dm, PetscReal nu, PetscReal t, Vec Y)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem;
  PetscReal           decay_v, decay_p;

  PetscFunctionBegin;
  decay_v = PetscExpReal(-2. * nu * t);
  decay_p = PetscExpReal(-4. * nu * t);

  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));

  for (PetscInt j = ys; j < ys + ym; j++) {
    for (PetscInt i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil stencils[3];

      vals[0] = -PetscCosReal(x) * PetscSinReal(y) * decay_v;
      vals[1] = PetscSinReal(x) * PetscCosReal(y) * decay_v;
      vals[2] = -0.25 * (PetscCosReal(2. * x) + PetscCosReal(2. * y)) * decay_p;

      for (PetscInt c = 0; c < 3; c++) {
        stencils[c].i   = i;
        stencils[c].j   = j;
        stencils[c].k   = 0;
        stencils[c].loc = DMSTAG_ELEMENT;
        stencils[c].c   = c;
      }
      PetscCall(DMStagVecSetValuesStencil(sol_dm, Y, 3, stencils, vals, INSERT_VALUES));
    }
  }

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute L2 error of the numerical solution against the exact TGV at time t.
   Uses the cell-centered L2 norm: ||e||_2 = sqrt(h^2 * sum(e_i^2)) where h is the cell width. */
static PetscErrorCode ComputeL2Error(DM sol_dm, PetscReal nu, PetscReal t, Vec Y, PetscReal *err_u, PetscReal *err_v, PetscReal *err_p)
{
  Vec       Y_exact, diff;
  PetscReal h2, norm_u, norm_v, norm_p;
  PetscInt  N;

  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(sol_dm, &Y_exact));
  PetscCall(FillExactSolution(sol_dm, nu, t, Y_exact));

  /* diff = Y - Y_exact */
  PetscCall(VecDuplicate(Y, &diff));
  PetscCall(VecCopy(Y, diff));
  PetscCall(VecAXPY(diff, -1., Y_exact));

  /* h^2 for scaling (uniform grid on [0, 2*pi]^2) */
  PetscCall(DMStagGetGlobalSizes(sol_dm, &N, NULL, NULL));
  h2 = (2. * PETSC_PI / N) * (2. * PETSC_PI / N);

  /* Compute per-component L2 norms via IS decomposition.
     For the u-component, extract DOF 0; for v, DOF 1; for p, DOF 2. */
  {
    DMStagStencil u_s = {.loc = DMSTAG_ELEMENT, .c = 0};
    DMStagStencil v_s = {.loc = DMSTAG_ELEMENT, .c = 1};
    DMStagStencil p_s = {.loc = DMSTAG_ELEMENT, .c = 2};
    IS            is_u, is_v, is_p;
    Vec           sub;

    PetscCall(DMStagCreateISFromStencils(sol_dm, 1, &u_s, &is_u));
    PetscCall(DMStagCreateISFromStencils(sol_dm, 1, &v_s, &is_v));
    PetscCall(DMStagCreateISFromStencils(sol_dm, 1, &p_s, &is_p));

    PetscCall(VecGetSubVector(diff, is_u, &sub));
    PetscCall(VecNorm(sub, NORM_2, &norm_u));
    PetscCall(VecRestoreSubVector(diff, is_u, &sub));

    PetscCall(VecGetSubVector(diff, is_v, &sub));
    PetscCall(VecNorm(sub, NORM_2, &norm_v));
    PetscCall(VecRestoreSubVector(diff, is_v, &sub));

    /* Pressure is determined up to a constant in a periodic domain.
       Subtract the mean pressure error before computing the norm. */
    PetscCall(VecGetSubVector(diff, is_p, &sub));
    {
      PetscScalar mean;
      PetscInt    np;

      PetscCall(VecGetSize(sub, &np));
      PetscCall(VecSum(sub, &mean));
      mean /= np;
      PetscCall(VecShift(sub, -mean));
    }
    PetscCall(VecNorm(sub, NORM_2, &norm_p));
    PetscCall(VecRestoreSubVector(diff, is_p, &sub));

    PetscCall(ISDestroy(&is_u));
    PetscCall(ISDestroy(&is_v));
    PetscCall(ISDestroy(&is_p));
  }

  /* Scale by h to get continuous L2 norm approximation */
  *err_u = norm_u * PetscSqrtReal(h2);
  *err_v = norm_v * PetscSqrtReal(h2);
  *err_p = norm_p * PetscSqrtReal(h2);

  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&Y_exact));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  TS        ts;
  Vec       Y;
  PetscReal rho = 1., mu = 0.01, nu, t_final;
  PetscReal err_u, err_v, err_p;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Parse options */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho", &rho, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &mu, NULL));
  nu = mu / rho;

  /* Create 2D periodic base DMStag.
     The base DM defines grid topology and coordinates only.
     DOF layout is set by Phys during setup.
     Grid size is set via -stag_grid_x / -stag_grid_y through DMSetFromOptions. */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, 32, 32, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));

  /* Create and configure Phys INS.
     Periodic domain needs no explicit boundary conditions. */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysINSSetDensity(phys, rho));
  PetscCall(PhysINSSetViscosity(phys, mu));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  PetscCall(PhysGetSolutionDM(phys, &sol_dm));

  /* Create TS and wire Phys callbacks.
     PhysSetUpTS sets DM, IFunction, IJacobian, RHSFunction, defaults to TSARKIMEX,
     and configures a velocity/pressure PCFIELDSPLIT Schur complement. Every
     sub-solve comes from the options database. */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(PhysSetUpTS(phys, ts));
  /* Defaults; override with -ts_max_time, -ts_dt, -ts_adapt_type */
  PetscCall(TSSetMaxTime(ts, 1.));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetFromOptions(ts));

  /* Set initial condition: exact TGV at t=0 */
  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(FillExactSolution(sol_dm, nu, 0., Y));

  /* Solve */
  PetscCall(TSSolve(ts, Y));
  PetscCall(TSGetTime(ts, &t_final));

  /* Compute and report L2 errors */
  PetscCall(ComputeL2Error(sol_dm, nu, t_final, Y, &err_u, &err_v, &err_p));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "t = %.4f, L2 errors: u = %.4e, v = %.4e, p = %.4e\n", (double)t_final, (double)err_u, (double)err_v, (double)err_p));

  /* Cleanup in reverse creation order */
  PetscCall(VecDestroy(&Y));
  PetscCall(TSDestroy(&ts));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: lu
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -ts_max_time 0.1 -ts_dt 0.01 -ts_type arkimex -ts_arkimex_type 3 -ts_adapt_type none -ksp_type preonly -pc_type lu -pc_factor_shift_type nonzero

  test:
    suffix: schur
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -ts_max_time 0.1 -ts_dt 0.01 -ts_type arkimex -ts_arkimex_type 3 -ts_adapt_type none -fieldsplit_velocity_ksp_type preonly -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_type gmres -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

  # SIMPLE preconditioner: diag(A00)^-1 must replace A00^-1 in BOTH the Schur complement
  # (A1: selfp/ainv_type diag for the preconditioner, inner jacobi for the operator) and
  # the velocity correction (A2: upper jacobi). A1 = A2 is what keeps SIMPLE mass
  # preserving; approximating the Schur complement alone would leave A2 = A.
  test:
    suffix: simple
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -ts_max_time 0.1 -ts_dt 0.01 -ts_type arkimex -ts_arkimex_type 3 -ts_adapt_type none -pc_fieldsplit_schur_precondition selfp -fieldsplit_pressure_mat_schur_complement_ainv_type diag -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -fieldsplit_velocity_ksp_type preonly -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_type gmres -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

TEST*/
