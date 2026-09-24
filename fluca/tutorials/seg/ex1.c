#include <flucaphys.h>
#include <flucaseg.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "2D Taylor-Green vortex with SEGCNLINEAR\n"
                           "Exact solution on periodic [0, 2*pi]^2:\n"
                           "  u = -cos(x)*sin(y)*exp(-2*nu*t)\n"
                           "  v =  sin(x)*cos(y)*exp(-2*nu*t)\n"
                           "  p = -(cos(2x)+cos(2y))/4 * exp(-4*nu*t)\n"
                           "The solution holds p^{n+1/2}, so pressure is compared with the exact pressure at t - dt/2.\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : Grid cells per direction (default: 32)\n"
                           "  -rho <real> : Density (default: 1.0)\n"
                           "  -mu <real>  : Dynamic viscosity (default: 0.01)\n"
                           "  -repeat_solve : Refill Y with the same initial condition and SegSolve again over the same\n"
                           "                  interval, checking that the errors reproduce exactly (default: false)\n";

/* Fill cell velocity at t_vel and cell pressure at t_p with the exact TGV; face velocity is left for SEGCNLINEAR to project */
static PetscErrorCode FillExactSolution(Phys phys, PetscReal nu, PetscReal t_vel, PetscReal t_p, Vec Y)
{
  DM                  sol_dm;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            c_vel, c_p, xs, ys, xm, ym, slot_elem, i, j;
  PetscReal           decay_v, decay_p;

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  decay_v = PetscExpReal(-2. * nu * t_vel);
  decay_p = PetscExpReal(-4. * nu * t_p);
  PetscCall(VecZeroEntries(Y));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil st[3];
      PetscInt      c;

      vals[0] = -PetscCosReal(x) * PetscSinReal(y) * decay_v;
      vals[1] = PetscSinReal(x) * PetscCosReal(y) * decay_v;
      vals[2] = -0.25 * (PetscCosReal(2. * x) + PetscCosReal(2. * y)) * decay_p;
      for (c = 0; c < 3; c++) {
        st[c].i   = i;
        st[c].j   = j;
        st[c].k   = 0;
        st[c].loc = DMSTAG_ELEMENT;
        st[c].c   = c < 2 ? c_vel + c : c_p;
      }
      PetscCall(DMStagVecSetValuesStencil(sol_dm, Y, 3, st, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Cell-centered L2 errors of u, v at t and of the mean-free p against the exact pressure at t - dt/2 */
static PetscErrorCode ComputeL2Error(Phys phys, PetscReal nu, PetscReal t, PetscReal dt, Vec Y, PetscReal err[3])
{
  DM            sol_dm;
  Vec           Y_exact, diff, sub;
  DMStagStencil st;
  IS            is;
  PetscInt      c_vel, c_p, N, c, np;
  PetscScalar   mean;
  PetscReal     h;

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(DMCreateGlobalVector(sol_dm, &Y_exact));
  PetscCall(FillExactSolution(phys, nu, t, t - dt / 2., Y_exact));
  PetscCall(DMCreateGlobalVector(sol_dm, &diff));
  PetscCall(VecWAXPY(diff, -1., Y_exact, Y));
  PetscCall(DMStagGetGlobalSizes(sol_dm, &N, NULL, NULL));
  h = 2. * PETSC_PI / N;
  for (c = 0; c < 3; c++) {
    st.i   = 0;
    st.j   = 0;
    st.k   = 0;
    st.loc = DMSTAG_ELEMENT;
    st.c   = c < 2 ? c_vel + c : c_p;
    PetscCall(DMStagCreateISFromStencils(sol_dm, 1, &st, &is));
    PetscCall(VecGetSubVector(diff, is, &sub));
    if (c == 2) {
      /* Pressure is determined up to a constant */
      PetscCall(VecGetSize(sub, &np));
      PetscCall(VecSum(sub, &mean));
      PetscCall(VecShift(sub, -mean / np));
    }
    PetscCall(VecNorm(sub, NORM_2, &err[c]));
    err[c] *= h;
    PetscCall(VecRestoreSubVector(diff, is, &sub));
    PetscCall(ISDestroy(&is));
  }
  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&Y_exact));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  Seg       seg;
  Vec       Y;
  PetscReal rho = 1., mu = 0.01, nu, t_final, dt, dt0, err[3], err_repeat[3];
  PetscBool repeat_solve = PETSC_FALSE;
  PetscInt  c;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho", &rho, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &mu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-repeat_solve", &repeat_solve, NULL));
  nu = mu / rho;

  /* Base DM: grid topology and coordinates; stencil width 4 is required by the Seg spatial operators */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, 32, 32, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysSetDensity(phys, rho));
  PetscCall(PhysSetViscosity(phys, mu));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetType(seg, SEGCNLINEAR));
  PetscCall(SegSetPhys(seg, phys));
  PetscCall(SegSetMaxTime(seg, 1.));
  PetscCall(SegSetTimeStep(seg, 0.01));
  PetscCall(SegSetFromOptions(seg));
  PetscCall(SegSetUp(seg));
  PetscCall(SegGetTimeStep(seg, &dt0));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));

  /* Initial condition: exact TGV at t = 0; the initial pressure is q at the first step */
  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(FillExactSolution(phys, nu, 0., 0., Y));

  PetscCall(SegSolve(seg, Y));
  PetscCall(SegGetTime(seg, &t_final));
  PetscCall(SegGetTimeStep(seg, &dt));
  PetscCall(ComputeL2Error(phys, nu, t_final, dt, Y, err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "t = %.4f, L2 errors: u = %.4e, v = %.4e, p(t - dt/2) = %.4e\n", (double)t_final, (double)err[0], (double)err[1], (double)err[2]));
  /* Velocity errors only: pressure error is spatial-discretization dominated. Measured values are ~8e-5 at
     16x16, so this bound leaves roughly one order of margin and catches breakage without being a golden-value check. */
  PetscCheck(err[0] < 1.e-3 && err[1] < 1.e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Velocity L2 errors %g, %g exceed the expected bound for this grid", (double)err[0], (double)err[1]);

  if (repeat_solve) {
    /* Reuse the same Seg and the same Vec Y for a second, identical solve. SegSolve() projects the initial
       face velocity once per call, so this run must start from the same discretely divergence-free state as
       the first one; if it did not, it would integrate against the stale face velocity from the first solve
       and its errors would diverge from the first solve's. */
    PetscCall(SegSetTime(seg, 0.));
    PetscCall(SegSetStepNumber(seg, 0));
    PetscCall(SegSetTimeStep(seg, dt0));
    PetscCall(SegSetConvergedReason(seg, SEG_CONVERGED_ITERATING));
    PetscCall(FillExactSolution(phys, nu, 0., 0., Y));

    PetscCall(SegSolve(seg, Y));
    PetscCall(SegGetTime(seg, &t_final));
    PetscCall(SegGetTimeStep(seg, &dt));
    PetscCall(ComputeL2Error(phys, nu, t_final, dt, Y, err_repeat));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "repeat: t = %.4f, L2 errors: u = %.4e, v = %.4e, p(t - dt/2) = %.4e\n", (double)t_final, (double)err_repeat[0], (double)err_repeat[1], (double)err_repeat[2]));
    for (c = 0; c < 3; ++c) PetscCheck(PetscAbsReal(err_repeat[c] - err[c]) < 1.e-12, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Repeated solve error[%" PetscInt_FMT "] %g differs from the first solve's %g", c, (double)err_repeat[c], (double)err[c]);
  }

  PetscCall(VecDestroy(&Y));
  PetscCall(SegDestroy(&seg));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: cnlinear
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -seg_max_time 0.1 -seg_dt 0.01 -seg_ksp_max_it 1 -seg_ksp_convergence_test skip

  test:
    suffix: cnlinear_repeat
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -seg_max_time 0.1 -seg_dt 0.01 -seg_ksp_max_it 1 -seg_ksp_convergence_test skip -repeat_solve

  test:
    suffix: cnlinear_converged
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -seg_max_time 0.1 -seg_dt 0.01 -seg_ksp_rtol 1e-10 -seg_abf_momentum_ksp_type preonly -seg_abf_momentum_pc_type lu -seg_abf_schur_ksp_type preonly -seg_abf_schur_pc_type lu -seg_abf_schur_pc_factor_shift_type nonzero

TEST*/
