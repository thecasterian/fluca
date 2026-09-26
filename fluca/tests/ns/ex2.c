#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test one PCABF sweep on the NS coupled system against the theory guide's error\n"
                           "With A1 = A2 = I, the Rhie-Chow and continuity rows are solved exactly and\n"
                           "the momentum residual equals (A - I) G p'.\n";

/* Lid-driven cavity: u = 1 on the top wall */
static PetscErrorCode LidVelocity_Private(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = comp == 0 ? 1. : 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fail unless max |a - b| over the entries in is is within tol */
static PetscErrorCode CheckField_Private(IS is, const char name[], Vec a, Vec b, PetscReal tol)
{
  Vec       sa, sb, diff;
  PetscReal err;

  PetscFunctionBeginUser;
  PetscCall(VecGetSubVector(a, is, &sa));
  PetscCall(VecGetSubVector(b, is, &sb));
  PetscCall(VecDuplicate(sa, &diff));
  PetscCall(VecWAXPY(diff, -1., sb, sa));
  PetscCall(VecNorm(diff, NORM_INFINITY, &err));
  PetscCall(VecDestroy(&diff));
  PetscCall(VecRestoreSubVector(b, is, &sb));
  PetscCall(VecRestoreSubVector(a, is, &sa));
  PetscCheck(err <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Field %s: max error %g exceeds tolerance %g", name, (double)err, (double)tol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM           dm, sol_dm;
  Phys         phys;
  NS           ns;
  SNES         snes;
  Mat          J, M, A, G;
  Vec          X, f, x, r, E, xp, w, Aw, Ev;
  IS           is[3];
  MatNullSpace nullspace;
  KSP          ksp;
  PC           pc;
  PhysBC       wall = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL}, lid = {PHYS_BC_VELOCITY, LidVelocity_Private, NULL, NULL, NULL};
  PetscInt     c_vel, c_p, c_U, xs, ys, xm, ym, nx, ny, i, j, k;
  PetscReal    h        = 1. / 8., nrm;
  const char  *names[3] = {PHYS_FIELD_VELOCITY, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_PRESSURE};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysSetViscosity(phys, 0.1));
  for (k = 0; k < 3; ++k) PetscCall(PhysSetBoundaryCondition(phys, k, wall));
  PetscCall(PhysSetBoundaryCondition(phys, 3, lid));
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetTimeStepSize(ns, 0.05));
  PetscCall(NSSetUp(ns));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  for (k = 0; k < 3; ++k) PetscCall(NSGetField(ns, names[k], &is[k]));

  /* A nontrivial state: smooth u^n, uniform U^n, linear q */
  PetscCall(NSGetSolution(ns, &X));
  PetscCall(VecZeroEntries(X));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, &nx, &ny, NULL));
  for (j = ys; j < ys + ym + ny; ++j) {
    for (i = xs; i < xs + xm + nx; ++i) {
      PetscReal     xc = (i + 0.5) * h, yc = (j + 0.5) * h;
      DMStagStencil st;
      PetscScalar   v;

      st.i = i;
      st.j = j;
      st.k = 0;
      if (i < xs + xm && j < ys + ym) {
        st.loc = DMSTAG_ELEMENT;
        st.c   = c_vel;
        v      = PetscSinReal(PETSC_PI * xc) * PetscCosReal(PETSC_PI * yc);
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
        st.c = c_vel + 1;
        v    = -PetscCosReal(PETSC_PI * xc) * PetscSinReal(PETSC_PI * yc);
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
        st.c = c_p;
        v    = xc + 2. * yc;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      }
      st.c = c_U;
      v    = 0.3;
      if (j < ys + ym) {
        st.loc = DMSTAG_LEFT;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      }
      if (i < xs + xm) {
        st.loc = DMSTAG_DOWN;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  /* One step linearizes the operators at X; M and f are then the system of that step */
  PetscCall(NSStep(ns));
  PetscCall(DMCreateMatrix(sol_dm, &M));
  PetscCall(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(NSFormJacobian(ns, NULL, M));
  PetscCall(DMCreateGlobalVector(sol_dm, &f));
  PetscCall(NSFormFunction(ns, NULL, f));
  PetscCall(NSGetSNES(ns, &snes));
  PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
  PetscCall(MatGetNullSpace(J, &nullspace));
  PetscCall(MatSetNullSpace(M, nullspace));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetType(ksp, KSPRICHARDSON));
  PetscCall(KSPSetTolerances(ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 1));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCABF));
  PetscCall(PCABFSetFieldIS(pc, is[0], is[1], is[2]));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(DMCreateGlobalVector(sol_dm, &x));
  PetscCall(VecZeroEntries(x));
  PetscCall(KSPSolve(ksp, f, x));

  /* r = f - M x */
  PetscCall(DMCreateGlobalVector(sol_dm, &r));
  PetscCall(MatMult(M, x, r));
  PetscCall(VecAYPX(r, -1., f));

  /* E: (A - I) G p' in the velocity rows, zero elsewhere */
  PetscCall(MatCreateSubMatrix(M, is[0], is[0], MAT_INITIAL_MATRIX, &A));
  PetscCall(MatCreateSubMatrix(M, is[0], is[2], MAT_INITIAL_MATRIX, &G));
  PetscCall(MatCreateVecs(G, NULL, &w));
  PetscCall(VecDuplicate(w, &Aw));
  PetscCall(VecGetSubVector(x, is[2], &xp));
  PetscCall(MatMult(G, xp, w));
  PetscCall(VecRestoreSubVector(x, is[2], &xp));
  PetscCall(MatMult(A, w, Aw));
  PetscCall(VecAXPY(Aw, -1., w));
  PetscCall(VecNorm(Aw, NORM_INFINITY, &nrm));
  PetscCheck(nrm > 1e-6, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Test is degenerate: (A - I) G p' vanishes");
  PetscCall(DMCreateGlobalVector(sol_dm, &E));
  PetscCall(VecZeroEntries(E));
  PetscCall(VecGetSubVector(E, is[0], &Ev));
  PetscCall(VecCopy(Aw, Ev));
  PetscCall(VecRestoreSubVector(E, is[0], &Ev));

  for (k = 0; k < 3; ++k) PetscCall(CheckField_Private(is[k], names[k], r, E, 1e-10));

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&Aw));
  PetscCall(VecDestroy(&w));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&M));
  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: abf_sweep
    nsize: 1
    args: -abf_momentum_ksp_type preonly -abf_momentum_pc_type lu -abf_schur_ksp_type preonly -abf_schur_pc_type lu -abf_schur_pc_factor_shift_type nonzero
    output_file: output/empty.out

TEST*/
