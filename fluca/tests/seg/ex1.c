#include "segtest.h"
#include <flucaseg.h>

static const char help[] = "Test one PCABF sweep on the coupled system (13) against the theory guide's error (17)\n"
                           "With A1 = A2 = I, the Rhie-Chow and continuity rows are solved exactly and\n"
                           "the momentum residual equals (A - I) G p'.\n";

/* Lid-driven cavity: u = 1 on the top wall */
static PetscErrorCode LidVelocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = (comp == 0 && x[1] > 1. - 1e-12) ? 1. : 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                dm, sol_dm;
  Phys              phys;
  Mat               M, P, A, G, blocks[9];
  Vec               X, f, x, r, E, nullvec, sub, xp, w, Aw, Ev;
  IS                is[3];
  MatNullSpace      nullspace;
  KSP               ksp;
  PC                pc;
  PhysFieldLocation loc;
  PetscInt          c_vel, c_p, c_U, Nx, Ny, xs, ys, xm, ym, nx, ny, i, j, k, n, N;
  PetscReal         dt       = 0.05, h, nrm;
  const char       *names[3] = {PHYS_FIELD_VELOCITY, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_PRESSURE};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  /* PCABF is registered by SegInitializePackage; this test uses only Phys, which never triggers it */
  PetscCall(SegInitializePackage());
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(PhysTestCreateLaminar(dm, 1., 0.1, LidVelocity, &phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, &loc, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, &loc, &c_p, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, &loc, &c_U, NULL));
  PetscCall(DMStagGetGlobalSizes(sol_dm, &Nx, &Ny, NULL));
  h = 1. / Nx;

  /* A nontrivial state: smooth u^n, uniform U^n, linear q */
  PetscCall(DMCreateGlobalVector(sol_dm, &X));
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
      if (i < Nx && j < Ny) {
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
      if (j < Ny) {
        st.loc = DMSTAG_LEFT;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      }
      if (i < Nx) {
        st.loc = DMSTAG_DOWN;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(PhysTestCreateSystem(phys, &M, &f));
  PetscCall(PhysComputeMomentumSystem(phys, 0., dt, X, M, f));
  PetscCall(PhysComputeCouplingSystem(phys, dt, dt, M, f));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  /* Constant pressure null space; PCABF forwards it to the Schur complement */
  for (k = 0; k < 3; ++k) PetscCall(PhysGetFieldIS(phys, names[k], &is[k]));
  PetscCall(DMCreateGlobalVector(sol_dm, &nullvec));
  PetscCall(VecZeroEntries(nullvec));
  PetscCall(VecGetSubVector(nullvec, is[2], &sub));
  PetscCall(VecGetSize(sub, &N));
  PetscCall(VecSet(sub, 1. / PetscSqrtReal((PetscReal)N)));
  PetscCall(VecRestoreSubVector(nullvec, is[2], &sub));
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nullvec, &nullspace));
  PetscCall(MatSetNullSpace(M, nullspace));

  /* MATNEST carrying the field index sets that PCABF reads (its values come from M) */
  for (k = 0; k < 9; ++k) blocks[k] = NULL;
  for (k = 0; k < 3; ++k) {
    PetscCall(ISGetLocalSize(is[k], &n));
    PetscCall(ISGetSize(is[k], &N));
    PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, n, n, N, N, 1., &blocks[4 * k]));
  }
  PetscCall(MatCreateNest(PETSC_COMM_WORLD, 3, is, 3, is, blocks, &P));
  for (k = 0; k < 3; ++k) PetscCall(MatDestroy(&blocks[4 * k]));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, M, P));
  PetscCall(KSPSetType(ksp, KSPRICHARDSON));
  PetscCall(KSPSetTolerances(ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 1));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCABF));
  PetscCall(PCABFSetFields(pc, 0, 1, 2));
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

  PetscCall(PhysTestCheckField(phys, PHYS_FIELD_VELOCITY, r, E, 1e-10));
  PetscCall(PhysTestCheckField(phys, PHYS_FIELD_FACE_VELOCITY, r, E, 1e-10));
  PetscCall(PhysTestCheckField(phys, PHYS_FIELD_PRESSURE, r, E, 1e-10));

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&Aw));
  PetscCall(VecDestroy(&w));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&P));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscCall(VecDestroy(&nullvec));
  for (k = 0; k < 3; ++k) PetscCall(ISDestroy(&is[k]));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&M));
  PetscCall(VecDestroy(&X));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: fsm_sweep
    nsize: 1
    args: -abf_momentum_ksp_type preonly -abf_momentum_pc_type lu -abf_schur_ksp_type preonly -abf_schur_pc_type lu -abf_schur_pc_factor_shift_type nonzero
    output_file: output/empty.out

TEST*/
