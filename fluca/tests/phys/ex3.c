#include "phystest.h"

static const char help[] = "Test the coupling blocks of PhysComputeCouplingSystem on a walled 2D grid\n"
                           "S = D((-T) G - (-R)) must equal -(dt/rho) times the compact Neumann Laplacian,\n"
                           "the Schur complement of the fractional step method (theory guide eq. (18)).\n";

static PetscScalar PressureValue(PetscInt i, PetscInt j)
{
  return (PetscScalar)((i + 1) * (i + 1) + 3 * (j + 1) * (i + 2));
}

int main(int argc, char **argv)
{
  DM                dm, sol_dm;
  Phys              phys;
  Mat               M, negT, G, negR, D, W, S;
  Vec               X, f, P, SP, E, Psub, SPsub;
  IS                is_v, is_U, is_p;
  PhysFieldLocation loc;
  PetscInt          c_p, Nx, Ny, xs, ys, xm, ym, i, j;
  PetscReal         rho = 2., mu = 0.5, dt = 0.1, hx, hy;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 6, 5, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(PhysTestCreateINS(dm, rho, mu, NULL, &phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, &loc, &c_p, NULL));
  PetscCall(DMStagGetGlobalSizes(sol_dm, &Nx, &Ny, NULL));
  hx = 1. / Nx;
  hy = 1. / Ny;

  PetscCall(PhysTestCreateSystem(phys, &M, &f));
  PetscCall(DMCreateGlobalVector(sol_dm, &X));
  PetscCall(VecZeroEntries(X));
  PetscCall(PhysComputeMomentumSystem(phys, 0., dt, X, M, f));
  PetscCall(PhysComputeCouplingSystem(phys, dt, dt, M, f));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_VELOCITY, &is_v));
  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_FACE_VELOCITY, &is_U));
  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_PRESSURE, &is_p));
  PetscCall(MatCreateSubMatrix(M, is_U, is_v, MAT_INITIAL_MATRIX, &negT));
  PetscCall(MatCreateSubMatrix(M, is_v, is_p, MAT_INITIAL_MATRIX, &G));
  PetscCall(MatCreateSubMatrix(M, is_U, is_p, MAT_INITIAL_MATRIX, &negR));
  PetscCall(MatCreateSubMatrix(M, is_p, is_U, MAT_INITIAL_MATRIX, &D));
  PetscCall(MatMatMult(negT, G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &W));
  PetscCall(MatAXPY(W, -1., negR, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatMatMult(D, W, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &S));

  /* P: arbitrary pressure; E: -(dt/rho) * compact Laplacian with homogeneous Neumann walls */
  PetscCall(DMCreateGlobalVector(sol_dm, &P));
  PetscCall(DMCreateGlobalVector(sol_dm, &SP));
  PetscCall(DMCreateGlobalVector(sol_dm, &E));
  PetscCall(VecZeroEntries(P));
  PetscCall(VecZeroEntries(SP));
  PetscCall(VecZeroEntries(E));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  for (j = ys; j < ys + ym; ++j) {
    for (i = xs; i < xs + xm; ++i) {
      DMStagStencil st;
      PetscScalar   v, fxp, fxm, fyp, fym;

      st.i   = i;
      st.j   = j;
      st.k   = 0;
      st.loc = DMSTAG_ELEMENT;
      st.c   = c_p;
      v      = PressureValue(i, j);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, P, 1, &st, &v, INSERT_VALUES));
      fxp = i + 1 < Nx ? (PressureValue(i + 1, j) - PressureValue(i, j)) / hx : 0.;
      fxm = i > 0 ? (PressureValue(i, j) - PressureValue(i - 1, j)) / hx : 0.;
      fyp = j + 1 < Ny ? (PressureValue(i, j + 1) - PressureValue(i, j)) / hy : 0.;
      fym = j > 0 ? (PressureValue(i, j) - PressureValue(i, j - 1)) / hy : 0.;
      v   = -(dt / rho) * ((fxp - fxm) / hx + (fyp - fym) / hy);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, E, 1, &st, &v, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(P));
  PetscCall(VecAssemblyEnd(P));
  PetscCall(VecAssemblyBegin(E));
  PetscCall(VecAssemblyEnd(E));

  PetscCall(VecGetSubVector(P, is_p, &Psub));
  PetscCall(VecGetSubVector(SP, is_p, &SPsub));
  PetscCall(MatMult(S, Psub, SPsub));
  PetscCall(VecRestoreSubVector(SP, is_p, &SPsub));
  PetscCall(VecRestoreSubVector(P, is_p, &Psub));
  PetscCall(PhysTestCheckField(phys, PHYS_FIELD_PRESSURE, SP, E, 1e-10));

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&SP));
  PetscCall(VecDestroy(&P));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&negR));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&negT));
  PetscCall(ISDestroy(&is_p));
  PetscCall(ISDestroy(&is_U));
  PetscCall(ISDestroy(&is_v));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&M));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: walled
    nsize: 1
    output_file: output/empty.out

TEST*/
