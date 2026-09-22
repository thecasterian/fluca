#include "phystest.h"

static const char help[] = "Test the coupling blocks of PhysComputeCouplingSystem on a walled 2D grid\n"
                           "S = D((-T) G - (-R)) must equal -(dt/rho) times the compact Neumann Laplacian,\n"
                           "the Schur complement of the fractional step method (theory guide eq. (18)).\n"
                           "Also checks, row by row: W = (-T) G - (-R) equals -(dt/rho) times the compact\n"
                           "face pressure gradient G^st at a boundary, a wall-adjacent, and a bulk face;\n"
                           "and that -R annihilates a pressure field that is quadratic in each direction,\n"
                           "at every face row including the wall-adjacent ones.\n";

static PetscScalar PressureValue(PetscInt i, PetscInt j)
{
  return (PetscScalar)((i + 1) * (i + 1) + 3 * (j + 1) * (i + 2) + 2 * (j + 1) * (j + 1));
}

/* W = (-T) G - (-R) at x-direction face (fi, fj) must equal -(dt/rho) G^st exactly: G^st is the
   compact two-point face derivative of P, zero at a boundary face (pinned to the BC row by
   MatZeroRowsLocal). WPlocal already holds (the local form of) W applied to the test pressure
   field P. */
static PetscErrorCode CheckWRow(DM sol_dm, Vec WPlocal, PetscInt Nx, PetscInt fi, PetscInt fj, PetscInt c_U, PetscReal hx, PetscReal dt, PetscReal rho, PetscReal tol)
{
  DMStagStencil st;
  PetscScalar   wval, expected;

  PetscFunctionBeginUser;
  st.i   = fi;
  st.j   = fj;
  st.k   = 0;
  st.loc = DMSTAG_LEFT;
  st.c   = c_U;
  PetscCall(DMStagVecGetValuesStencil(sol_dm, WPlocal, 1, &st, &wval));
  expected = (fi == 0 || fi == Nx) ? 0. : -(dt / rho) * (PressureValue(fi, fj) - PressureValue(fi - 1, fj)) / hx;
  PetscCheck(PetscAbsScalar(wval - expected) <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "W row (i=%" PetscInt_FMT ", j=%" PetscInt_FMT "): got %g, expected %g", fi, fj, (double)PetscRealPart(wval), (double)PetscRealPart(expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                dm, sol_dm;
  Phys              phys;
  Mat               M, negT, G, negR, D, W, S;
  Vec               X, f, P, SP, E, Psub, SPsub;
  Vec               WPfull, WPlocal, Psub2, WPsub;
  Vec               RPfull, RPsub;
  IS                is_v, is_U, is_p;
  PhysFieldLocation loc;
  PetscInt          c_p, c_U, Nx, Ny, xs, ys, xm, ym, i, j;
  PetscReal         rho = 2., mu = 0.5, dt = 0.1, hx, hy, scale, tol, nrm;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 6, 5, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(PhysTestCreateLaminar(dm, rho, mu, NULL, &phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, &loc, &c_p, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, &loc, &c_U, NULL));
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

  /* Row-level checks: W must equal -(dt/rho) G^st exactly at a boundary face (i=0, pinned by the
     BC row so both sides are zero), a wall-adjacent interior face (i=1), and a bulk interior face
     (i=3), all at j=2.

     -R is then checked on the whole face-velocity block at once: R = T G_c - G^st annihilates any
     pressure field that is quadratic in the face-normal direction, because the cell gradient G_c is
     exact on quadratics (one-sided next to a wall, central elsewhere), the interpolation T is exact
     on the resulting linear field, and G^st is the exact face derivative of a quadratic on a uniform
     grid. P below is quadratic in both i and j, so -R P must vanish in every face row, whether the
     row's face-normal direction is i (LEFT faces) or j (DOWN faces).

     Until the four-point interpolation replaced the two-point average in T, -R vanished *identically*
     at a wall-adjacent face: the average of the one-sided cell gradient at the wall cell and the
     central one at its neighbour telescoped to exactly G^st there. That cancellation was an accident
     of the two-point average and is gone; the wall-adjacent rows now carry the same O(h^2 d3p/dn3)
     Rhie-Chow correction that every interior row carries, which is why the assertion is now that -R
     has no part of lower order rather than that it is zero. See laminarops.c for why T had to change. */
  scale = (dt / rho) / PetscMin(hx, hy);
  tol   = 1e-12 * scale;
  PetscCall(DMCreateGlobalVector(sol_dm, &WPfull));
  PetscCall(VecZeroEntries(WPfull));
  PetscCall(VecGetSubVector(P, is_p, &Psub2));
  PetscCall(VecGetSubVector(WPfull, is_U, &WPsub));
  PetscCall(MatMult(W, Psub2, WPsub));
  PetscCall(VecRestoreSubVector(WPfull, is_U, &WPsub));
  PetscCall(VecRestoreSubVector(P, is_p, &Psub2));
  PetscCall(DMGetLocalVector(sol_dm, &WPlocal));
  PetscCall(DMGlobalToLocalBegin(sol_dm, WPfull, INSERT_VALUES, WPlocal));
  PetscCall(DMGlobalToLocalEnd(sol_dm, WPfull, INSERT_VALUES, WPlocal));
  PetscCall(CheckWRow(sol_dm, WPlocal, Nx, 0, 2, c_U, hx, dt, rho, tol));
  PetscCall(CheckWRow(sol_dm, WPlocal, Nx, 1, 2, c_U, hx, dt, rho, tol));
  PetscCall(CheckWRow(sol_dm, WPlocal, Nx, 3, 2, c_U, hx, dt, rho, tol));
  PetscCall(DMRestoreLocalVector(sol_dm, &WPlocal));
  PetscCall(VecDestroy(&WPfull));

  PetscCall(DMCreateGlobalVector(sol_dm, &RPfull));
  PetscCall(VecZeroEntries(RPfull));
  PetscCall(VecGetSubVector(P, is_p, &Psub2));
  PetscCall(VecGetSubVector(RPfull, is_U, &RPsub));
  PetscCall(MatMult(negR, Psub2, RPsub));
  PetscCall(VecNorm(RPsub, NORM_INFINITY, &nrm));
  PetscCall(VecRestoreSubVector(RPfull, is_U, &RPsub));
  PetscCall(VecRestoreSubVector(P, is_p, &Psub2));
  PetscCheck(nrm <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "-R does not annihilate the quadratic pressure: max |-R P| = %g exceeds %g", (double)nrm, (double)tol);
  PetscCall(VecDestroy(&RPfull));

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
