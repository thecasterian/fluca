#include "phystest.h"

/* CHARACTERIZATION TEST — part of what is asserted here is a known defect, not a desired property.

   Push the analytic, exactly divergence-free Taylor-Green field u = (-cos(x) sin(y), sin(x) cos(y))
   on [0, 2 pi]^2 through the assembled coupling operators: form the face velocity U = b_interp + T u
   and then the discrete continuity residual r = D U - b_cont, cell by cell. No solver is involved;
   only the two assembled blocks and the affine boundary terms of the coupled system (13) are used.

   Periodic geometry: every interior continuity row is the wide difference (u_{i+1} - u_{i-1})/2h in
   each direction, and for this field the two directions cancel exactly, so r is round-off in every
   cell. That is a genuine correctness property of D and T and is asserted as such.

   Walled geometry: every cell that does not touch a wall is still at round-off, which is also a
   correctness property. The single cell layer next to the wall is not: its continuity row is the
   compact one-sided form ((u_0 + u_1)/2 - u_b)/h instead of the wide difference. Expanding that row
   about the cell center gives u_x + v_y + (h/8) u_xx + O(h^2), and the first two terms vanish for a
   divergence-free field, so the row leaves an O(h) residual behind wherever the prescribed boundary
   velocity has a nonzero wall-normal component. That one-cell-thick layer is the source of the
   accumulating walled pressure checkerboard (.superpowers/sdd/tsfsm/wall-source-ablation.md), and a
   one-cell-wide feature is Nyquist-scale by construction.

   The wall-layer assertions below (nonzero, and refining at O(h)) therefore PIN A DEFECT. If the
   wall cell's continuity row is ever made consistent with the interior rows — by extending the wide
   difference to the wall cell with a ghost built from the prescribed wall velocity, or by moving the
   interior rows to the compact face-difference form — the wall-layer residual will drop to round-off
   and this test will start failing. That is the intended outcome, and the fix is to tighten the wall
   bucket here to the same round-off tolerance the bulk bucket already uses.

   Note for whoever reads wall-source-ablation.md alongside this test: that report calls the wall row
   second-order consistent and quotes refinement ratios near 4. Measured here from the assembled
   blocks, on that report's own configuration as well as on this one, the ratios are near 2 and the
   residual is an order of magnitude larger than the report's table. The defect is real and localized
   exactly where the report says, but it is first order, not second. */

static const char help[] = "Continuity-row consistency of the assembled D and T blocks (solver free)\n"
                           "Pushes an analytic divergence-free field through U = b_interp + T u and\n"
                           "checks the per-cell residual D U - b_cont on periodic and walled grids.\n";

/* Taylor-Green velocity; div u = sin(x) sin(y) - sin(x) sin(y) = 0 for every (x, y) */
static PetscScalar TaylorGreenU(PetscReal x, PetscReal y)
{
  return -PetscCosReal(x) * PetscSinReal(y);
}

static PetscScalar TaylorGreenV(PetscReal x, PetscReal y)
{
  return PetscSinReal(x) * PetscCosReal(y);
}

/* The same field as the boundary velocity, so b_interp carries the exact wall data */
static PetscErrorCode TaylorGreenBC(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  if (comp == 0) *val = TaylorGreenU(x[0], x[1]);
  else if (comp == 1) *val = TaylorGreenV(x[0], x[1]);
  else *val = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Assemble the coupled system on an N x N grid of [0, 2 pi]^2 with boundary type bt, form
   U = b_interp + T u for the analytic field, and return the rms of the per-cell continuity residual
   D U - b_cont split into the cells touching a wall and the rest. u_max is the field scale, used to
   build a dimensionally correct round-off tolerance. Serial only: the bucketing uses global indices. */
static PetscErrorCode PhysTestContinuityResidual(DMBoundaryType bt, PetscInt N, PetscReal *rms_wall, PetscReal *rms_bulk, PetscReal *u_max)
{
  DM                dm, sol_dm;
  Phys              phys;
  Mat               M, negT, D;
  Vec               X, f, U, R, Rloc, Z;
  Vec               xv, bU, bp, rp;
  IS                is_v, is_U, is_p;
  PhysFieldLocation loc;
  PetscBool         walled;
  PetscInt          c_vel, c_p, xs, ys, xm, ym, i, j, nwall, nbulk;
  PetscReal         rho = 1., mu = 1., dt = 0.1, h, s_wall, s_bulk;

  PetscFunctionBeginUser;
  walled = bt == DM_BOUNDARY_PERIODIC ? PETSC_FALSE : PETSC_TRUE;
  h      = 2. * PETSC_PI / N;
  nwall  = 0;
  nbulk  = 0;
  s_wall = 0.;
  s_bulk = 0.;

  /* The grid sequence is intrinsic to the refinement study, so the DM is not taken from options */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, bt, bt, N, N, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));
  PetscCall(PhysTestCreateINS(dm, rho, mu, TaylorGreenBC, &phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, &loc, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, &loc, &c_p, NULL));

  /* X: the analytic field sampled at cell centers, face velocity and pressure left at zero */
  PetscCall(DMCreateGlobalVector(sol_dm, &X));
  PetscCall(VecZeroEntries(X));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  for (j = ys; j < ys + ym; ++j) {
    for (i = xs; i < xs + xm; ++i) {
      DMStagStencil st;
      PetscReal     xc, yc;
      PetscScalar   v;

      xc     = (i + .5) * h;
      yc     = (j + .5) * h;
      st.i   = i;
      st.j   = j;
      st.k   = 0;
      st.loc = DMSTAG_ELEMENT;
      st.c   = c_vel;
      v      = TaylorGreenU(xc, yc);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      st.c = c_vel + 1;
      v    = TaylorGreenV(xc, yc);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(PhysTestCreateSystem(phys, &M, &f));
  PetscCall(PhysComputeMomentumSystem(phys, 0., dt, X, M, f));
  PetscCall(PhysComputeCouplingSystem(phys, 0., dt, M, f));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_VELOCITY, &is_v));
  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_FACE_VELOCITY, &is_U));
  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_PRESSURE, &is_p));
  PetscCall(MatCreateSubMatrix(M, is_U, is_v, MAT_INITIAL_MATRIX, &negT));
  PetscCall(MatCreateSubMatrix(M, is_p, is_U, MAT_INITIAL_MATRIX, &D));

  /* Rhie-Chow row: -T u + U - R p' = b_interp, so U = b_interp + T u with T = -negT. At a boundary
     face the negT row was zeroed by PhysComputeCouplingSystem and b_interp is already u_b . n. */
  PetscCall(VecGetSubVector(X, is_v, &xv));
  PetscCall(VecGetSubVector(f, is_U, &bU));
  PetscCall(VecDuplicate(bU, &U));
  PetscCall(MatMult(negT, xv, U));
  PetscCall(VecAYPX(U, -1., bU));
  PetscCall(VecNorm(xv, NORM_INFINITY, u_max));
  PetscCall(VecRestoreSubVector(f, is_U, &bU));
  PetscCall(VecRestoreSubVector(X, is_v, &xv));

  /* Continuity row: D U = b_cont, so the residual of the analytic field is D U - b_cont */
  PetscCall(DMCreateGlobalVector(sol_dm, &R));
  PetscCall(VecZeroEntries(R));
  PetscCall(VecGetSubVector(R, is_p, &rp));
  PetscCall(VecGetSubVector(f, is_p, &bp));
  PetscCall(MatMult(D, U, rp));
  PetscCall(VecAXPY(rp, -1., bp));
  PetscCall(VecRestoreSubVector(f, is_p, &bp));
  PetscCall(VecRestoreSubVector(R, is_p, &rp));

  /* Periodic geometry has no wall row anywhere, so assert the stronger max-norm statement directly:
     the whole pressure field of the residual is at round-off. */
  if (!walled) {
    PetscCall(DMCreateGlobalVector(sol_dm, &Z));
    PetscCall(VecZeroEntries(Z));
    PetscCall(PhysTestCheckField(phys, PHYS_FIELD_PRESSURE, R, Z, 1e-12 * *u_max / h));
    PetscCall(VecDestroy(&Z));
  }

  PetscCall(DMGetLocalVector(sol_dm, &Rloc));
  PetscCall(DMGlobalToLocalBegin(sol_dm, R, INSERT_VALUES, Rloc));
  PetscCall(DMGlobalToLocalEnd(sol_dm, R, INSERT_VALUES, Rloc));
  for (j = ys; j < ys + ym; ++j) {
    for (i = xs; i < xs + xm; ++i) {
      DMStagStencil st;
      PetscScalar   r;
      PetscReal     r2;

      st.i   = i;
      st.j   = j;
      st.k   = 0;
      st.loc = DMSTAG_ELEMENT;
      st.c   = c_p;
      PetscCall(DMStagVecGetValuesStencil(sol_dm, Rloc, 1, &st, &r));
      r2 = PetscAbsScalar(r);
      r2 *= r2;
      if (walled && (i == 0 || i == N - 1 || j == 0 || j == N - 1)) {
        s_wall += r2;
        ++nwall;
      } else {
        s_bulk += r2;
        ++nbulk;
      }
    }
  }
  PetscCall(DMRestoreLocalVector(sol_dm, &Rloc));
  *rms_wall = nwall > 0 ? PetscSqrtReal(s_wall / nwall) : 0.;
  *rms_bulk = nbulk > 0 ? PetscSqrtReal(s_bulk / nbulk) : 0.;

  PetscCall(VecDestroy(&R));
  PetscCall(VecDestroy(&U));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&negT));
  PetscCall(ISDestroy(&is_p));
  PetscCall(ISDestroy(&is_U));
  PetscCall(ISDestroy(&is_v));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&M));
  PetscCall(VecDestroy(&X));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt  grid[3] = {16, 32, 64};
  PetscReal wall[3], bulk[3];
  PetscReal u_max, tol, ratio;
  PetscInt  k;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Periodic: the residual must be round-off in every cell. D U has the units of u / h, so the
     round-off tolerance carries a 1 / h. */
  for (k = 0; k < 3; ++k) {
    PetscCall(PhysTestContinuityResidual(DM_BOUNDARY_PERIODIC, grid[k], &wall[k], &bulk[k], &u_max));
    PetscCheck(u_max > .5, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Periodic %" PetscInt_FMT ": the sampled field is trivial, |u|_max = %g", grid[k], (double)u_max);
    tol = 1e-12 * u_max * grid[k] / (2. * PETSC_PI);
    PetscCheck(bulk[k] <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Periodic %" PetscInt_FMT ": continuity residual %g exceeds round-off %g", grid[k], (double)bulk[k], (double)tol);
  }

  /* Walled: every cell away from the wall is still at round-off, while the wall-adjacent layer
     carries a nonzero residual. CHARACTERIZATION — see the note at the top of this file. */
  for (k = 0; k < 3; ++k) {
    PetscCall(PhysTestContinuityResidual(DM_BOUNDARY_NONE, grid[k], &wall[k], &bulk[k], &u_max));
    PetscCheck(u_max > .5, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled %" PetscInt_FMT ": the sampled field is trivial, |u|_max = %g", grid[k], (double)u_max);
    tol = 1e-12 * u_max * grid[k] / (2. * PETSC_PI);
    PetscCheck(bulk[k] <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled %" PetscInt_FMT ": continuity residual away from the wall is %g, above round-off %g", grid[k], (double)bulk[k], (double)tol);
    PetscCheck(wall[k] > 1e6 * tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled %" PetscInt_FMT ": the wall-layer residual %g collapsed to round-off; if the wall continuity row was made consistent, tighten this test", grid[k], (double)wall[k]);
  }

  /* The wall-layer residual refines at first order: successive rms values fall by ~2 */
  for (k = 0; k + 1 < 3; ++k) {
    ratio = wall[k] / wall[k + 1];
    PetscCheck(ratio >= 1.8 && ratio <= 2.2, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Wall-layer residual %" PetscInt_FMT " -> %" PetscInt_FMT ": ratio %g is not O(h)", grid[k], grid[k + 1], (double)ratio);
  }

  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: continuity_consistency
    nsize: 1
    output_file: output/empty.out

TEST*/
