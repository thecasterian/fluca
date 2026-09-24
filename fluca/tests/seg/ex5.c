#include "segtest.h"

/* Continuity-row consistency of the assembled D and T blocks, solver free.

   Push an analytic, exactly divergence-free field through the assembled coupling operators: form the
   face velocity U = b_interp + T u and then the discrete continuity residual r = D U - b_cont, cell by
   cell. No solver is involved; only the two assembled blocks and the affine boundary terms of the
   coupled system (13) are used. For a divergence-free field the exact answer is zero everywhere, so r
   is the truncation error of the continuity operator itself.

   The continuity row of a cell is the flux difference (U_{f+1} - U_f)/h, so its truncation error is
   the difference of the face errors of T, not the face errors themselves. The interior rows are
   therefore second order even though each face carries an O(h^2) interpolation error: that error is a
   smooth field and its difference across a cell is O(h^3). A prescribed-velocity boundary face is
   where the argument used to break, because U there is the boundary datum itself and carries no error
   at all; differencing an O(h^2) face error against zero left the wall cell's row first order and the
   whole continuity operator with a one-cell-thick O(h) truncation layer. The fourth-order accurate
   interpolation T now used in the Rhie-Chow row (see fluca/src/seg/utils/ops/segops.c) makes every
   face error O(h^3) instead, which removes the layer without touching either invariant: U is still
   exactly u_b . n on a prescribed-velocity face, and D is still a flux difference.

   Grids 32, 64 and 128: the coarsest pair of a 16, 32, 64 sequence is still pre-asymptotic for the
   oblique field (ratio 4.8), and the two-sided brackets below would have to be loosened to hold it.

   What is asserted here, on two independent divergence-free fields and both with a nonzero
   wall-normal component, so that neither the prescribed through-wall flux nor a cancellation special
   to one field is doing the work:

     1. periodic Taylor-Green: r is at round-off in every cell. The interior row is the same
        translation-invariant antisymmetric stencil in both directions, and it annihilates this field
        exactly at any h.
     2. periodic oblique field: r refines at second order. The same interior rows, on a field the
        stencil does not annihilate.
     3. walled, both fields: the residual of the cells touching a wall refines at second order. This
        is the property the wall-row redesign exists to deliver; it was first order (ratios near 2)
        while T was the two-point average.
     4. walled oblique field: the wall layer is no larger than twice the bulk on every grid. A layer
        one order lower would separate from the bulk like 1/h under refinement, which is what the
        previous discretization did: its wall-to-bulk ratio ran 2.2, 4.1, 8.1 over these three grids,
        against 0.99, 0.99, 0.99 now.
     5. guard: |u|_inf is O(1) on every grid, so a passing refinement check cannot come from a
        trivial field.

   The Taylor-Green wall ratios are near 8, not 4: that field is superconvergent at the wall for this
   stencil. Its wall assertion is therefore one-sided (at least second order), while the oblique field,
   which has no such cancellation, carries the two-sided bracket.

   The four-point interpolation is one-sided at the face next to the wall, so a wall perturbs two cell
   rows, not one; the wall bucket below is therefore two cells deep (i <= 1 || i >= N-2, same in j), not
   one. With that bucket, Taylor-Green is again at round-off in the bulk on the walled grid, exactly as
   in the periodic case, because cell 2 and beyond see only the symmetric {i-2..i+1} face stencil, which
   annihilates Taylor-Green exactly; the walled bulk assertion is therefore the same round-off check as
   the periodic one, not a refinement ratio. */

static const char help[] = "Continuity-row consistency of the assembled D and T blocks (solver free)\n"
                           "Pushes analytic divergence-free fields through U = b_interp + T u and\n"
                           "checks the per-cell residual D U - b_cont on periodic and walled grids.\n";

/* Field 0 is Taylor-Green, psi = -cos(x) cos(y); field 1 is the oblique field psi = cos(x + 2y).
   Both are divergence free identically and both have a nonzero normal component on every wall of
   [0, 2 pi]^2, so the walled cases all carry prescribed through-wall flux. */
static PetscInt field_id = 0;

static PetscScalar FieldU(PetscReal x, PetscReal y)
{
  if (field_id == 0) return -PetscCosReal(x) * PetscSinReal(y);
  return -2. * PetscSinReal(x + 2. * y);
}

static PetscScalar FieldV(PetscReal x, PetscReal y)
{
  if (field_id == 0) return PetscSinReal(x) * PetscCosReal(y);
  return PetscSinReal(x + 2. * y);
}

/* The same field as the boundary velocity, so b_interp carries the exact wall data */
static PetscErrorCode FieldBC(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  if (comp == 0) *val = FieldU(x[0], x[1]);
  else if (comp == 1) *val = FieldV(x[0], x[1]);
  else *val = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Assemble the coupled system on an N x N grid of [0, 2 pi]^2 with boundary type bt, form
   U = b_interp + T u for the analytic field, and return the rms of the per-cell continuity residual
   D U - b_cont split into the cells touching a wall and the rest. u_max is the field scale, used to
   build a dimensionally correct round-off tolerance. Serial only: the bucketing uses global indices. */
static PetscErrorCode SegTestContinuityResidual(DMBoundaryType bt, PetscInt N, PetscReal *rms_wall, PetscReal *rms_bulk, PetscReal *u_max)
{
  DM                dm, sol_dm;
  Phys              phys;
  Seg               seg;
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
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, bt, bt, N, N, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));
  PetscCall(SegTestSetUp(dm, rho, mu, FieldBC, &phys, &seg));
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
      v      = FieldU(xc, yc);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      st.c = c_vel + 1;
      v    = FieldV(xc, yc);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(SegTestCreateSystem(phys, &M, &f));
  PetscCall(SegCNLinearComputeMomentumSystem_Internal(seg, 0., dt, X, M, f));
  PetscCall(SegCNLinearComputeCouplingSystem_Internal(seg, 0., dt, M, f));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_VELOCITY, &is_v));
  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_FACE_VELOCITY, &is_U));
  PetscCall(PhysGetFieldIS(phys, PHYS_FIELD_PRESSURE, &is_p));
  PetscCall(MatCreateSubMatrix(M, is_U, is_v, MAT_INITIAL_MATRIX, &negT));
  PetscCall(MatCreateSubMatrix(M, is_p, is_U, MAT_INITIAL_MATRIX, &D));

  /* Rhie-Chow row: -T u + U - R p' = b_interp, so U = b_interp + T u with T = -negT. At a boundary
     face the negT row was zeroed by the coupling rows and b_interp is already u_b . n. */
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

  /* Periodic Taylor-Green: the interior stencil annihilates the field exactly, so assert the
     stronger max-norm statement directly. */
  if (!walled && field_id == 0) {
    PetscCall(DMCreateGlobalVector(sol_dm, &Z));
    PetscCall(VecZeroEntries(Z));
    PetscCall(SegTestCheckField(phys, PHYS_FIELD_PRESSURE, R, Z, 1e-12 * *u_max / h));
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
      if (walled && (i <= 1 || i >= N - 2 || j <= 1 || j >= N - 2)) {
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
  PetscCall(SegDestroy(&seg));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const PetscInt grid[3] = {32, 64, 128};
  PetscReal      wall[3], bulk[3];
  PetscReal      u_max, tol, ratio;
  PetscInt       k;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Periodic Taylor-Green: the residual must be round-off in every cell. D U has the units of u / h,
     so the round-off tolerance carries a 1 / h. */
  field_id = 0;
  for (k = 0; k < 3; ++k) {
    PetscCall(SegTestContinuityResidual(DM_BOUNDARY_PERIODIC, grid[k], &wall[k], &bulk[k], &u_max));
    PetscCheck(u_max > .5, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Periodic %" PetscInt_FMT ": the sampled field is trivial, |u|_max = %g", grid[k], (double)u_max);
    tol = 1e-12 * u_max * grid[k] / (2. * PETSC_PI);
    PetscCheck(bulk[k] <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Periodic %" PetscInt_FMT ": continuity residual %g exceeds round-off %g", grid[k], (double)bulk[k], (double)tol);
  }

  /* Periodic oblique field: the interior rows carry a genuine truncation error and it is second
     order. This is what the periodic round-off check above cannot see. */
  field_id = 1;
  for (k = 0; k < 3; ++k) {
    PetscCall(SegTestContinuityResidual(DM_BOUNDARY_PERIODIC, grid[k], &wall[k], &bulk[k], &u_max));
    PetscCheck(u_max > .5, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Periodic oblique %" PetscInt_FMT ": the sampled field is trivial, |u|_max = %g", grid[k], (double)u_max);
  }
  for (k = 0; k + 1 < 3; ++k) {
    ratio = bulk[k] / bulk[k + 1];
    PetscCheck(ratio >= 3.5 && ratio <= 5., PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Periodic oblique residual %" PetscInt_FMT " -> %" PetscInt_FMT ": ratio %g is not O(h^2)", grid[k], grid[k + 1], (double)ratio);
  }

  /* Walled Taylor-Green: the bulk is at round-off, same as the periodic case, now that the wall
     bucket is two cells deep. The wall layer refines at least at second order; the bracket is
     one-sided because this field is superconvergent at the wall for this stencil (measured ratios
     near 8). */
  field_id = 0;
  for (k = 0; k < 3; ++k) {
    PetscCall(SegTestContinuityResidual(DM_BOUNDARY_NONE, grid[k], &wall[k], &bulk[k], &u_max));
    PetscCheck(u_max > .5, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled %" PetscInt_FMT ": the sampled field is trivial, |u|_max = %g", grid[k], (double)u_max);
    tol = 1e-12 * u_max * grid[k] / (2. * PETSC_PI);
    PetscCheck(bulk[k] <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled %" PetscInt_FMT ": bulk continuity residual %g exceeds round-off %g", grid[k], (double)bulk[k], (double)tol);
  }
  for (k = 0; k + 1 < 3; ++k) {
    ratio = wall[k] / wall[k + 1];
    PetscCheck(ratio >= 3.5, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled wall-layer residual %" PetscInt_FMT " -> %" PetscInt_FMT ": ratio %g is below O(h^2)", grid[k], grid[k + 1], (double)ratio);
  }

  /* Walled oblique field: the wall layer and the bulk both refine at second order, and the wall layer
     stays the size of the bulk instead of separating from it like 1 / h. */
  field_id = 1;
  for (k = 0; k < 3; ++k) {
    PetscCall(SegTestContinuityResidual(DM_BOUNDARY_NONE, grid[k], &wall[k], &bulk[k], &u_max));
    PetscCheck(u_max > .5, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled oblique %" PetscInt_FMT ": the sampled field is trivial, |u|_max = %g", grid[k], (double)u_max);
    PetscCheck(wall[k] <= 2. * bulk[k], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled oblique %" PetscInt_FMT ": wall-layer residual %g is more than twice the bulk residual %g, i.e. a lower-order wall layer", grid[k], (double)wall[k], (double)bulk[k]);
  }
  for (k = 0; k + 1 < 3; ++k) {
    ratio = wall[k] / wall[k + 1];
    PetscCheck(ratio >= 3.5 && ratio <= 5., PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled oblique wall-layer residual %" PetscInt_FMT " -> %" PetscInt_FMT ": ratio %g is not O(h^2)", grid[k], grid[k + 1], (double)ratio);
    ratio = bulk[k] / bulk[k + 1];
    PetscCheck(ratio >= 3.5 && ratio <= 5., PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Walled oblique bulk residual %" PetscInt_FMT " -> %" PetscInt_FMT ": ratio %g is not O(h^2)", grid[k], grid[k + 1], (double)ratio);
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
