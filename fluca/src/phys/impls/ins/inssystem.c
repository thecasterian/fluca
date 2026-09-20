#include <fluca/private/physinsimpl.h>

/* Face stencil locations indexed by direction: LEFT for x, DOWN for y, BACK for z */
static const DMStagStencilLocation face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* Add 1 to the diagonal of every locally owned row at (loc, c), including the extra boundary faces */
static PetscErrorCode AddIdentity_Private(DM dm, Mat M, DMStagStencilLocation loc, PetscInt c)
{
  PetscInt      dim, xs, ys, zs, xm, ym, zm, nx, ny, nz, ie, je, ke, i, j, k;
  DMStagStencil row;
  PetscScalar   one = 1.;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm, &nx, &ny, &nz));
  ie = xs + xm + (loc == DMSTAG_LEFT ? nx : 0);
  je = dim >= 2 ? ys + ym + (loc == DMSTAG_DOWN ? ny : 0) : 1;
  ke = dim >= 3 ? zs + zm + (loc == DMSTAG_BACK ? nz : 0) : 1;
  if (dim < 2) ys = 0;
  if (dim < 3) zs = 0;
  row.loc = loc;
  row.c   = c;
  for (k = zs; k < ke; ++k) {
    for (j = ys; j < je; ++j) {
      for (i = xs; i < ie; ++i) {
        row.i = i;
        row.j = j;
        row.k = k;
        PetscCall(DMStagMatSetValuesStencil(dm, M, 1, &row, 1, &row, &one, ADD_VALUES));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* f_d += scale * body_force_d(t) at every cell center */
static PetscErrorCode AddBodyForce_Private(Phys phys, PetscReal t, PetscReal scale, Vec f)
{
  Phys_INS           *ins     = (Phys_INS *)phys->data;
  DM                  sol_dm  = phys->sol_dm;
  PetscInt            dim     = phys->dim;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, zs, xm, ym, zm, slot_elem, i, j, k, d;

  PetscFunctionBegin;
  if (!phys->bodyforce) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));
  if (dim < 3) {
    zs = 0;
    zm = 1;
  }
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal     coords[3] = {0., 0., 0.};
        PetscScalar   force[3], v;
        DMStagStencil row;

        coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
        coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
        if (dim == 3) coords[2] = PetscRealPart(arrc[2][k][slot_elem]);
        PetscCall(phys->bodyforce(dim, t, coords, force, phys->bodyforce_ctx));
        row.i   = i;
        row.j   = j;
        row.k   = k;
        row.loc = DMSTAG_ELEMENT;
        for (d = 0; d < dim; d++) {
          row.c = ins->c_vel + d;
          v     = scale * force[d];
          PetscCall(DMStagVecSetValuesStencil(sol_dm, f, 1, &row, &v, ADD_VALUES));
        }
      }
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Momentum rows of the coupled system (13), guide eq. (6) and (9):
   A u^{n+1} + G p' = u^n + (dt/2) nu lap(u^n) - (dt/rho) grad(q) + boundary terms */
PetscErrorCode PhysComputeMomentumSystem_INS(Phys phys, PetscReal t, PetscReal dt, Vec X, Mat M, Vec f)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d, e;
  Vec       tmp, fv, xv;

  PetscFunctionBegin;
  /* Coefficients that depend on dt */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDScaleSetConstant(ins->fd_visc[d], dt / (2. * ins->rho)));
    PetscCall(FlucaFDScaleSetConstant(ins->fd_conv[d], dt / 2.));
    PetscCall(FlucaFDScaleSetConstant(ins->fd_grad[d], dt / ins->rho));
  }

  /* Linearization state: U^n from X, and ubar^n with boundary values at t */
  for (d = 0; d < dim; d++) {
    PetscCall(VecZeroEntries(ins->ubar[d]));
    for (e = 0; e < dim; e++) PetscCall(FlucaFDApply(ins->fd_interp_vel[d][e], t, sol_dm, ins->dm_face, X, ins->ubar[d]));
  }
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) {
      PetscCall(FlucaFDScaleSetVector(ins->fd_conv_U[d][e], X, face_loc[e], ins->c_U));
      PetscCall(FlucaFDScaleSetVector(ins->fd_conv_ubar[d][e], ins->ubar[d], face_loc[e], 0));
    }
  }

  /* Matrix: A in the velocity columns, G in the pressure columns */
  for (d = 0; d < dim; d++) {
    PetscCall(AddIdentity_Private(sol_dm, M, DMSTAG_ELEMENT, ins->c_vel + d));
    PetscCall(FlucaFDGetOperator(ins->fd_visc[d], sol_dm, sol_dm, M));
    PetscCall(FlucaFDGetOperator(ins->fd_conv[d], sol_dm, sol_dm, M));
    PetscCall(FlucaFDGetOperator(ins->fd_grad[d], sol_dm, sol_dm, M));
  }

  /* Right-hand side */
  PetscCall(DMGetGlobalVector(sol_dm, &tmp));
  for (d = 0; d < dim; d++) {
    /* +(dt/2) nu (lap u^n + b^n) */
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ins->fd_visc[d], t, sol_dm, sol_dm, X, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    /* -(dt/rho) grad q */
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ins->fd_grad[d], t, sol_dm, sol_dm, X, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    /* Boundary parts of A at t + dt move to the right-hand side */
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ins->fd_visc[d], t + dt, sol_dm, sol_dm, ins->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ins->fd_conv[d], t + dt, sol_dm, sol_dm, ins->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
  }
  PetscCall(DMRestoreGlobalVector(sol_dm, &tmp));

  /* +u^n */
  PetscCall(VecGetSubVector(f, ins->is_vel, &fv));
  PetscCall(VecGetSubVector(X, ins->is_vel, &xv));
  PetscCall(VecAXPY(fv, 1., xv));
  PetscCall(VecRestoreSubVector(X, ins->is_vel, &xv));
  PetscCall(VecRestoreSubVector(f, ins->is_vel, &fv));

  /* Body force per unit mass, time-centered */
  PetscCall(AddBodyForce_Private(phys, t + dt / 2., dt / ins->rho, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* M += scale * negR on the face-velocity rows. negR is assembled once, unscaled, in insops.c; only
   the dt/rho factor of guide eq. (11) changes from step to step. */
static PetscErrorCode AddNegR_Private(Mat M, Mat negR, PetscScalar scale)
{
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscScalar       *row;
  PetscInt           rstart, rend, r, ncols, maxcols, c;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(negR, &rstart, &rend));
  maxcols = 0;
  for (r = rstart; r < rend; ++r) {
    PetscCall(MatGetRow(negR, r, &ncols, NULL, NULL));
    maxcols = PetscMax(maxcols, ncols);
    PetscCall(MatRestoreRow(negR, r, &ncols, NULL, NULL));
  }
  PetscCall(PetscMalloc1(maxcols, &row));
  for (r = rstart; r < rend; ++r) {
    PetscCall(MatGetRow(negR, r, &ncols, &cols, &vals));
    for (c = 0; c < ncols; ++c) row[c] = scale * vals[c];
    if (ncols > 0) PetscCall(MatSetValues(M, 1, &r, ncols, cols, row, ADD_VALUES));
    PetscCall(MatRestoreRow(negR, r, &ncols, &cols, &vals));
  }
  PetscCall(PetscFree(row));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Rhie-Chow rows (guide eq. (11)): -T u + U - R p' = b_interp, and continuity rows (guide eq. (10)): D U = b_cont.
   Boundary faces carry the prescribed normal velocity, U = u_b . n. */
PetscErrorCode PhysComputeCouplingSystem_INS(Phys phys, PetscReal t, PetscReal dt, Mat M, Vec f)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, e;

  PetscFunctionBegin;
  for (e = 0; e < dim; e++) {
    PetscCall(AddIdentity_Private(sol_dm, M, face_loc[e], ins->c_U));
    PetscCall(FlucaFDGetOperator(ins->fd_negT[e], sol_dm, sol_dm, M));
    PetscCall(FlucaFDApply(ins->fd_bface[e], t, sol_dm, sol_dm, ins->zero, f));
  }
  PetscCall(AddNegR_Private(M, ins->negR, dt / ins->rho));
  PetscCall(FlucaFDGetOperator(ins->fd_D, sol_dm, sol_dm, M));
  PetscCall(FlucaFDApply(ins->fd_D, t, sol_dm, sol_dm, ins->zero, f));

  /* A boundary-face row is the boundary condition itself, so replace it by a unit row; the right-hand
     side already holds u_b . n, because the interpolation reproduces the Dirichlet datum there with a
     unit weight and every interior weight zero. */
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatZeroRowsLocal(M, ins->nbface, ins->bface, 1., NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
