#include <fluca/private/segcnlinearimpl.h>

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
static PetscErrorCode AddBodyForce_Private(Seg seg, PetscReal t, PetscReal scale, Vec f)
{
  Seg_CNLinear       *cn      = (Seg_CNLinear *)seg->data;
  Seg_Ops            *ops     = &cn->ops;
  Phys                phys    = seg->phys;
  PetscInt            dim     = ops->dim;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, zs, xm, ym, zm, slot_elem, i, j, k, d;
  DM                  sol_dm;

  PetscFunctionBegin;
  if (!phys->bodyforce) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
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
          row.c = ops->c_vel + d;
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
   A u^{n+1} + G p' = u^n + (dt/2) nu lap(u^n) - (dt/rho) grad(q) + boundary terms

   Input: t = t^n, dt, X = state at t^n on the solution DM (velocity u^n, face velocity U^n, pressure q = p^{n-1/2}).
   Adds A (velocity columns) and G (pressure columns) into the velocity rows of M with ADD_VALUES and
   without assembling M, and adds r + b_mom into the velocity rows of f. Other rows are untouched. */
PetscErrorCode SegCNLinearComputeMomentumSystem_Internal(Seg seg, PetscReal t, PetscReal dt, Vec X, Mat M, Vec f)
{
  Seg_CNLinear *cn  = (Seg_CNLinear *)seg->data;
  Seg_Ops      *ops = &cn->ops;
  PetscInt      dim = ops->dim, d, e;
  Vec           tmp, fv, xv;
  PetscScalar   rho;
  DM            sol_dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCheck(seg->setupcalled, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Must call SegSetUp() before assembling the system");
  PetscValidLogicalCollectiveReal(seg, t, 2);
  PetscValidLogicalCollectiveReal(seg, dt, 3);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(M, MAT_CLASSID, 5);
  PetscValidHeaderSpecific(f, VEC_CLASSID, 6);
  PetscCheck(dt > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Time step must be positive, got %g", (double)dt);
  PetscCall(PhysGetSolutionDM(seg->phys, &sol_dm));
  PetscCall(PhysGetPropertyConstant(seg->phys, PHYS_PROPERTY_DENSITY, &rho));
  /* Coefficients that depend on dt */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDScaleSetConstant(ops->fd_visc[d], dt / (2. * rho)));
    PetscCall(FlucaFDScaleSetConstant(ops->fd_conv[d], dt / 2.));
    PetscCall(FlucaFDScaleSetConstant(ops->fd_grad[d], dt / rho));
  }

  /* Linearization state: U^n from X, and ubar^n with boundary values at t */
  for (d = 0; d < dim; d++) {
    PetscCall(VecZeroEntries(ops->ubar[d]));
    for (e = 0; e < dim; e++) PetscCall(FlucaFDApply(ops->fd_interp_vel[d][e], t, sol_dm, ops->dm_face, X, ops->ubar[d]));
  }
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) {
      PetscCall(FlucaFDScaleSetVector(ops->fd_conv_U[d][e], X, face_loc[e], ops->c_U));
      PetscCall(FlucaFDScaleSetVector(ops->fd_conv_ubar[d][e], ops->ubar[d], face_loc[e], 0));
    }
  }

  /* Matrix: A in the velocity columns, G in the pressure columns */
  for (d = 0; d < dim; d++) {
    PetscCall(AddIdentity_Private(sol_dm, M, DMSTAG_ELEMENT, ops->c_vel + d));
    PetscCall(FlucaFDGetOperator(ops->fd_visc[d], sol_dm, sol_dm, M));
    PetscCall(FlucaFDGetOperator(ops->fd_conv[d], sol_dm, sol_dm, M));
    PetscCall(FlucaFDGetOperator(ops->fd_grad[d], sol_dm, sol_dm, M));
  }

  /* Right-hand side */
  PetscCall(DMGetGlobalVector(sol_dm, &tmp));
  for (d = 0; d < dim; d++) {
    /* +(dt/2) nu (lap u^n + b^n) */
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ops->fd_visc[d], t, sol_dm, sol_dm, X, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    /* -(dt/rho) grad q */
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ops->fd_grad[d], t, sol_dm, sol_dm, X, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    /* Boundary parts of A at t + dt move to the right-hand side */
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ops->fd_visc[d], t + dt, sol_dm, sol_dm, ops->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ops->fd_conv[d], t + dt, sol_dm, sol_dm, ops->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
  }
  PetscCall(DMRestoreGlobalVector(sol_dm, &tmp));

  /* +u^n */
  PetscCall(VecGetSubVector(f, ops->is_vel, &fv));
  PetscCall(VecGetSubVector(X, ops->is_vel, &xv));
  PetscCall(VecAXPY(fv, 1., xv));
  PetscCall(VecRestoreSubVector(X, ops->is_vel, &xv));
  PetscCall(VecRestoreSubVector(f, ops->is_vel, &fv));

  /* Body force per unit mass, time-centered */
  PetscCall(AddBodyForce_Private(seg, t + dt / 2., dt / rho, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* M += scale * negR on the face-velocity rows. negR is assembled once, unscaled, in segops.c; only
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
   Boundary faces carry the prescribed normal velocity, U = u_b . n.

   Input: t = time of the boundary data (t^{n+1} within a step), dt.
   Adds -T, I, -R into the face-velocity rows and D into the pressure rows of M, and b_interp, b_cont into f.
   M is assembled on return, with boundary-face rows replaced by unit rows. */
PetscErrorCode SegCNLinearComputeCouplingSystem_Internal(Seg seg, PetscReal t, PetscReal dt, Mat M, Vec f)
{
  Seg_CNLinear *cn  = (Seg_CNLinear *)seg->data;
  Seg_Ops      *ops = &cn->ops;
  PetscInt      dim = ops->dim, e;
  PetscScalar   rho;
  DM            sol_dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCheck(seg->setupcalled, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Must call SegSetUp() before assembling the system");
  PetscValidLogicalCollectiveReal(seg, t, 2);
  PetscValidLogicalCollectiveReal(seg, dt, 3);
  PetscValidHeaderSpecific(M, MAT_CLASSID, 4);
  PetscValidHeaderSpecific(f, VEC_CLASSID, 5);
  PetscCheck(dt > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Time step must be positive, got %g", (double)dt);
  PetscCall(PhysGetSolutionDM(seg->phys, &sol_dm));
  PetscCall(PhysGetPropertyConstant(seg->phys, PHYS_PROPERTY_DENSITY, &rho));
  for (e = 0; e < dim; e++) {
    PetscCall(AddIdentity_Private(sol_dm, M, face_loc[e], ops->c_U));
    PetscCall(FlucaFDGetOperator(ops->fd_negT[e], sol_dm, sol_dm, M));
    PetscCall(FlucaFDApply(ops->fd_bface[e], t, sol_dm, sol_dm, ops->zero, f));
  }
  PetscCall(AddNegR_Private(M, ops->negR, dt / rho));
  PetscCall(FlucaFDGetOperator(ops->fd_D, sol_dm, sol_dm, M));
  PetscCall(FlucaFDApply(ops->fd_D, t, sol_dm, sol_dm, ops->zero, f));

  /* A boundary-face row is the boundary condition itself, so replace it by a unit row; the right-hand
     side already holds u_b . n, because the interpolation reproduces the Dirichlet datum there with a
     unit weight and every interior weight zero. */
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatZeroRowsLocal(M, ops->nbface, ops->bface, 1., NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
