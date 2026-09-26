#include <fluca/private/nsimpl.h>

typedef struct {
  Vec     phalf;                                     /* pressure at n-1/2 in its pressure entries */
  DM      dm_face;                                   /* one DOF per face */
  Vec     ubar[PHYS_MAX_DIM];                        /* u_d^n interpolated to faces, with boundary values */
  FlucaFD fd_interp_vel[PHYS_MAX_DIM][PHYS_MAX_DIM]; /* [d][e]: u_d -> faces normal to e */
  FlucaFD fd_conv_U[PHYS_MAX_DIM][PHYS_MAX_DIM];     /* [d][e]: interp(u_d) * U_e^n */
  FlucaFD fd_conv_ubar[PHYS_MAX_DIM][PHYS_MAX_DIM];  /* [d][e]: ubar_d^n * interp(u_e) */
  FlucaFD fd_conv[PHYS_MAX_DIM];                     /* (dt/2) sum_e d/dx_e(...) */
  FlucaFD fd_visc[PHYS_MAX_DIM];                     /* (dt/(2 rho)) fd_laplacian[d] */
  FlucaFD fd_grad[PHYS_MAX_DIM];                     /* (dt/rho) fd_grad_p[d] */
} NS_CNLinear;

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

/* M += scale * A, row by row with ADD_VALUES; M is not assembled */
static PetscErrorCode AddScaledMatrix_Private(Mat M, Mat A, PetscScalar scale)
{
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscScalar       *row;
  PetscInt           rstart, rend, r, ncols, maxcols = 0, c;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (r = rstart; r < rend; ++r) {
    PetscCall(MatGetRow(A, r, &ncols, NULL, NULL));
    maxcols = PetscMax(maxcols, ncols);
    PetscCall(MatRestoreRow(A, r, &ncols, NULL, NULL));
  }
  PetscCall(PetscMalloc1(maxcols, &row));
  for (r = rstart; r < rend; ++r) {
    PetscCall(MatGetRow(A, r, &ncols, &cols, &vals));
    for (c = 0; c < ncols; ++c) row[c] = scale * vals[c];
    if (ncols > 0) PetscCall(MatSetValues(M, 1, &r, ncols, cols, row, ADD_VALUES));
    PetscCall(MatRestoreRow(A, r, &ncols, &cols, &vals));
  }
  PetscCall(PetscFree(row));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* f_d += scale * body_force_d(t) at every cell center */
static PetscErrorCode AddBodyForce_Private(NS ns, PetscReal t, PetscReal scale, Vec f)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PhysBodyForceFn    *bodyforce;
  void               *bodyforce_ctx;
  PetscInt            dim, c_vel, xs, ys, zs, xm, ym, zm, slot_elem, i, j, k, d;
  DM                  dm;

  PetscFunctionBegin;
  PetscCall(PhysGetBodyForce(ns->phys, &bodyforce, &bodyforce_ctx));
  if (!bodyforce) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));
  if (dim < 3) {
    zs = 0;
    zm = 1;
  }
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal     coords[3] = {0., 0., 0.};
        PetscScalar   force[3];
        PetscScalar   v;
        DMStagStencil row;

        coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
        coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
        if (dim == 3) coords[2] = PetscRealPart(arrc[2][k][slot_elem]);
        PetscCall(bodyforce(dim, t, coords, force, bodyforce_ctx));
        row.i   = i;
        row.j   = j;
        row.k   = k;
        row.loc = DMSTAG_ELEMENT;
        for (d = 0; d < dim; d++) {
          row.c = c_vel + d;
          v     = scale * force[d];
          PetscCall(DMStagVecSetValuesStencil(dm, f, 1, &row, &v, ADD_VALUES));
        }
      }
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Refresh what depends on the material properties, dt, or the state at t^n (sol0) */
static PetscErrorCode UpdateOperators_Private(NS ns)
{
  NS_CNLinear *cn = (NS_CNLinear *)ns->data;
  PetscScalar  mu, rho;
  PetscInt     dim, c_U, d, e;
  DM           dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetProperty(ns->phys, PHYS_PROPERTY_VISCOSITY, &mu));
  PetscCall(PhysGetProperty(ns->phys, PHYS_PROPERTY_DENSITY, &rho));
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) PetscCall(FlucaFDScaleSetConstant(ns->fd_negmu[d][e], -mu));
    PetscCall(FlucaFDScaleSetConstant(cn->fd_visc[d], ns->dt / (2. * rho)));
    PetscCall(FlucaFDScaleSetConstant(cn->fd_conv[d], ns->dt / 2.));
    PetscCall(FlucaFDScaleSetConstant(cn->fd_grad[d], ns->dt / rho));
  }
  for (d = 0; d < dim; d++) {
    PetscCall(VecZeroEntries(cn->ubar[d]));
    for (e = 0; e < dim; e++) PetscCall(FlucaFDApply(cn->fd_interp_vel[d][e], ns->t, dm, cn->dm_face, ns->sol0, cn->ubar[d]));
  }
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) {
      PetscCall(FlucaFDScaleSetVector(cn->fd_conv_U[d][e], ns->sol0, face_loc[e], c_U));
      PetscCall(FlucaFDScaleSetVector(cn->fd_conv_ubar[d][e], cn->ubar[d], face_loc[e], 0));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Time-discrete operators on top of the spatial operators of the NS base class. Coefficients that
   depend on dt and the linearization state are set per step by UpdateOperators_Private(). */
static PetscErrorCode NSSetUp_CNLinear(NS ns)
{
  NS_CNLinear *cn = (NS_CNLinear *)ns->data;
  PetscInt     dim, c_vel, c_U, d, e;
  DM           dm, cdm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));

  PetscCall(DMCreateGlobalVector(dm, &cn->phalf));

  /* ubar_d^n lives on a one-DOF-per-face DM */
  switch (dim) {
  case 2:
    PetscCall(DMStagCreateCompatibleDMStag(dm, 0, 1, 0, 0, &cn->dm_face));
    break;
  case 3:
    PetscCall(DMStagCreateCompatibleDMStag(dm, 0, 0, 1, 0, &cn->dm_face));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
  }
  PetscCall(DMStagSetCoordinateDMType(cn->dm_face, DMPRODUCT));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMSetCoordinateDM(cn->dm_face, cdm));
  for (d = 0; d < dim; d++) {
    PetscCall(DMCreateGlobalVector(cn->dm_face, &cn->ubar[d]));
    for (e = 0; e < dim; e++) {
      PetscCall(FlucaFDDerivativeCreate(dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, c_vel + d, face_loc[e], 0, &cn->fd_interp_vel[d][e]));
      PetscCall(NSSetVelocityBCs_Internal(ns, cn->fd_interp_vel[d][e], d));
      PetscCall(FlucaFDSetUp(cn->fd_interp_vel[d][e]));
    }
  }

  /* Viscous and pressure-gradient blocks */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDScaleCreateConstant(ns->fd_laplacian[d], 0., &cn->fd_visc[d]));
    PetscCall(NSSetVelocityBCs_Internal(ns, cn->fd_visc[d], d));
    PetscCall(FlucaFDSetUp(cn->fd_visc[d]));
    PetscCall(FlucaFDScaleCreateConstant(ns->fd_grad_p[d], 0., &cn->fd_grad[d]));
    PetscCall(FlucaFDSetUp(cn->fd_grad[d]));
  }

  /* Linearized convection: d/dx_e(ubar_d^{n+1} U_e^n + ubar_d^n ubar_e^{n+1}) */
  for (d = 0; d < dim; d++) {
    FlucaFD terms[2 * PHYS_MAX_DIM];
    FlucaFD sum;

    for (e = 0; e < dim; e++) {
      FlucaFD interp_d, interp_e, outer;

      PetscCall(FlucaFDDerivativeCreate(dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, c_vel + d, face_loc[e], c_U, &interp_d));
      PetscCall(FlucaFDSetUp(interp_d));
      PetscCall(FlucaFDScaleCreateVector(interp_d, ns->zero, c_U, &cn->fd_conv_U[d][e]));
      PetscCall(FlucaFDSetUp(cn->fd_conv_U[d][e]));
      PetscCall(FlucaFDDerivativeCreate(dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, c_vel + e, face_loc[e], c_U, &interp_e));
      PetscCall(FlucaFDSetUp(interp_e));
      PetscCall(FlucaFDScaleCreateVector(interp_e, cn->ubar[d], 0, &cn->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDSetUp(cn->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDDerivativeCreate(dm, (FlucaFDDirection)e, 1, 2, face_loc[e], c_U, DMSTAG_ELEMENT, c_vel + d, &outer));
      PetscCall(FlucaFDSetUp(outer));
      PetscCall(FlucaFDCompositionCreate(cn->fd_conv_U[d][e], outer, &terms[2 * e]));
      PetscCall(FlucaFDSetUp(terms[2 * e]));
      PetscCall(FlucaFDCompositionCreate(cn->fd_conv_ubar[d][e], outer, &terms[2 * e + 1]));
      PetscCall(FlucaFDSetUp(terms[2 * e + 1]));
      PetscCall(FlucaFDDestroy(&outer));
      PetscCall(FlucaFDDestroy(&interp_e));
      PetscCall(FlucaFDDestroy(&interp_d));
    }
    PetscCall(FlucaFDSumCreate(2 * dim, terms, &sum));
    PetscCall(FlucaFDSetUp(sum));
    PetscCall(FlucaFDScaleCreateConstant(sum, 0., &cn->fd_conv[d]));
    for (e = 0; e < dim; e++) PetscCall(NSSetVelocityBCs_Internal(ns, cn->fd_conv[d], e));
    PetscCall(FlucaFDSetUp(cn->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&sum));
    for (e = 0; e < 2 * dim; e++) PetscCall(FlucaFDDestroy(&terms[e]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The coupled system, with the operators refreshed by the step:
   momentum rows       [A  0  G  ]   A = I + (dt/2) conv + (dt/(2 rho)) (-mu lap), G = (dt/rho) grad
   Rhie-Chow rows      [-T I  -R ]   boundary-face rows reduce to U = u_b . n
   continuity rows     [0  D  0  ] */
static PetscErrorCode NSFormJacobian_CNLinear(NS ns, Vec x, Mat J)
{
  NS_CNLinear *cn = (NS_CNLinear *)ns->data;
  PetscScalar  rho;
  PetscInt     dim, c_vel, c_U, d, e;
  DM           dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetProperty(ns->phys, PHYS_PROPERTY_DENSITY, &rho));
  PetscCall(MatZeroEntries(J));
  for (d = 0; d < dim; d++) {
    PetscCall(AddIdentity_Private(dm, J, DMSTAG_ELEMENT, c_vel + d));
    PetscCall(FlucaFDGetOperator(cn->fd_visc[d], dm, dm, J));
    PetscCall(FlucaFDGetOperator(cn->fd_conv[d], dm, dm, J));
    PetscCall(FlucaFDGetOperator(cn->fd_grad[d], dm, dm, J));
  }
  for (e = 0; e < dim; e++) {
    PetscCall(AddIdentity_Private(dm, J, face_loc[e], c_U));
    PetscCall(FlucaFDGetOperator(ns->fd_negT[e], dm, dm, J));
  }
  PetscCall(AddScaledMatrix_Private(J, ns->negR, ns->dt / rho));
  PetscCall(FlucaFDGetOperator(ns->fd_D, dm, dm, J));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  /* DMCreateMatrix() lays out explicit zeros that give the default ILU sub-solves zero pivots on coarse grids */
  PetscCall(MatEliminateZeros(J, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Right-hand side of the coupled system at t = t^n:
   momentum:   u^n + (dt/(2 rho)) mu (lap u^n + b^n) - (dt/rho) grad q - (boundary parts of A at t + dt)
               + (dt/rho) f(t + dt/2), with q = p0 at the first step and p^{n-1/2} afterwards
   Rhie-Chow:  u_b . n at t + dt on the boundary faces, zero elsewhere
   continuity: zero */
static PetscErrorCode NSFormFunction_CNLinear(NS ns, Vec x, Vec f)
{
  NS_CNLinear *cn = (NS_CNLinear *)ns->data;
  PetscReal    t = ns->t, dt = ns->dt;
  PetscScalar  rho;
  PetscInt     dim, d, e;
  Vec          q, tmp, fv, uv;
  IS           is_vel;
  DM           dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetProperty(ns->phys, PHYS_PROPERTY_DENSITY, &rho));
  q = ns->step == 0 ? ns->sol0 : cn->phalf;

  PetscCall(VecZeroEntries(f));
  PetscCall(DMGetGlobalVector(dm, &tmp));
  for (d = 0; d < dim; d++) {
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(cn->fd_visc[d], t, dm, dm, ns->sol0, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(cn->fd_grad[d], t, dm, dm, q, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(cn->fd_visc[d], t + dt, dm, dm, ns->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(cn->fd_conv[d], t + dt, dm, dm, ns->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
  }
  for (e = 0; e < dim; e++) {
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ns->fd_negT[e], t + dt, dm, dm, ns->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
  }
  PetscCall(DMRestoreGlobalVector(dm, &tmp));

  PetscCall(NSGetField(ns, PHYS_FIELD_VELOCITY, &is_vel));
  PetscCall(VecGetSubVector(f, is_vel, &fv));
  PetscCall(VecGetSubVector(ns->sol0, is_vel, &uv));
  PetscCall(VecAXPY(fv, 1., uv));
  PetscCall(VecRestoreSubVector(ns->sol0, is_vel, &uv));
  PetscCall(VecRestoreSubVector(f, is_vel, &fv));

  PetscCall(AddBodyForce_Private(ns, t + dt / 2., dt / PetscRealPart(rho), f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSStep_CNLinear(NS ns)
{
  NS_CNLinear *cn = (NS_CNLinear *)ns->data;
  IS           is[2];
  IS           is_p;
  Vec          xs, ss, dp, ph, p0;
  PetscInt     k;

  PetscFunctionBegin;
  PetscCall(UpdateOperators_Private(ns));
  PetscCall(SNESSolve(ns->snes, NULL, ns->x));
  PetscCall(NSCheckDiverged(ns));
  if (ns->reason < 0) PetscFunctionReturn(PETSC_SUCCESS);

  /* u^{n+1} and U^{n+1} */
  PetscCall(NSGetField(ns, PHYS_FIELD_VELOCITY, &is[0]));
  PetscCall(NSGetField(ns, PHYS_FIELD_FACE_VELOCITY, &is[1]));
  for (k = 0; k < 2; ++k) {
    PetscCall(VecGetSubVector(ns->x, is[k], &xs));
    PetscCall(VecGetSubVector(ns->sol, is[k], &ss));
    PetscCall(VecCopy(xs, ss));
    PetscCall(VecRestoreSubVector(ns->sol, is[k], &ss));
    PetscCall(VecRestoreSubVector(ns->x, is[k], &xs));
  }

  /* p^{n+1/2} = q + p' is kept in phalf; the solution holds the extrapolated p^{n+1} */
  PetscCall(NSGetField(ns, PHYS_FIELD_PRESSURE, &is_p));
  PetscCall(VecGetSubVector(ns->x, is_p, &dp));
  PetscCall(VecGetSubVector(ns->sol, is_p, &ss));
  PetscCall(VecGetSubVector(cn->phalf, is_p, &ph));
  if (ns->step == 0) {
    PetscCall(VecGetSubVector(ns->sol0, is_p, &p0));
    PetscCall(VecWAXPY(ss, 2., dp, p0));
    PetscCall(VecWAXPY(ph, 1., dp, p0));
    PetscCall(VecRestoreSubVector(ns->sol0, is_p, &p0));
  } else {
    PetscCall(VecWAXPY(ss, 1.5, dp, ph));
    PetscCall(VecAXPY(ph, 1., dp));
  }
  PetscCall(VecRestoreSubVector(cn->phalf, is_p, &ph));
  PetscCall(VecRestoreSubVector(ns->sol, is_p, &ss));
  PetscCall(VecRestoreSubVector(ns->x, is_p, &dp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSDestroy_CNLinear(NS ns)
{
  NS_CNLinear *cn = (NS_CNLinear *)ns->data;
  PetscInt     d, e;

  PetscFunctionBegin;
  for (d = 0; d < PHYS_MAX_DIM; d++) {
    PetscCall(FlucaFDDestroy(&cn->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&cn->fd_grad[d]));
    PetscCall(FlucaFDDestroy(&cn->fd_visc[d]));
    for (e = 0; e < PHYS_MAX_DIM; e++) {
      PetscCall(FlucaFDDestroy(&cn->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDDestroy(&cn->fd_conv_U[d][e]));
      PetscCall(FlucaFDDestroy(&cn->fd_interp_vel[d][e]));
    }
    PetscCall(VecDestroy(&cn->ubar[d]));
  }
  PetscCall(DMDestroy(&cn->dm_face));
  PetscCall(VecDestroy(&cn->phalf));
  PetscCall(PetscFree(ns->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCreate_CNLinear(NS ns)
{
  NS_CNLinear *cn;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cn));
  ns->data = (void *)cn;

  ns->ops->setup        = NSSetUp_CNLinear;
  ns->ops->step         = NSStep_CNLinear;
  ns->ops->formjacobian = NSFormJacobian_CNLinear;
  ns->ops->formfunction = NSFormFunction_CNLinear;
  ns->ops->destroy      = NSDestroy_CNLinear;
  PetscFunctionReturn(PETSC_SUCCESS);
}
