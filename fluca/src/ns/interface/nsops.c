#include <fluca/private/nsimpl.h>

static PetscErrorCode BCAdapterFn_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  NS_BCAdapter *a = (NS_BCAdapter *)ctx;

  PetscFunctionBegin;
  PetscCall(a->fn(dim, t, x, a->comp, value, a->fn_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BCAdapterFnDot_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  NS_BCAdapter *a = (NS_BCAdapter *)ctx;

  PetscFunctionBegin;
  PetscCall(a->fn_dot(dim, t, x, a->comp, value, a->fn_dot_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Velocity Dirichlet BCs of velocity component d, from the Phys, on fd */
PetscErrorCode NSSetVelocityBCs_Internal(NS ns, FlucaFD fd, PetscInt d)
{
  FlucaFDBoundaryCondition bcs[PHYS_MAX_FACES] = {{0}};
  PhysBC                   bc;
  PetscInt                 dim, c_vel, f;
  DM                       dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  for (f = 0; f < 2 * dim; f++) {
    PetscCall(PhysGetBoundaryCondition(ns->phys, f, &bc));
    if (bc.type != PHYS_BC_VELOCITY) continue;
    bcs[f].type = FLUCAFD_BC_DIRICHLET;
    if (!bc.fn) {
      bcs[f].value = 0.;
      continue;
    }
    ns->bcadapters[d][f].fn         = bc.fn;
    ns->bcadapters[d][f].fn_dot     = bc.fn_dot;
    ns->bcadapters[d][f].fn_ctx     = bc.ctx;
    ns->bcadapters[d][f].fn_dot_ctx = bc.fn_dot_ctx;
    ns->bcadapters[d][f].comp       = d;
    bcs[f].fn                       = BCAdapterFn_Private;
    bcs[f].fn_ctx                   = &ns->bcadapters[d][f];
    bcs[f].fn_dot                   = bc.fn_dot ? BCAdapterFnDot_Private : NULL;
    bcs[f].fn_dot_ctx               = &ns->bcadapters[d][f];
  }
  PetscCall(FlucaFDSetBoundaryConditions(fd, c_vel + d, bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -R = (-T) G_c + G^st as an explicit matrix. G^st is the face-normal pressure derivative with a
   homogeneous Neumann BC on every velocity face, so its boundary-face rows are empty; the two-point
   -T has empty boundary-face rows too (its whole weight is on the boundary datum). */
static PetscErrorCode CreateNegR_Private(NS ns)
{
  FlucaFDBoundaryCondition pbcs[PHYS_MAX_FACES] = {{0}};
  PhysBC                   bc;
  PetscInt                 dim, c_p, c_U, e, f;
  Mat                      negTmat, Gmat, Gstmat;
  FlucaFD                  Gst;
  DM                       dm;
  Mesh                     mesh;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(PhysGetMesh(ns->phys, &mesh));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  for (f = 0; f < 2 * dim; f++) {
    PetscCall(PhysGetBoundaryCondition(ns->phys, f, &bc));
    if (bc.type != PHYS_BC_VELOCITY) continue;
    pbcs[f].type  = FLUCAFD_BC_NEUMANN;
    pbcs[f].value = 0.;
  }
  PetscCall(DMCreateMatrix(dm, &negTmat));
  PetscCall(DMCreateMatrix(dm, &Gmat));
  PetscCall(DMCreateMatrix(dm, &Gstmat));
  PetscCall(MatSetOption(negTmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(Gmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(Gstmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  for (e = 0; e < dim; e++) {
    PetscCall(FlucaFDGetOperator(ns->fd_negT[e], dm, dm, negTmat));
    PetscCall(FlucaFDGetOperator(ns->fd_grad_p[e], dm, dm, Gmat));
    PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, c_p, face_loc[e], c_U, &Gst));
    PetscCall(FlucaFDSetBoundaryConditions(Gst, c_p, pbcs));
    PetscCall(FlucaFDSetUp(Gst));
    PetscCall(FlucaFDGetOperator(Gst, dm, dm, Gstmat));
    PetscCall(FlucaFDDestroy(&Gst));
  }
  PetscCall(MatAssemblyBegin(negTmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(negTmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Gmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Gstmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatMatMult(negTmat, Gmat, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &ns->negR));
  PetscCall(MatAXPY(ns->negR, 1., Gstmat, DIFFERENT_NONZERO_PATTERN));
  /* DMCreateMatrix() lays out explicit zeros; drop them so negR stays inside the star stencil */
  PetscCall(MatEliminateZeros(ns->negR, PETSC_FALSE));
  PetscCall(MatDestroy(&Gstmat));
  PetscCall(MatDestroy(&Gmat));
  PetscCall(MatDestroy(&negTmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Operators of the continuous equations on the Phys solution DM; none carries a time step */
PetscErrorCode NSSetUpSpatialOperators_Internal(NS ns)
{
  PetscInt    dim, c_vel, c_U, c_p, d, e;
  PetscScalar mu;
  FlucaFD     ops[PHYS_MAX_DIM];
  DM          dm;
  Mesh        mesh;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(PhysGetMesh(ns->phys, &mesh));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(PhysGetProperty(ns->phys, PHYS_PROPERTY_VISCOSITY, &mu));

  /* fd_laplacian[d] = sum_e d/dx_e(-mu d(u_d)/dx_e); fd_negmu[d][e] is kept so the step can refresh mu */
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) {
      FlucaFD inner, outer;

      PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, c_vel + d, face_loc[e], c_U, &inner));
      PetscCall(FlucaFDSetUp(inner));
      PetscCall(FlucaFDScaleCreateConstant(inner, -mu, &ns->fd_negmu[d][e]));
      PetscCall(FlucaFDSetUp(ns->fd_negmu[d][e]));
      PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 1, 2, face_loc[e], c_U, DMSTAG_ELEMENT, c_vel + d, &outer));
      PetscCall(FlucaFDSetUp(outer));
      PetscCall(FlucaFDCompositionCreate(ns->fd_negmu[d][e], outer, &ops[e]));
      PetscCall(FlucaFDSetUp(ops[e]));
      PetscCall(FlucaFDDestroy(&outer));
      PetscCall(FlucaFDDestroy(&inner));
    }
    PetscCall(FlucaFDSumCreate(dim, ops, &ns->fd_laplacian[d]));
    PetscCall(NSSetVelocityBCs_Internal(ns, ns->fd_laplacian[d], d));
    PetscCall(FlucaFDSetUp(ns->fd_laplacian[d]));
    for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&ops[e]));
  }

  /* fd_grad_p[d] = dp/dx_d; no BC, so FlucaFD closes it one-sided at walls */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, c_p, DMSTAG_ELEMENT, c_vel + d, &ns->fd_grad_p[d]));
    PetscCall(FlucaFDSetUp(ns->fd_grad_p[d]));
  }

  /* fd_negT[e] = -(two-point interpolation of u_e to the faces normal to e) */
  for (e = 0; e < dim; e++) {
    FlucaFD T;

    PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, c_vel + e, face_loc[e], c_U, &T));
    PetscCall(FlucaFDSetUp(T));
    PetscCall(FlucaFDScaleCreateConstant(T, -1., &ns->fd_negT[e]));
    PetscCall(NSSetVelocityBCs_Internal(ns, ns->fd_negT[e], e));
    PetscCall(FlucaFDSetUp(ns->fd_negT[e]));
    PetscCall(FlucaFDDestroy(&T));
  }

  /* fd_D = sum_e d/dx_e(U_e) */
  for (e = 0; e < dim; e++) {
    PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 1, 2, face_loc[e], c_U, DMSTAG_ELEMENT, c_p, &ops[e]));
    PetscCall(FlucaFDSetUp(ops[e]));
  }
  PetscCall(FlucaFDSumCreate(dim, ops, &ns->fd_D));
  PetscCall(FlucaFDSetUp(ns->fd_D));
  for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&ops[e]));

  PetscCall(CreateNegR_Private(ns));

  PetscCall(DMCreateGlobalVector(dm, &ns->zero));
  PetscCall(VecZeroEntries(ns->zero));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSDestroySpatialOperators_Internal(NS ns)
{
  PetscInt d, e;

  PetscFunctionBegin;
  for (d = 0; d < PHYS_MAX_DIM; d++) {
    for (e = 0; e < PHYS_MAX_DIM; e++) PetscCall(FlucaFDDestroy(&ns->fd_negmu[d][e]));
    PetscCall(FlucaFDDestroy(&ns->fd_laplacian[d]));
    PetscCall(FlucaFDDestroy(&ns->fd_grad_p[d]));
    PetscCall(FlucaFDDestroy(&ns->fd_negT[d]));
  }
  PetscCall(FlucaFDDestroy(&ns->fd_D));
  PetscCall(MatDestroy(&ns->negR));
  PetscCall(VecDestroy(&ns->zero));
  PetscFunctionReturn(PETSC_SUCCESS);
}
