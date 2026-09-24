#include <fluca/private/segcnlinearimpl.h>

/* Operators of the momentum rows: A = I + (dt/2) J - (dt/2) nu lap and G = (dt/rho) grad, built on
   the spatial operators in cn->sops. Coefficients depending on dt and the linearization state are
   set per step. */
static PetscErrorCode BuildMomentumOperators_Private(Seg seg)
{
  Seg_CNLinear  *cn   = (Seg_CNLinear *)seg->data;
  SegSpatialOps *sops = &cn->sops;
  PetscInt       dim  = sops->dim, d, e;
  DM             sol_dm, cdm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(seg->phys, &sol_dm));

  /* ubar_d^n lives on a one-DOF-per-face DM */
  switch (dim) {
  case 2:
    PetscCall(DMStagCreateCompatibleDMStag(sol_dm, 0, 1, 0, 0, &cn->dm_face));
    break;
  case 3:
    PetscCall(DMStagCreateCompatibleDMStag(sol_dm, 0, 0, 1, 0, &cn->dm_face));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)seg), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
  }
  PetscCall(DMStagSetCoordinateDMType(cn->dm_face, DMPRODUCT));
  PetscCall(DMGetCoordinateDM(sol_dm, &cdm));
  PetscCall(DMSetCoordinateDM(cn->dm_face, cdm));
  for (d = 0; d < dim; d++) PetscCall(DMCreateGlobalVector(cn->dm_face, &cn->ubar[d]));

  /* Viscous and pressure-gradient blocks */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDScaleCreateConstant(sops->fd_laplacian[d], 0., &cn->fd_visc[d]));
    PetscCall(SegSpatialOpsSetVelocityBCs_Internal(seg->phys, sops, cn->fd_visc[d], d));
    PetscCall(FlucaFDSetUp(cn->fd_visc[d]));
    PetscCall(FlucaFDScaleCreateConstant(sops->fd_grad_p[d], 0., &cn->fd_grad[d]));
    PetscCall(FlucaFDSetUp(cn->fd_grad[d]));
  }

  /* Linearized convection, guide eq. (5) and section Spatial Discretization:
     d/dx_e(ubar_d^{n+1} U_e^n + ubar_d^n ubar_e^{n+1}) */
  for (d = 0; d < dim; d++) {
    FlucaFD terms[2 * FLUCA_MAX_DIM], sum;

    for (e = 0; e < dim; e++) {
      FlucaFD interp_d, interp_e, outer;

      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, sops->c_vel + d, face_loc[e], sops->c_U, &interp_d));
      PetscCall(FlucaFDSetUp(interp_d));
      PetscCall(FlucaFDScaleCreateVector(interp_d, sops->zero, sops->c_U, &cn->fd_conv_U[d][e]));
      PetscCall(FlucaFDSetUp(cn->fd_conv_U[d][e]));
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, sops->c_vel + e, face_loc[e], sops->c_U, &interp_e));
      PetscCall(FlucaFDSetUp(interp_e));
      PetscCall(FlucaFDScaleCreateVector(interp_e, cn->ubar[d], 0, &cn->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDSetUp(cn->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], sops->c_U, DMSTAG_ELEMENT, sops->c_vel + d, &outer));
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
    for (e = 0; e < dim; e++) PetscCall(SegSpatialOpsSetVelocityBCs_Internal(seg->phys, sops, cn->fd_conv[d], e));
    PetscCall(FlucaFDSetUp(cn->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&sum));
    for (e = 0; e < 2 * dim; e++) PetscCall(FlucaFDDestroy(&terms[e]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* M and f of eq. (13): momentum rows from the state at t^n, coupling rows with boundary data at t_coupling */
static PetscErrorCode SegCNLinearAssembleSystem_Private(Seg seg, PetscReal t_coupling)
{
  Seg_CNLinear *cn = (Seg_CNLinear *)seg->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(cn->M));
  PetscCall(VecZeroEntries(cn->f));
  PetscCall(SegCNLinearComputeMomentumSystem_Internal(seg, seg->t, seg->dt, seg->sol, cn->M, cn->f));
  PetscCall(SegCNLinearComputeCouplingSystem_Internal(seg, t_coupling, seg->dt, cn->M, cn->f));
  PetscCall(MatAssemblyBegin(cn->M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(cn->M, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Replace the face velocity of the initial state by the discretely divergence-free projection of
   T u0 + b_interp: U0 = U* - G^st phi with D G^st phi = D U* - b_cont. u0 and p0 are unchanged. */
static PetscErrorCode SegPreSolve_CNLinear(Seg seg)
{
  Seg_CNLinear      *cn   = (Seg_CNLinear *)seg->data;
  IS                 is_u = seg->fields[SEG_CNLINEAR_FIELD_VELOCITY].is;
  IS                 is_U = seg->fields[SEG_CNLINEAR_FIELD_FACE_VELOCITY].is;
  IS                 is_p = seg->fields[SEG_CNLINEAR_FIELD_PRESSURE].is;
  MPI_Comm           comm;
  Mat                negT, G, negR, D, W, S;
  Vec                u, U, fU, fp, Ustar, phi, rhs;
  MatNullSpace       nullspace;
  KSP                ksp;
  PC                 pc;
  KSPConvergedReason reason;
  const char        *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)seg, &comm));
  PetscCall(SegCNLinearAssembleSystem_Private(seg, seg->t));
  PetscCall(MatCreateSubMatrix(cn->M, is_U, is_u, MAT_INITIAL_MATRIX, &negT));
  PetscCall(MatCreateSubMatrix(cn->M, is_u, is_p, MAT_INITIAL_MATRIX, &G));
  PetscCall(MatCreateSubMatrix(cn->M, is_U, is_p, MAT_INITIAL_MATRIX, &negR));
  PetscCall(MatCreateSubMatrix(cn->M, is_p, is_U, MAT_INITIAL_MATRIX, &D));

  /* W = (-T) G - (-R) = -G^st, S = D W: the Schur complement of eq. (18) */
  PetscCall(MatMatMult(negT, G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &W));
  PetscCall(MatAXPY(W, -1., negR, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatMatMult(D, W, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &S));
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(S, nullspace));

  /* U* = T u0 + b_interp */
  PetscCall(MatCreateVecs(negT, NULL, &Ustar));
  PetscCall(VecGetSubVector(seg->sol, is_u, &u));
  PetscCall(VecGetSubVector(cn->f, is_U, &fU));
  PetscCall(MatMult(negT, u, Ustar));
  PetscCall(VecAYPX(Ustar, -1., fU));
  PetscCall(VecRestoreSubVector(cn->f, is_U, &fU));
  PetscCall(VecRestoreSubVector(seg->sol, is_u, &u));

  /* S phi = b_cont - D U* */
  PetscCall(MatCreateVecs(S, &phi, &rhs));
  PetscCall(VecGetSubVector(cn->f, is_p, &fp));
  PetscCall(MatMult(D, Ustar, rhs));
  PetscCall(VecAYPX(rhs, -1., fp));
  PetscCall(VecRestoreSubVector(cn->f, is_p, &fp));
  PetscCall(MatNullSpaceRemove(nullspace, rhs));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)ksp, (PetscObject)seg, 1));
  PetscCall(SegGetOptionsPrefix(seg, &prefix));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(ksp, "seg_init_"));
  PetscCall(KSPSetOperators(ksp, S, S));
  PetscCall(KSPSetTolerances(ksp, 1.e-12, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCGAMG));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(VecZeroEntries(phi));
  PetscCall(KSPSolve(ksp, rhs, phi));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCheck(reason > 0, comm, PETSC_ERR_NOT_CONVERGED, "Initial face velocity projection did not converge: %s", KSPConvergedReasons[reason]);

  /* U0 = U* + W phi */
  PetscCall(VecGetSubVector(seg->sol, is_U, &U));
  PetscCall(MatMult(W, phi, U));
  PetscCall(VecAXPY(U, 1., Ustar));
  PetscCall(VecRestoreSubVector(seg->sol, is_U, &U));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&phi));
  PetscCall(VecDestroy(&Ustar));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&negR));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&negT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegSetUp_CNLinear(Seg seg)
{
  Seg_CNLinear        *cn                               = (Seg_CNLinear *)seg->data;
  const SegFieldUpdate updates[SEG_CNLINEAR_NUM_FIELDS] = {
    [SEG_CNLINEAR_FIELD_VELOCITY]      = SEG_FIELD_UPDATE_VALUE,
    [SEG_CNLINEAR_FIELD_FACE_VELOCITY] = SEG_FIELD_UPDATE_VALUE,
    [SEG_CNLINEAR_FIELD_PRESSURE]      = SEG_FIELD_UPDATE_INCREMENT,
  };
  const char *names[SEG_CNLINEAR_NUM_FIELDS] = {
    [SEG_CNLINEAR_FIELD_VELOCITY]      = PHYS_FIELD_VELOCITY,
    [SEG_CNLINEAR_FIELD_FACE_VELOCITY] = PHYS_FIELD_FACE_VELOCITY,
    [SEG_CNLINEAR_FIELD_PRESSURE]      = PHYS_FIELD_PRESSURE,
  };
  MPI_Comm          comm;
  DM                dm;
  Vec               nullvecs[SEG_MAX_FIELDS];
  Vec               sub;
  IS                is[SEG_CNLINEAR_NUM_FIELDS];
  Mat               blocks[SEG_CNLINEAR_NUM_FIELDS * SEG_CNLINEAR_NUM_FIELDS];
  KSP               ksp, kspA, kspS;
  PC                pc, subpc;
  PetscBool         setupcalled, isconst;
  PetscInt          k, nnull, n, N, ncomp;
  PhysFieldLocation loc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)seg, &comm));
  PetscCall(PhysGetSetUpCalled(seg->phys, &setupcalled));
  PetscCheck(setupcalled, comm, PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before SegSetUp() with SEGCNLINEAR");
  PetscCall(SegSpatialOpsBuild_Internal(seg->phys, &cn->sops));
  PetscCall(BuildMomentumOperators_Private(seg));
  PetscCall(PhysGetSolutionDM(seg->phys, &dm));

  /* The row blocks of the coupled system, in the order a step writes them back: the velocity and
     the face velocity rows solve for the new value, the pressure rows for a correction. */
  PetscCheck(SEG_CNLINEAR_NUM_FIELDS <= SEG_MAX_FIELDS, comm, PETSC_ERR_SUP, "SegCNLinear needs %d fields but Seg holds at most %d", SEG_CNLINEAR_NUM_FIELDS, SEG_MAX_FIELDS);
  seg->nfields = SEG_CNLINEAR_NUM_FIELDS;
  for (k = 0; k < SEG_CNLINEAR_NUM_FIELDS; ++k) {
    seg->fields[k].name   = names[k];
    seg->fields[k].update = updates[k];
    PetscCall(PhysGetFieldIS(seg->phys, names[k], &seg->fields[k].is));
  }

  PetscCall(DMCreateMatrix(dm, &cn->M));
  PetscCall(MatSetOption(cn->M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(cn->M, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &cn->f));
  PetscCall(DMCreateGlobalVector(dm, &cn->x));

  /* Each field the Phys declares as determined only up to a constant contributes one null-space
     vector: constant on that field's entries and zero on every other. Deriving a single vector this
     way is only valid for a single-component element field, since a face field or a field with more
     than one component needs one null-space dimension per component. Their supports are disjoint,
     so normalising each one on its own entries makes the set orthonormal, as MatNullSpaceCreate()
     requires. For PHYSLAMINAR the pressure is the only such field. */
  nnull = 0;
  for (k = 0; k < seg->nfields; ++k) {
    PetscCall(PhysGetFieldNullSpaceConstant(seg->phys, seg->fields[k].name, &isconst));
    if (!isconst) continue;
    PetscCall(PhysGetField(seg->phys, seg->fields[k].name, &loc, NULL, &ncomp));
    PetscCheck(loc == PHYS_FIELD_ELEMENT && ncomp == 1, PetscObjectComm((PetscObject)seg), PETSC_ERR_SUP, "Field %s declares a constant null space, but deriving one requires a single element-located component; it has %" PetscInt_FMT " component(s) at %s",
               seg->fields[k].name, ncomp, PhysFieldLocations[loc]);
    PetscCall(DMCreateGlobalVector(dm, &nullvecs[nnull]));
    PetscCall(VecZeroEntries(nullvecs[nnull]));
    PetscCall(VecGetSubVector(nullvecs[nnull], seg->fields[k].is, &sub));
    PetscCall(VecGetSize(sub, &N));
    PetscCall(VecSet(sub, 1. / PetscSqrtReal((PetscReal)N)));
    PetscCall(VecRestoreSubVector(nullvecs[nnull], seg->fields[k].is, &sub));
    ++nnull;
  }
  if (nnull > 0) {
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, nnull, nullvecs, &cn->nullspace));
    PetscCall(MatSetNullSpace(cn->M, cn->nullspace));
  }
  for (k = 0; k < nnull; ++k) PetscCall(VecDestroy(&nullvecs[k]));

  /* PCABF reads the field index sets from a MATNEST preconditioning matrix and the blocks from M */
  for (k = 0; k < SEG_CNLINEAR_NUM_FIELDS * SEG_CNLINEAR_NUM_FIELDS; ++k) blocks[k] = NULL;
  for (k = 0; k < SEG_CNLINEAR_NUM_FIELDS; ++k) {
    is[k] = seg->fields[k].is;
    PetscCall(ISGetLocalSize(is[k], &n));
    PetscCall(ISGetSize(is[k], &N));
    PetscCall(MatCreateConstantDiagonal(comm, n, n, N, N, 1., &blocks[(SEG_CNLINEAR_NUM_FIELDS + 1) * k]));
  }
  PetscCall(MatCreateNest(comm, SEG_CNLINEAR_NUM_FIELDS, is, SEG_CNLINEAR_NUM_FIELDS, is, blocks, &cn->P));
  for (k = 0; k < SEG_CNLINEAR_NUM_FIELDS; ++k) PetscCall(MatDestroy(&blocks[(SEG_CNLINEAR_NUM_FIELDS + 1) * k]));

  PetscCall(SegGetKSP(seg, &ksp));
  PetscCall(KSPSetOperators(ksp, cn->M, cn->P));
  PetscCall(KSPSetType(ksp, KSPRICHARDSON));
  PetscCall(KSPSetTolerances(ksp, 1.e-8, PETSC_CURRENT, PETSC_CURRENT, 1000));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCABF));
  PetscCall(PCABFSetFields(pc, 0, 1, 2));
  PetscCall(PCABFSetSchurComplementAinvType(pc, PC_ABF_AINV_ID));
  PetscCall(PCABFSetUpperTriangularAinvType(pc, PC_ABF_AINV_ID));
  PetscCall(PCABFGetSubKSPs(pc, &kspA, &kspS));
  PetscCall(KSPSetTolerances(kspA, 1.e-10, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPGetPC(kspA, &subpc));
  PetscCall(PCSetType(subpc, PCJACOBI));
  PetscCall(KSPSetTolerances(kspS, 1.e-10, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPGetPC(kspS, &subpc));
  PetscCall(PCSetType(subpc, PCGAMG));
  PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegStep_CNLinear(Seg seg)
{
  Seg_CNLinear      *cn = (Seg_CNLinear *)seg->data;
  KSP                ksp;
  KSPConvergedReason reason;
  Vec                xs, Xs;
  PetscInt           f;

  PetscFunctionBegin;
  PetscCall(SegGetKSP(seg, &ksp));
  PetscCall(SegCNLinearAssembleSystem_Private(seg, seg->t + seg->dt));
  /* PCABF takes its blocks from M; mark P changed so that the preconditioner is rebuilt */
  PetscCall(PetscObjectStateIncrease((PetscObject)cn->P));
  PetscCall(VecZeroEntries(cn->x));
  PetscCall(KSPSolve(ksp, cn->f, cn->x));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (reason == KSP_DIVERGED_ITS) {
    PetscInt max_it;

    /* Stopping at the iteration limit is an accepted outcome only for a single sweep
       (-seg_ksp_max_it 1, the classic fractional step method), which ends there by design.
       With any other limit the requested tolerance was simply not reached, so the step is
       rejected like any other solve failure. */
    PetscCall(KSPGetTolerances(ksp, NULL, NULL, NULL, &max_it));
    if (max_it != 1) {
      PetscCall(PetscInfo(seg, "Step=%" PetscInt_FMT ", coupled solve stopped at the iteration limit %" PetscInt_FMT " before reaching the requested tolerance\n", seg->step, max_it));
      seg->reason = SEG_DIVERGED_LINEAR_SOLVE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else if (reason < 0) {
    PetscCall(PetscInfo(seg, "Step=%" PetscInt_FMT ", coupled solve failed: %s\n", seg->step, KSPConvergedReasons[reason]));
    seg->reason = SEG_DIVERGED_LINEAR_SOLVE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* X <- [u^{n+1}, U^{n+1}, q + p'], each field updated the way it was declared */
  for (f = 0; f < seg->nfields; ++f) {
    PetscCall(VecGetSubVector(cn->x, seg->fields[f].is, &xs));
    PetscCall(VecGetSubVector(seg->sol, seg->fields[f].is, &Xs));
    if (seg->fields[f].update == SEG_FIELD_UPDATE_VALUE) PetscCall(VecCopy(xs, Xs));
    else if (seg->fields[f].update == SEG_FIELD_UPDATE_INCREMENT) PetscCall(VecAXPY(Xs, 1., xs));
    else SETERRQ(PetscObjectComm((PetscObject)seg), PETSC_ERR_SUP, "Unsupported field update mode %d", (int)seg->fields[f].update);
    PetscCall(VecRestoreSubVector(seg->sol, seg->fields[f].is, &Xs));
    PetscCall(VecRestoreSubVector(cn->x, seg->fields[f].is, &xs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegView_CNLinear(Seg seg, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (seg->ksp) PetscCall(KSPView(seg->ksp, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegDestroy_CNLinear(Seg seg)
{
  Seg_CNLinear *cn = (Seg_CNLinear *)seg->data;
  PetscInt      d, e;

  PetscFunctionBegin;
  for (d = 0; d < FLUCA_MAX_DIM; d++) {
    PetscCall(FlucaFDDestroy(&cn->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&cn->fd_grad[d]));
    PetscCall(FlucaFDDestroy(&cn->fd_visc[d]));
    for (e = 0; e < FLUCA_MAX_DIM; e++) {
      PetscCall(FlucaFDDestroy(&cn->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDDestroy(&cn->fd_conv_U[d][e]));
    }
    PetscCall(VecDestroy(&cn->ubar[d]));
  }
  PetscCall(DMDestroy(&cn->dm_face));
  PetscCall(SegSpatialOpsDestroy_Internal(&cn->sops));
  PetscCall(MatDestroy(&cn->P));
  PetscCall(MatNullSpaceDestroy(&cn->nullspace));
  PetscCall(MatDestroy(&cn->M));
  PetscCall(VecDestroy(&cn->x));
  PetscCall(VecDestroy(&cn->f));
  PetscCall(PetscFree(seg->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegCreate_CNLinear(Seg seg)
{
  Seg_CNLinear *cn;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cn));
  seg->data = (void *)cn;

  cn->M         = NULL;
  cn->P         = NULL;
  cn->nullspace = NULL;
  cn->f         = NULL;
  cn->x         = NULL;

  seg->ops->setup    = SegSetUp_CNLinear;
  seg->ops->presolve = SegPreSolve_CNLinear;
  seg->ops->step     = SegStep_CNLinear;
  seg->ops->destroy  = SegDestroy_CNLinear;
  seg->ops->view     = SegView_CNLinear;
  PetscFunctionReturn(PETSC_SUCCESS);
}
