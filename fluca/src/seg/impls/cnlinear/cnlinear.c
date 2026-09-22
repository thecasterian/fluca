#include <fluca/private/segcnlinearimpl.h>

/* M and f of eq. (13): momentum rows from the state at t^n, coupling rows with boundary data at t_coupling */
static PetscErrorCode SegCNLinearAssembleSystem_Private(Seg seg, PetscReal t_coupling)
{
  Seg_CNLinear *fsm = (Seg_CNLinear *)seg->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(fsm->M));
  PetscCall(VecZeroEntries(fsm->f));
  PetscCall(SegCNLinearComputeMomentumSystem_Internal(seg, seg->t, seg->dt, seg->sol, fsm->M, fsm->f));
  PetscCall(SegCNLinearComputeCouplingSystem_Internal(seg, t_coupling, seg->dt, fsm->M, fsm->f));
  PetscCall(MatAssemblyBegin(fsm->M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(fsm->M, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Replace the face velocity of the initial state by the discretely divergence-free projection of
   T u0 + b_interp: U0 = U* - G^st phi with D G^st phi = D U* - b_cont. u0 and p0 are unchanged. */
static PetscErrorCode SegPreSolve_CNLinear(Seg seg)
{
  Seg_CNLinear      *fsm  = (Seg_CNLinear *)seg->data;
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
  PetscCall(MatCreateSubMatrix(fsm->M, is_U, is_u, MAT_INITIAL_MATRIX, &negT));
  PetscCall(MatCreateSubMatrix(fsm->M, is_u, is_p, MAT_INITIAL_MATRIX, &G));
  PetscCall(MatCreateSubMatrix(fsm->M, is_U, is_p, MAT_INITIAL_MATRIX, &negR));
  PetscCall(MatCreateSubMatrix(fsm->M, is_p, is_U, MAT_INITIAL_MATRIX, &D));

  /* W = (-T) G - (-R) = -G^st, S = D W: the Schur complement of eq. (18) */
  PetscCall(MatMatMult(negT, G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &W));
  PetscCall(MatAXPY(W, -1., negR, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatMatMult(D, W, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &S));
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(S, nullspace));

  /* U* = T u0 + b_interp */
  PetscCall(MatCreateVecs(negT, NULL, &Ustar));
  PetscCall(VecGetSubVector(seg->sol, is_u, &u));
  PetscCall(VecGetSubVector(fsm->f, is_U, &fU));
  PetscCall(MatMult(negT, u, Ustar));
  PetscCall(VecAYPX(Ustar, -1., fU));
  PetscCall(VecRestoreSubVector(fsm->f, is_U, &fU));
  PetscCall(VecRestoreSubVector(seg->sol, is_u, &u));

  /* S phi = b_cont - D U* */
  PetscCall(MatCreateVecs(S, &phi, &rhs));
  PetscCall(VecGetSubVector(fsm->f, is_p, &fp));
  PetscCall(MatMult(D, Ustar, rhs));
  PetscCall(VecAYPX(rhs, -1., fp));
  PetscCall(VecRestoreSubVector(fsm->f, is_p, &fp));
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
  Seg_CNLinear        *fsm                              = (Seg_CNLinear *)seg->data;
  const SegFieldUpdate updates[SEG_CNLINEAR_NUM_FIELDS] = {SEG_FIELD_UPDATE_VALUE, SEG_FIELD_UPDATE_VALUE, SEG_FIELD_UPDATE_INCREMENT};
  const char          *names[SEG_CNLINEAR_NUM_FIELDS]   = {PHYS_FIELD_VELOCITY, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_PRESSURE};
  MPI_Comm             comm;
  DM                   dm;
  Vec                  nullvecs[SEG_MAX_FIELDS];
  Vec                  sub;
  IS                   is[SEG_CNLINEAR_NUM_FIELDS];
  Mat                  blocks[SEG_CNLINEAR_NUM_FIELDS * SEG_CNLINEAR_NUM_FIELDS];
  KSP                  ksp, kspA, kspS;
  PC                   pc, subpc;
  PetscBool            setupcalled, isconst;
  PetscInt             k, nnull, n, N;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)seg, &comm));
  /* Two-phase setup: the Phys has frozen its own declarations, so the face velocity that the
     fractional step method needs can be added here before the solution DM is laid out. */
  PetscCall(PhysGetSetUpCalled(seg->phys, &setupcalled));
  PetscCheck(setupcalled, comm, PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before SegSetUp() with SEGCNLINEAR");
  PetscCall(PhysDeclareField(seg->phys, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_FACE, 1, PHYS_EQN_AUXILIARY));
  PetscCall(PhysCreateSolutionDM(seg->phys));
  PetscCall(SegOpsBuild_Internal(seg));
  PetscCall(PhysGetSolutionDM(seg->phys, &dm));

  /* The row blocks of the coupled system, in the order a step writes them back: the velocity and
     the face velocity rows solve for the new value, the pressure rows for a correction. */
  seg->nfields = SEG_CNLINEAR_NUM_FIELDS;
  for (k = 0; k < SEG_CNLINEAR_NUM_FIELDS; ++k) {
    seg->fields[k].name   = names[k];
    seg->fields[k].update = updates[k];
    PetscCall(PhysGetFieldIS(seg->phys, names[k], &seg->fields[k].is));
  }

  PetscCall(DMCreateMatrix(dm, &fsm->M));
  PetscCall(MatSetOption(fsm->M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(fsm->M, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &fsm->f));
  PetscCall(DMCreateGlobalVector(dm, &fsm->x));

  /* Every field the Phys declares as determined only up to a constant contributes one null-space
     vector: constant on that field's entries and zero on every other. Their supports are disjoint,
     so normalising each one on its own entries makes the set orthonormal, as MatNullSpaceCreate()
     requires. For PHYSLAMINAR the pressure is the only such field. */
  nnull = 0;
  for (k = 0; k < seg->nfields; ++k) {
    PetscCall(PhysGetFieldNullSpaceConstant(seg->phys, seg->fields[k].name, &isconst));
    if (!isconst) continue;
    PetscCall(DMCreateGlobalVector(dm, &nullvecs[nnull]));
    PetscCall(VecZeroEntries(nullvecs[nnull]));
    PetscCall(VecGetSubVector(nullvecs[nnull], seg->fields[k].is, &sub));
    PetscCall(VecGetSize(sub, &N));
    PetscCall(VecSet(sub, 1. / PetscSqrtReal((PetscReal)N)));
    PetscCall(VecRestoreSubVector(nullvecs[nnull], seg->fields[k].is, &sub));
    ++nnull;
  }
  if (nnull > 0) {
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, nnull, nullvecs, &fsm->nullspace));
    PetscCall(MatSetNullSpace(fsm->M, fsm->nullspace));
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
  PetscCall(MatCreateNest(comm, SEG_CNLINEAR_NUM_FIELDS, is, SEG_CNLINEAR_NUM_FIELDS, is, blocks, &fsm->P));
  for (k = 0; k < SEG_CNLINEAR_NUM_FIELDS; ++k) PetscCall(MatDestroy(&blocks[(SEG_CNLINEAR_NUM_FIELDS + 1) * k]));

  PetscCall(SegGetKSP(seg, &ksp));
  PetscCall(KSPSetOperators(ksp, fsm->M, fsm->P));
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
  Seg_CNLinear      *fsm = (Seg_CNLinear *)seg->data;
  KSP                ksp;
  KSPConvergedReason reason;
  Vec                xs, Xs;
  PetscInt           f;

  PetscFunctionBegin;
  PetscCall(SegGetKSP(seg, &ksp));
  PetscCall(SegCNLinearAssembleSystem_Private(seg, seg->t + seg->dt));
  /* PCABF takes its blocks from M; mark P changed so that the preconditioner is rebuilt */
  PetscCall(PetscObjectStateIncrease((PetscObject)fsm->P));
  PetscCall(VecZeroEntries(fsm->x));
  PetscCall(KSPSolve(ksp, fsm->f, fsm->x));
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
    PetscCall(VecGetSubVector(fsm->x, seg->fields[f].is, &xs));
    PetscCall(VecGetSubVector(seg->sol, seg->fields[f].is, &Xs));
    if (seg->fields[f].update == SEG_FIELD_UPDATE_VALUE) PetscCall(VecCopy(xs, Xs));
    else PetscCall(VecAXPY(Xs, 1., xs));
    PetscCall(VecRestoreSubVector(seg->sol, seg->fields[f].is, &Xs));
    PetscCall(VecRestoreSubVector(fsm->x, seg->fields[f].is, &xs));
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
  Seg_CNLinear *fsm = (Seg_CNLinear *)seg->data;

  PetscFunctionBegin;
  PetscCall(SegOpsDestroy_Internal(seg));
  PetscCall(MatDestroy(&fsm->P));
  PetscCall(MatNullSpaceDestroy(&fsm->nullspace));
  PetscCall(MatDestroy(&fsm->M));
  PetscCall(VecDestroy(&fsm->x));
  PetscCall(VecDestroy(&fsm->f));
  PetscCall(PetscFree(seg->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegCreate_CNLinear(Seg seg)
{
  Seg_CNLinear *fsm;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fsm));
  seg->data = (void *)fsm;

  fsm->M         = NULL;
  fsm->P         = NULL;
  fsm->nullspace = NULL;
  fsm->f         = NULL;
  fsm->x         = NULL;

  seg->ops->setup    = SegSetUp_CNLinear;
  seg->ops->presolve = SegPreSolve_CNLinear;
  seg->ops->step     = SegStep_CNLinear;
  seg->ops->destroy  = SegDestroy_CNLinear;
  seg->ops->view     = SegView_CNLinear;
  PetscFunctionReturn(PETSC_SUCCESS);
}
