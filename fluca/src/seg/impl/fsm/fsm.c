#include <fluca/private/segfsmimpl.h>

/* M and f of eq. (13): momentum rows from the state at t^n, coupling rows with boundary data at t_coupling */
static PetscErrorCode SegFSMAssembleSystem_Private(Seg seg, PetscReal t_coupling)
{
  Seg_FSM *fsm = (Seg_FSM *)seg->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(fsm->M));
  PetscCall(VecZeroEntries(fsm->f));
  PetscCall(PhysComputeMomentumSystem(seg->phys, seg->t, seg->dt, seg->sol, fsm->M, fsm->f));
  PetscCall(PhysComputeCouplingSystem(seg->phys, t_coupling, seg->dt, fsm->M, fsm->f));
  PetscCall(MatAssemblyBegin(fsm->M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(fsm->M, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Replace the face velocity of the initial state by the discretely divergence-free projection of
   T u0 + b_interp: U0 = U* - G^st phi with D G^st phi = D U* - b_cont. u0 and p0 are unchanged. */
static PetscErrorCode SegPreSolve_FSM(Seg seg)
{
  Seg_FSM           *fsm = (Seg_FSM *)seg->data;
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
  PetscCall(SegFSMAssembleSystem_Private(seg, seg->t));
  PetscCall(MatCreateSubMatrix(fsm->M, fsm->is[1], fsm->is[0], MAT_INITIAL_MATRIX, &negT));
  PetscCall(MatCreateSubMatrix(fsm->M, fsm->is[0], fsm->is[2], MAT_INITIAL_MATRIX, &G));
  PetscCall(MatCreateSubMatrix(fsm->M, fsm->is[1], fsm->is[2], MAT_INITIAL_MATRIX, &negR));
  PetscCall(MatCreateSubMatrix(fsm->M, fsm->is[2], fsm->is[1], MAT_INITIAL_MATRIX, &D));

  /* W = (-T) G - (-R) = -G^st, S = D W: the Schur complement of eq. (18) */
  PetscCall(MatMatMult(negT, G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &W));
  PetscCall(MatAXPY(W, -1., negR, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatMatMult(D, W, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &S));
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(S, nullspace));

  /* U* = T u0 + b_interp */
  PetscCall(MatCreateVecs(negT, NULL, &Ustar));
  PetscCall(VecGetSubVector(seg->sol, fsm->is[0], &u));
  PetscCall(VecGetSubVector(fsm->f, fsm->is[1], &fU));
  PetscCall(MatMult(negT, u, Ustar));
  PetscCall(VecAYPX(Ustar, -1., fU));
  PetscCall(VecRestoreSubVector(fsm->f, fsm->is[1], &fU));
  PetscCall(VecRestoreSubVector(seg->sol, fsm->is[0], &u));

  /* S phi = b_cont - D U* */
  PetscCall(MatCreateVecs(S, &phi, &rhs));
  PetscCall(VecGetSubVector(fsm->f, fsm->is[2], &fp));
  PetscCall(MatMult(D, Ustar, rhs));
  PetscCall(VecAYPX(rhs, -1., fp));
  PetscCall(VecRestoreSubVector(fsm->f, fsm->is[2], &fp));
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
  PetscCall(VecGetSubVector(seg->sol, fsm->is[1], &U));
  PetscCall(MatMult(W, phi, U));
  PetscCall(VecAXPY(U, 1., Ustar));
  PetscCall(VecRestoreSubVector(seg->sol, fsm->is[1], &U));

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

static PetscErrorCode SegSetUp_FSM(Seg seg)
{
  Seg_FSM    *fsm = (Seg_FSM *)seg->data;
  MPI_Comm    comm;
  DM          dm;
  Vec         nullvec, sub;
  Mat         blocks[9];
  KSP         ksp, kspA, kspS;
  PC          pc, subpc;
  PetscInt    k, n, N;
  const char *names[3] = {PHYS_FIELD_VELOCITY, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_PRESSURE};

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)seg, &comm));
  PetscCheck(seg->phys->setupcalled, comm, PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before SegSetUp() with SEGFSM");
  PetscCall(PhysGetSolutionDM(seg->phys, &dm));

  for (k = 0; k < 3; ++k) PetscCall(PhysGetFieldIS(seg->phys, names[k], &fsm->is[k]));
  PetscCall(DMCreateMatrix(dm, &fsm->M));
  PetscCall(MatSetOption(fsm->M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(fsm->M, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &fsm->f));
  PetscCall(DMCreateGlobalVector(dm, &fsm->x));

  /* Pressure is determined up to a constant (velocity or periodic boundaries only) */
  PetscCall(DMCreateGlobalVector(dm, &nullvec));
  PetscCall(VecZeroEntries(nullvec));
  PetscCall(VecGetSubVector(nullvec, fsm->is[2], &sub));
  PetscCall(VecGetSize(sub, &N));
  PetscCall(VecSet(sub, 1. / PetscSqrtReal((PetscReal)N)));
  PetscCall(VecRestoreSubVector(nullvec, fsm->is[2], &sub));
  PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, 1, &nullvec, &fsm->nullspace));
  PetscCall(VecDestroy(&nullvec));
  PetscCall(MatSetNullSpace(fsm->M, fsm->nullspace));

  /* PCABF reads the field index sets from a MATNEST preconditioning matrix and the blocks from M */
  for (k = 0; k < 9; ++k) blocks[k] = NULL;
  for (k = 0; k < 3; ++k) {
    PetscCall(ISGetLocalSize(fsm->is[k], &n));
    PetscCall(ISGetSize(fsm->is[k], &N));
    PetscCall(MatCreateConstantDiagonal(comm, n, n, N, N, 1., &blocks[4 * k]));
  }
  PetscCall(MatCreateNest(comm, 3, fsm->is, 3, fsm->is, blocks, &fsm->P));
  for (k = 0; k < 3; ++k) PetscCall(MatDestroy(&blocks[4 * k]));

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

static PetscErrorCode SegStep_FSM(Seg seg)
{
  Seg_FSM           *fsm = (Seg_FSM *)seg->data;
  KSP                ksp;
  KSPConvergedReason reason;
  Vec                xs, Xs;
  PetscInt           k;

  PetscFunctionBegin;
  PetscCall(SegGetKSP(seg, &ksp));
  PetscCall(SegFSMAssembleSystem_Private(seg, seg->t + seg->dt));
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

  /* X <- [u^{n+1}, U^{n+1}, q + p'] */
  for (k = 0; k < 2; ++k) {
    PetscCall(VecGetSubVector(fsm->x, fsm->is[k], &xs));
    PetscCall(VecGetSubVector(seg->sol, fsm->is[k], &Xs));
    PetscCall(VecCopy(xs, Xs));
    PetscCall(VecRestoreSubVector(seg->sol, fsm->is[k], &Xs));
    PetscCall(VecRestoreSubVector(fsm->x, fsm->is[k], &xs));
  }
  PetscCall(VecGetSubVector(fsm->x, fsm->is[2], &xs));
  PetscCall(VecGetSubVector(seg->sol, fsm->is[2], &Xs));
  PetscCall(VecAXPY(Xs, 1., xs));
  PetscCall(VecRestoreSubVector(seg->sol, fsm->is[2], &Xs));
  PetscCall(VecRestoreSubVector(fsm->x, fsm->is[2], &xs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegView_FSM(Seg seg, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (seg->ksp) PetscCall(KSPView(seg->ksp, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegDestroy_FSM(Seg seg)
{
  Seg_FSM *fsm = (Seg_FSM *)seg->data;
  PetscInt k;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&fsm->P));
  PetscCall(MatNullSpaceDestroy(&fsm->nullspace));
  PetscCall(MatDestroy(&fsm->M));
  PetscCall(VecDestroy(&fsm->x));
  PetscCall(VecDestroy(&fsm->f));
  for (k = 0; k < 3; ++k) PetscCall(ISDestroy(&fsm->is[k]));
  PetscCall(PetscFree(seg->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegCreate_FSM(Seg seg)
{
  Seg_FSM *fsm;
  PetscInt k;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fsm));
  seg->data = (void *)fsm;

  fsm->M         = NULL;
  fsm->P         = NULL;
  fsm->nullspace = NULL;
  fsm->f         = NULL;
  fsm->x         = NULL;
  for (k = 0; k < 3; ++k) fsm->is[k] = NULL;

  seg->ops->setup    = SegSetUp_FSM;
  seg->ops->presolve = SegPreSolve_FSM;
  seg->ops->step     = SegStep_FSM;
  seg->ops->destroy  = SegDestroy_FSM;
  seg->ops->view     = SegView_FSM;
  PetscFunctionReturn(PETSC_SUCCESS);
}
