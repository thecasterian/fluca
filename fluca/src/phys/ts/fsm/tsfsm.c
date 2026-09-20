#include <petsc/private/tsimpl.h>
#include <fluca/private/physimpl.h>

typedef struct {
  Phys             phys;
  Mat              M;     /* coupled system (13) on the solution DM */
  Mat              P;     /* MATNEST carrying the field index sets that PCABF reads */
  IS               is[3]; /* velocity, face velocity, pressure */
  MatNullSpace     nullspace;
  KSP              ksp;
  Vec              f, x;
  PetscObjectId    projected_id;    /* id of the vec_sol whose face velocity was last projected */
  PetscObjectState projected_state; /* its state right after that projection */
} TS_FSM;

/* M and f of eq. (13): momentum rows from the state at t^n, coupling rows with boundary data at t_coupling */
static PetscErrorCode TSFSMAssembleSystem_Private(TS ts, PetscReal t_coupling)
{
  TS_FSM *fsm = (TS_FSM *)ts->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(fsm->M));
  PetscCall(VecZeroEntries(fsm->f));
  PetscCall(PhysComputeMomentumSystem(fsm->phys, ts->ptime, ts->time_step, ts->vec_sol, fsm->M, fsm->f));
  PetscCall(PhysComputeCouplingSystem(fsm->phys, t_coupling, ts->time_step, fsm->M, fsm->f));
  PetscCall(MatAssemblyBegin(fsm->M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(fsm->M, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Replace the face velocity of the initial state by the discretely divergence-free projection of
   T u0 + b_interp: U0 = U* - G^st phi with D G^st phi = D U* - b_cont. u0 and p0 are unchanged. */
static PetscErrorCode TSFSMProjectInitialFaceVelocity_Private(TS ts)
{
  TS_FSM            *fsm = (TS_FSM *)ts->data;
  MPI_Comm           comm;
  Mat                negT, G, negR, D, W, S;
  Vec                u, U, fU, fp, Ustar, phi, rhs;
  MatNullSpace       nullspace;
  KSP                ksp;
  PC                 pc;
  KSPConvergedReason reason;
  const char        *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(TSFSMAssembleSystem_Private(ts, ts->ptime));
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
  PetscCall(VecGetSubVector(ts->vec_sol, fsm->is[0], &u));
  PetscCall(VecGetSubVector(fsm->f, fsm->is[1], &fU));
  PetscCall(MatMult(negT, u, Ustar));
  PetscCall(VecAYPX(Ustar, -1., fU));
  PetscCall(VecRestoreSubVector(fsm->f, fsm->is[1], &fU));
  PetscCall(VecRestoreSubVector(ts->vec_sol, fsm->is[0], &u));

  /* S phi = b_cont - D U* */
  PetscCall(MatCreateVecs(S, &phi, &rhs));
  PetscCall(VecGetSubVector(fsm->f, fsm->is[2], &fp));
  PetscCall(MatMult(D, Ustar, rhs));
  PetscCall(VecAYPX(rhs, -1., fp));
  PetscCall(VecRestoreSubVector(fsm->f, fsm->is[2], &fp));
  PetscCall(MatNullSpaceRemove(nullspace, rhs));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)ksp, (PetscObject)ts, 1));
  PetscCall(TSGetOptionsPrefix(ts, &prefix));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(ksp, "ts_fsm_init_"));
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
  PetscCall(VecGetSubVector(ts->vec_sol, fsm->is[1], &U));
  PetscCall(MatMult(W, phi, U));
  PetscCall(VecAXPY(U, 1., Ustar));
  PetscCall(VecRestoreSubVector(ts->vec_sol, fsm->is[1], &U));

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

static PetscErrorCode TSSetUp_FSM(TS ts)
{
  TS_FSM     *fsm = (TS_FSM *)ts->data;
  MPI_Comm    comm;
  DM          dm;
  Vec         nullvec, sub;
  Mat         blocks[9];
  PC          pc, subpc;
  KSP         kspA, kspS;
  PetscBool   isnone;
  PetscInt    k, n, N;
  const char *prefix;
  const char *names[3] = {PHYS_FIELD_VELOCITY, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_PRESSURE};

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCheck(fsm->phys, comm, PETSC_ERR_ARG_WRONGSTATE, "No Phys attached to TSFSM; call PhysSetUpTS() or TSFSMSetPhys()");
  PetscCheck(fsm->phys->setupcalled, comm, PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before TSSetUp() with TSFSM");
  PetscCall(TSGetAdapt(ts, &ts->adapt));
  PetscCall(PetscObjectTypeCompare((PetscObject)ts->adapt, TSADAPTNONE, &isnone));
  PetscCheck(isnone, comm, PETSC_ERR_SUP, "TSFSM supports only TSADAPTNONE");
  PetscCall(TSGetDM(ts, &dm));

  for (k = 0; k < 3; ++k) PetscCall(PhysGetFieldIS(fsm->phys, names[k], &fsm->is[k]));
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

  PetscCall(KSPCreate(comm, &fsm->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)fsm->ksp, (PetscObject)ts, 1));
  PetscCall(TSGetOptionsPrefix(ts, &prefix));
  PetscCall(KSPSetOptionsPrefix(fsm->ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(fsm->ksp, "ts_fsm_"));
  PetscCall(KSPSetOperators(fsm->ksp, fsm->M, fsm->P));
  PetscCall(KSPSetType(fsm->ksp, KSPRICHARDSON));
  PetscCall(KSPSetTolerances(fsm->ksp, 1.e-8, PETSC_CURRENT, PETSC_CURRENT, 1000));
  PetscCall(KSPGetPC(fsm->ksp, &pc));
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
  PetscCall(KSPSetFromOptions(fsm->ksp));
  fsm->projected_id    = 0;
  fsm->projected_state = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_FSM(TS ts)
{
  TS_FSM            *fsm     = (TS_FSM *)ts->data;
  PetscReal          next_dt = ts->time_step;
  PetscBool          accept;
  KSPConvergedReason reason;
  Vec                xs, Xs;
  PetscObjectId      id;
  PetscObjectState   state;
  PetscInt           k;

  PetscFunctionBegin;
  /* fsm->projected_id/projected_state record vec_sol's id and state exactly as TSFSM last left them, either right
     after projecting or at the end of a successful step. A mismatch means the caller swapped in a different Vec or
     modified this one between solves, so re-project; within a solve TSStep is the only thing that touches vec_sol,
     so the record and the live id/state always agree and nothing re-projects. */
  PetscCall(PetscObjectGetId((PetscObject)ts->vec_sol, &id));
  PetscCall(PetscObjectStateGet((PetscObject)ts->vec_sol, &state));
  if (id != fsm->projected_id || state != fsm->projected_state) {
    PetscCall(TSFSMProjectInitialFaceVelocity_Private(ts));
    fsm->projected_id = id;
    PetscCall(PetscObjectStateGet((PetscObject)ts->vec_sol, &fsm->projected_state));
  }

  PetscCall(TSFSMAssembleSystem_Private(ts, ts->ptime + ts->time_step));
  /* PCABF takes its blocks from M; mark P changed so that the preconditioner is rebuilt */
  PetscCall(PetscObjectStateIncrease((PetscObject)fsm->P));
  PetscCall(VecZeroEntries(fsm->x));
  PetscCall(KSPSolve(fsm->ksp, fsm->f, fsm->x));
  PetscCall(KSPGetConvergedReason(fsm->ksp, &reason));
  if (reason == KSP_DIVERGED_ITS) {
    PetscInt max_it;

    /* A fixed number of sweeps (e.g. -ts_fsm_ksp_max_it 1, the classic FSM) ends here by design;
       any larger limit means the requested tolerance was not reached */
    PetscCall(KSPGetTolerances(fsm->ksp, NULL, NULL, NULL, &max_it));
    if (max_it > 1) PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", coupled solve stopped at the iteration limit %" PetscInt_FMT " before reaching the requested tolerance\n", ts->steps, max_it));
  } else if (reason < 0) {
    PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", coupled solve failed: %s\n", ts->steps, KSPConvergedReasons[reason]));
    ts->reason = TS_DIVERGED_STEP_REJECTED;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TSAdaptChoose(ts->adapt, ts, ts->time_step, NULL, &next_dt, &accept));
  if (!accept) {
    ts->reason = TS_DIVERGED_STEP_REJECTED;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* X <- [u^{n+1}, U^{n+1}, q + p'] */
  for (k = 0; k < 2; ++k) {
    PetscCall(VecGetSubVector(fsm->x, fsm->is[k], &xs));
    PetscCall(VecGetSubVector(ts->vec_sol, fsm->is[k], &Xs));
    PetscCall(VecCopy(xs, Xs));
    PetscCall(VecRestoreSubVector(ts->vec_sol, fsm->is[k], &Xs));
    PetscCall(VecRestoreSubVector(fsm->x, fsm->is[k], &xs));
  }
  PetscCall(VecGetSubVector(fsm->x, fsm->is[2], &xs));
  PetscCall(VecGetSubVector(ts->vec_sol, fsm->is[2], &Xs));
  PetscCall(VecAXPY(Xs, 1., xs));
  PetscCall(VecRestoreSubVector(ts->vec_sol, fsm->is[2], &Xs));
  PetscCall(VecRestoreSubVector(fsm->x, fsm->is[2], &xs));

  ts->ptime += ts->time_step;
  ts->time_step     = next_dt;
  fsm->projected_id = id;
  PetscCall(PetscObjectStateGet((PetscObject)ts->vec_sol, &fsm->projected_state));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_FSM(TS ts)
{
  TS_FSM  *fsm = (TS_FSM *)ts->data;
  PetscInt k;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&fsm->ksp));
  PetscCall(MatDestroy(&fsm->P));
  PetscCall(MatNullSpaceDestroy(&fsm->nullspace));
  PetscCall(MatDestroy(&fsm->M));
  PetscCall(VecDestroy(&fsm->x));
  PetscCall(VecDestroy(&fsm->f));
  for (k = 0; k < 3; ++k) PetscCall(ISDestroy(&fsm->is[k]));
  fsm->projected_id    = 0;
  fsm->projected_state = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_FSM(TS ts)
{
  TS_FSM *fsm = (TS_FSM *)ts->data;

  PetscFunctionBegin;
  PetscCall(TSReset_FSM(ts));
  PetscCall(PhysDestroy(&fsm->phys));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSFSMSetPhys_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSFSMGetPhys_C", NULL));
  PetscCall(PetscFree(ts->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_FSM(TS ts, PetscViewer viewer)
{
  TS_FSM *fsm = (TS_FSM *)ts->data;

  PetscFunctionBegin;
  if (fsm->ksp) PetscCall(KSPView(fsm->ksp, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSFSMSetPhys_FSM(TS ts, Phys phys)
{
  TS_FSM *fsm = (TS_FSM *)ts->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)phys));
  PetscCall(PhysDestroy(&fsm->phys));
  fsm->phys = phys;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSFSMGetPhys_FSM(TS ts, Phys *phys)
{
  TS_FSM *fsm = (TS_FSM *)ts->data;

  PetscFunctionBegin;
  *phys = fsm->phys;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSCreate_FSM(TS ts)
{
  TS_FSM *fsm;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fsm));
  ts->data = (void *)fsm;

  ts->ops->setup         = TSSetUp_FSM;
  ts->ops->step          = TSStep_FSM;
  ts->ops->reset         = TSReset_FSM;
  ts->ops->destroy       = TSDestroy_FSM;
  ts->ops->view          = TSView_FSM;
  ts->default_adapt_type = TSADAPTNONE;
  ts->usessnes           = PETSC_FALSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSFSMSetPhys_C", TSFSMSetPhys_FSM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSFSMGetPhys_C", TSFSMGetPhys_FSM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSFSMSetPhys(TS ts, Phys phys)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 2);
  PetscTryMethod(ts, "TSFSMSetPhys_C", (TS, Phys), (ts, phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSFSMGetPhys(TS ts, Phys *phys)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(phys, 2);
  PetscUseMethod(ts, "TSFSMGetPhys_C", (TS, Phys *), (ts, phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}
