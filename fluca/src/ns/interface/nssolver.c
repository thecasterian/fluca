#include <fluca/private/nslinearcnimpl.h>

static PetscErrorCode NSFSMPCCtxCreate_Private(NS ns, PC pc, NSFSMPCCtx **ctx)
{
  NSFSMPCCtx *c;
  MPI_Comm    comm;
  Vec         v, V, p;

  PetscFunctionBegin;
  PetscCall(PetscNew(&c));

  PetscCall(NSGetField(ns, NS_FIELD_VELOCITY, NULL, NULL, &c->vis));
  PetscCall(NSGetField(ns, NS_FIELD_FACE_NORMAL_VELOCITY, NULL, NULL, &c->Vis));
  PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, NULL, &c->pis));

  PetscCall(MatCreateSubMatrix(ns->J, c->vis, c->vis, MAT_INITIAL_MATRIX, &c->A));
  PetscCall(MatCreateSubMatrix(ns->J, c->Vis, c->vis, MAT_INITIAL_MATRIX, &c->T));
  PetscCall(MatCreateSubMatrix(ns->J, c->vis, c->pis, MAT_INITIAL_MATRIX, &c->G));
  PetscCall(PetscObjectQuery((PetscObject)ns->J, "StaggeredGradient", (PetscObject *)&c->Gst));
  PetscCall(PetscObjectReference((PetscObject)c->Gst));
  PetscCall(MatCreateSubMatrix(ns->J, c->pis, c->Vis, MAT_INITIAL_MATRIX, &c->D));
  PetscCall(MatMatMult(c->D, c->Gst, MAT_INITIAL_MATRIX, 1., &c->Lst));

  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_VELOCITY, &v));
  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_FACE_NORMAL_VELOCITY, &V));
  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_PRESSURE, &p));
  PetscCall(VecDuplicate(p, &c->divvstar));
  PetscCall(VecDuplicate(v, &c->gradpcorr));
  PetscCall(VecDuplicate(V, &c->gradstpcorr));
  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_VELOCITY, &v));
  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_FACE_NORMAL_VELOCITY, &V));
  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_PRESSURE, &p));

  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));
  PetscCall(KSPCreate(comm, &c->kspv));
  PetscCall(KSPSetOperators(c->kspv, c->A, c->A));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)c->kspv, (PetscObject)pc, 1));
  PetscCall(KSPSetOptionsPrefix(c->kspv, "ns_fsm_v_"));
  PetscCall(KSPSetFromOptions(c->kspv));
  PetscCall(KSPCreate(comm, &c->kspp));
  PetscCall(KSPSetOperators(c->kspp, c->Lst, c->Lst));
  PetscCall(KSPSetOptionsPrefix(c->kspp, "ns_fsm_p_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)c->kspp, (PetscObject)pc, 1));
  PetscCall(KSPSetFromOptions(c->kspp));
  if (ns->nullspace) {
    PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &c->nullspace));
    PetscCall(MatSetNullSpace(c->Lst, c->nullspace));
  } else {
    c->nullspace = NULL;
  }

  *ctx = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMPCApply_Private(PC pc, Vec x, Vec y)
{
  NSFSMPCCtx *ctx;
  Vec         xv, xV, xp, yv, yV, yp;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(VecGetSubVector(x, ctx->vis, &xv));
  PetscCall(VecGetSubVector(x, ctx->Vis, &xV));
  PetscCall(VecGetSubVector(x, ctx->pis, &xp));
  PetscCall(VecGetSubVector(y, ctx->vis, &yv));
  PetscCall(VecGetSubVector(y, ctx->Vis, &yV));
  PetscCall(VecGetSubVector(y, ctx->pis, &yp));

  /* Forward step */
  PetscCall(KSPSolve(ctx->kspv, xv, yv));
  PetscCall(MatMult(ctx->T, yv, yV));
  PetscCall(VecAXPY(yV, -1., xV));
  PetscCall(MatMult(ctx->D, yV, ctx->divvstar));
  PetscCall(VecAXPY(ctx->divvstar, -1., xp));
  if (ctx->nullspace) PetscCall(MatNullSpaceRemove(ctx->nullspace, ctx->divvstar));
  PetscCall(KSPSolve(ctx->kspp, ctx->divvstar, yp));

  /* Backward step */
  PetscCall(MatMult(ctx->G, yp, ctx->gradpcorr));
  PetscCall(VecAXPY(yv, -1., ctx->gradpcorr));
  PetscCall(MatMult(ctx->Gst, yp, ctx->gradstpcorr));
  PetscCall(VecAXPY(yV, -1., ctx->gradstpcorr));

  PetscCall(VecRestoreSubVector(x, ctx->vis, &xv));
  PetscCall(VecRestoreSubVector(x, ctx->Vis, &xV));
  PetscCall(VecRestoreSubVector(x, ctx->pis, &xp));
  PetscCall(VecRestoreSubVector(y, ctx->vis, &yv));
  PetscCall(VecRestoreSubVector(y, ctx->Vis, &yV));
  PetscCall(VecRestoreSubVector(y, ctx->pis, &yp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMPCDestroy_Private(PC pc)
{
  NSFSMPCCtx *ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(MatDestroy(&ctx->T));
  PetscCall(MatDestroy(&ctx->G));
  PetscCall(MatDestroy(&ctx->Gst));
  PetscCall(MatDestroy(&ctx->D));
  PetscCall(MatDestroy(&ctx->Lst));
  PetscCall(VecDestroy(&ctx->divvstar));
  PetscCall(VecDestroy(&ctx->gradpcorr));
  PetscCall(VecDestroy(&ctx->gradstpcorr));
  PetscCall(KSPDestroy(&ctx->kspv));
  PetscCall(KSPDestroy(&ctx->kspp));
  PetscCall(MatNullSpaceDestroy(&ctx->nullspace));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetPreconditioner_FSM(NS ns)
{
  KSP         ksp;
  PC          pc;
  NSFSMPCCtx *pcctx;

  PetscFunctionBegin;
  /* Set preconditioner */
  PetscCall(SNESGetKSP(ns->snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetName(pc, "FractionalStepMethod"));
  PetscCall(PCShellSetApply(pc, NSFSMPCApply_Private));
  PetscCall(NSFSMPCCtxCreate_Private(ns, pc, &pcctx));
  PetscCall(PCShellSetContext(pc, pcctx));
  PetscCall(PCShellSetDestroy(pc, NSFSMPCDestroy_Private));
  PetscFunctionReturn(PETSC_SUCCESS);
}
