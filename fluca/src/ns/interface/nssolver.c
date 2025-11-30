#include <fluca/private/nslinearcnimpl.h>

typedef struct {
  NS ns;

  KSP kspA;
  KSP kspS;

  Mat A;    /* momentum equation operator */
  Mat negT; /* negative face-normal velocity interpolation operator */
  Mat G;    /* pressure gradient operator */
  Mat D;    /* face-normal velocity divergence operator */
  Mat negR; /* Rhie-Chow interpolation correction operator */

  Mat invA2; /* inverse of approximation of A in the upper triangular matrix */
  Mat S;     /* approximate Schur complement */

  MatNullSpace nullspace;
} ABFCtx;

static PetscErrorCode ABFCtxCreateKSP_Private(NS ns, PC pc, const char prefix[], KSP *ksp)
{
  KSP         k;
  const char *nsprefix;

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &k));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)k, (PetscObject)pc, 1));
  PetscCall(PetscObjectSetOptions((PetscObject)k, ((PetscObject)pc)->options));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ns, &nsprefix));
  PetscCall(KSPSetOptionsPrefix(k, nsprefix));
  PetscCall(KSPAppendOptionsPrefix(k, prefix));
  *ksp = k;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ABFCtxCreate_Private(NS ns, PC pc, ABFCtx **ctx)
{
  ABFCtx *c;

  PetscFunctionBegin;
  PetscCall(PetscNew(&c));
  c->ns = ns;
  PetscCall(ABFCtxCreateKSP_Private(ns, pc, "ns_abf_A_", &c->kspA));
  PetscCall(ABFCtxCreateKSP_Private(ns, pc, "ns_abf_S_", &c->kspS));
  c->A         = NULL;
  c->negT      = NULL;
  c->G         = NULL;
  c->D         = NULL;
  c->negR      = NULL;
  c->invA2     = NULL;
  c->S         = NULL;
  c->nullspace = NULL;
  *ctx         = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ABFSetUp_Private(PC pc)
{
  ABFCtx      *ctx;
  IS           vis, Vis, pis;
  Mat          Jpre, tmp;
  PetscInt     m, n;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(NSGetField(ctx->ns, NS_FIELD_VELOCITY, NULL, NULL, &vis));
  PetscCall(NSGetField(ctx->ns, NS_FIELD_FACE_NORMAL_VELOCITY, NULL, NULL, &Vis));
  PetscCall(NSGetField(ctx->ns, NS_FIELD_PRESSURE, NULL, NULL, &pis));
  PetscCall(PCGetOperators(pc, NULL, &Jpre));

  PetscCall(MatDestroy(&ctx->A));
  PetscCall(MatDestroy(&ctx->negT));
  PetscCall(MatDestroy(&ctx->G));
  PetscCall(MatDestroy(&ctx->D));
  PetscCall(MatDestroy(&ctx->negR));
  PetscCall(MatDestroy(&ctx->invA2));
  PetscCall(MatDestroy(&ctx->S));
  PetscCall(MatNullSpaceDestroy(&ctx->nullspace));

  PetscCall(MatCreateSubMatrix(Jpre, vis, vis, MAT_INITIAL_MATRIX, &ctx->A));
  PetscCall(MatCreateSubMatrix(Jpre, Vis, vis, MAT_INITIAL_MATRIX, &ctx->negT));
  PetscCall(MatCreateSubMatrix(Jpre, vis, pis, MAT_INITIAL_MATRIX, &ctx->G));
  PetscCall(MatCreateSubMatrix(Jpre, pis, Vis, MAT_INITIAL_MATRIX, &ctx->D));
  PetscCall(MatCreateSubMatrix(Jpre, Vis, pis, MAT_INITIAL_MATRIX, &ctx->negR));

  // TODO: create invA2 and S based on options
  PetscCall(MatGetLocalSize(ctx->A, &m, &n));
  PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)Jpre), m, n, PETSC_DETERMINE, PETSC_DETERMINE, 1., &ctx->invA2));

  PetscCall(MatMatMult(ctx->negT, ctx->G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp));
  PetscCall(MatAXPY(tmp, -1., ctx->negR, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatMatMult(ctx->D, tmp, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &ctx->S));
  PetscCall(MatDestroy(&tmp));

  PetscCall(MatGetNullSpace(Jpre, &nullspace));
  if (nullspace) {
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)Jpre), PETSC_TRUE, 0, NULL, &ctx->nullspace));
    PetscCall(MatSetNullSpace(ctx->S, ctx->nullspace));
  }

  PetscCall(KSPSetOperators(ctx->kspA, ctx->A, ctx->A));
  PetscCall(KSPSetOperators(ctx->kspS, ctx->S, ctx->S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ABFApply_Private(PC pc, Vec b, Vec x)
{
  ABFCtx *ctx;
  DM      sdm, vdm, Sdm;
  IS      vis, Vis, pis;
  Vec     momrhs, interprhs, contrhs, v, V, pcorr, vstar, Vstar, Srhs, invA2Gpcorr, tmp;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(MeshGetDM(ctx->ns->mesh, MESH_DM_SCALAR, &sdm));
  PetscCall(MeshGetDM(ctx->ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDM(ctx->ns->mesh, MESH_DM_STAG_SCALAR, &Sdm));
  PetscCall(NSGetField(ctx->ns, NS_FIELD_VELOCITY, NULL, NULL, &vis));
  PetscCall(NSGetField(ctx->ns, NS_FIELD_FACE_NORMAL_VELOCITY, NULL, NULL, &Vis));
  PetscCall(NSGetField(ctx->ns, NS_FIELD_PRESSURE, NULL, NULL, &pis));

  PetscCall(VecGetSubVector(b, vis, &momrhs));
  PetscCall(VecGetSubVector(b, Vis, &interprhs));
  PetscCall(VecGetSubVector(b, pis, &contrhs));
  PetscCall(VecGetSubVector(x, vis, &v));
  PetscCall(VecGetSubVector(x, Vis, &V));
  PetscCall(VecGetSubVector(x, pis, &pcorr));
  PetscCall(DMGetGlobalVector(vdm, &vstar));
  PetscCall(DMGetGlobalVector(Sdm, &Vstar));
  PetscCall(DMGetGlobalVector(sdm, &Srhs));
  PetscCall(DMGetGlobalVector(vdm, &invA2Gpcorr));

  /* Stage 1: solve the lower triangular matrix */
  PetscCall(KSPSolve(ctx->kspA, momrhs, vstar));
  PetscCall(MatMult(ctx->negT, vstar, Vstar));
  PetscCall(VecAYPX(Vstar, -1, interprhs));
  PetscCall(MatMult(ctx->D, Vstar, Srhs));
  PetscCall(VecAYPX(Srhs, -1., contrhs));
  PetscCall(KSPSolve(ctx->kspS, Srhs, pcorr));

  /* Stage 2: solve the upper triangular matrix */
  PetscCall(DMGetGlobalVector(vdm, &tmp));
  PetscCall(MatMult(ctx->G, pcorr, tmp));
  PetscCall(MatMult(ctx->invA2, tmp, invA2Gpcorr));
  PetscCall(DMRestoreGlobalVector(vdm, &tmp));
  PetscCall(VecWAXPY(v, -1., invA2Gpcorr, vstar));
  PetscCall(MatMultAdd(ctx->negT, invA2Gpcorr, Vstar, V));
  if (ctx->negR) {
    PetscCall(DMGetGlobalVector(Sdm, &tmp));
    PetscCall(MatMult(ctx->negR, pcorr, tmp));
    PetscCall(VecAXPY(V, -1., tmp));
    PetscCall(DMRestoreGlobalVector(Sdm, &tmp));
  }

  PetscCall(VecRestoreSubVector(b, vis, &momrhs));
  PetscCall(VecRestoreSubVector(b, Vis, &interprhs));
  PetscCall(VecRestoreSubVector(b, pis, &contrhs));
  PetscCall(VecRestoreSubVector(x, vis, &v));
  PetscCall(VecRestoreSubVector(x, Vis, &V));
  PetscCall(VecRestoreSubVector(x, pis, &pcorr));
  PetscCall(DMRestoreGlobalVector(vdm, &vstar));
  PetscCall(DMRestoreGlobalVector(Sdm, &Vstar));
  PetscCall(DMRestoreGlobalVector(sdm, &Srhs));
  PetscCall(DMRestoreGlobalVector(vdm, &invA2Gpcorr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ABFDestroy_Private(PC pc)
{
  ABFCtx *ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(KSPDestroy(&ctx->kspS));
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(MatDestroy(&ctx->negT));
  PetscCall(MatDestroy(&ctx->G));
  PetscCall(MatDestroy(&ctx->D));
  PetscCall(MatDestroy(&ctx->invA2));
  PetscCall(MatDestroy(&ctx->S));
  PetscCall(MatNullSpaceDestroy(&ctx->nullspace));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ABFView_Private(PC pc, PetscViewer viewer)
{
  ABFCtx   *ctx;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "KSP solver for momentum equation operator\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(KSPView(ctx->kspA, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "KSP solver for Schur complement\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(KSPView(ctx->kspS, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetSNES(NS ns, SNES *snes)
{
  const char *prefix;
  KSP         ksp;
  PC          pc;
  ABFCtx     *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(snes, 2);
  if (!ns->snes) {
    PetscCall(SNESCreate(PetscObjectComm((PetscObject)ns), &ns->snes));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)ns->snes, (PetscObject)ns, 1));
    PetscCall(PetscObjectSetOptions((PetscObject)ns->snes, ((PetscObject)ns)->options));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ns, &prefix));
    PetscCall(SNESSetOptionsPrefix(ns->snes, prefix));
    PetscCall(SNESAppendOptionsPrefix(ns->snes, "ns_"));

    /* Default SNES and KSP options */
    PetscCall(SNESSetTolerances(ns->snes, PETSC_DECIDE, 1.e-5, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(SNESGetKSP(ns->snes, &ksp));
    PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));

    /* Construct approximate block factorization preconditioner (ABF) */
    PetscCall(PCCreate(PetscObjectComm((PetscObject)ns), &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetName(pc, "Approxmiate Block Factorization Preconditioner (ABF)"));
    PetscCall(ABFCtxCreate_Private(ns, pc, &ctx));
    PetscCall(PCShellSetContext(pc, ctx));
    PetscCall(PCShellSetSetUp(pc, ABFSetUp_Private));
    PetscCall(PCShellSetApply(pc, ABFApply_Private));
    PetscCall(PCShellSetDestroy(pc, ABFDestroy_Private));
    PetscCall(PCShellSetView(pc, ABFView_Private));
    PetscCall(KSPSetPC(ksp, pc));
  }
  *snes = ns->snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}
