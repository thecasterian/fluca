#include <fluca/private/nsfsmimpl.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

PetscErrorCode NSSetFromOptions_FSM(NS ns, PetscOptionItems PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "NSFSM Options");
  // TODO: Add options
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMFormInitialGuess_Private(SNES snes, Vec x, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSFSMPicardComputeFunction_Private(SNES snes, Vec x, Vec f, void *ctx)
{
  NS ns = (NS)ctx;

  PetscFunctionBegin;
  PetscCall(SNESPicardComputeFunction(snes, x, f, ctx));

  /* Remove null space */
  PetscAssert(ns->nullspace, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Null space must be set");
  PetscCall(MatNullSpaceRemove(ns->nullspace, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

static PetscErrorCode CreateOperatorFromDMToDM_Private(DM dmfrom, DM dmto, Mat *A)
{
  PetscInt               entriesfrom, entriesto;
  ISLocalToGlobalMapping ltogfrom, ltogto;
  MatType                mattype;

  PetscFunctionBegin;
  PetscCall(DMStagGetEntries(dmfrom, &entriesfrom));
  PetscCall(DMStagGetEntries(dmto, &entriesto));
  PetscCall(DMGetLocalToGlobalMapping(dmfrom, &ltogfrom));
  PetscCall(DMGetLocalToGlobalMapping(dmto, &ltogto));
  PetscCall(DMGetMatType(dmfrom, &mattype));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmfrom), A));
  PetscCall(MatSetSizes(*A, entriesto, entriesfrom, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(*A, mattype));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltogto, ltogfrom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetup_FSM(NS ns)
{
  NS_FSM     *fsm = (NS_FSM *)ns->data;
  MPI_Comm    comm;
  KSP         ksp;
  PC          pc;
  NSFSMPCCtx *pcctx;
  DM          vdm, Vdm;
  PetscInt    dim, nb, i;
  PetscBool   neednullspace, iscart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));

  /* Create intermediate solution vectors and spatial operators */
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_STAG_VECTOR, &Vdm));
  PetscCall(MeshGetDimension(ns->mesh, &dim));

  PetscCall(MeshCreateGlobalVector(ns->mesh, MESH_DM_STAG_VECTOR, &fsm->v0interp));
  PetscCall(MeshCreateGlobalVector(ns->mesh, MESH_DM_SCALAR, &fsm->phalf));
  PetscCall(CreateOperatorFromDMToDM_Private(vdm, Vdm, &fsm->B));

  PetscCall(PetscObjectSetName((PetscObject)fsm->phalf, "PressureHalfStep"));

  /* Preallocate Jacobian */
  if (iscart) PetscCall(NSFSMFormJacobian_Cart_Internal(ns->snes, ns->x, ns->J, ns->J, ns));
  else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");

  /* Create null space */
  neednullspace = PETSC_TRUE;
  PetscCall(MeshGetNumberBoundaries(ns->mesh, &nb));
  for (i = 0; i < nb; ++i) switch (ns->bcs[i].type) {
    case NS_BC_VELOCITY:
    case NS_BC_PERIODIC:
      /* Need null space */
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type");
    }
  if (neednullspace) {
    IS       is;
    Vec      vecs[1], subvec;
    PetscInt subvecsize;

    PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, NULL, &is));
    PetscCall(MatCreateVecs(ns->J, NULL, &vecs[0]));
    PetscCall(VecGetSubVector(vecs[0], is, &subvec));
    PetscCall(VecGetSize(subvec, &subvecsize));
    PetscCall(VecSet(subvec, 1. / PetscSqrtReal((PetscReal)subvecsize)));
    PetscCall(VecRestoreSubVector(vecs[0], is, &subvec));
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, 1, vecs, &ns->nullspace));
    PetscCall(VecDestroy(&vecs[0]));
  }

  /* Set solver functions */
  if (iscart) PetscCall(SNESSetPicard(ns->snes, ns->r, NSFSMFormFunction_Cart_Internal, ns->J, ns->J, NSFSMFormJacobian_Cart_Internal, ns));
  else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
  if (neednullspace) PetscCall(SNESSetFunction(ns->snes, ns->r, NSFSMPicardComputeFunction_Private, ns));
  /* Need zero initial guess to ensure least-square solution of pressure poisson equation */
  PetscCall(SNESSetComputeInitialGuess(ns->snes, NSFSMFormInitialGuess_Private, NULL));

  /* Set KSP options */
  PetscCall(SNESGetKSP(ns->snes, &ksp));
  PetscCall(KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, 100));
  PetscCall(KSPSetFromOptions(ksp));

  /* Set preconditioner */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetName(pc, "FractionalStepMethod"));
  PetscCall(PCShellSetApply(pc, NSFSMPCApply_Private));
  PetscCall(NSFSMPCCtxCreate_Private(ns, pc, &pcctx));
  PetscCall(PCShellSetContext(pc, pcctx));
  PetscCall(PCShellSetDestroy(pc, NSFSMPCDestroy_Private));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSIterate_FSM(NS ns)
{
  PetscInt  dim;
  PetscBool iscart;

  PetscFunctionBegin;
  PetscCall(MeshGetDimension(ns->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));
  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSFSMIterate2d_Cart_Internal(ns));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSDestroy_FSM(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&fsm->v0interp));
  PetscCall(VecDestroy(&fsm->phalf));
  PetscCall(MatDestroy(&fsm->B));
  PetscCall(PetscFree(ns->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView_FSM(NS ns, PetscViewer viewer)
{
  PetscFunctionBegin;
  // TODO: add view
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution_FSM(NS ns, PetscViewer viewer)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscCall(VecView(fsm->phalf, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSLoadSolution_FSM(NS ns, PetscViewer viewer)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscCall(FlucaVecLoad(fsm->phalf, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCreate_FSM(NS ns)
{
  NS_FSM *fsm;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fsm));
  ns->data = (void *)fsm;

  fsm->v0interp  = NULL;
  fsm->phalf     = NULL;
  fsm->B         = NULL;
  fsm->Bcomputed = PETSC_FALSE;

  ns->ops->setfromoptions = NSSetFromOptions_FSM;
  ns->ops->setup          = NSSetup_FSM;
  ns->ops->iterate        = NSIterate_FSM;
  ns->ops->destroy        = NSDestroy_FSM;
  ns->ops->view           = NSView_FSM;
  ns->ops->viewsolution   = NSViewSolution_FSM;
  ns->ops->loadsolution   = NSLoadSolution_FSM;
  PetscFunctionReturn(PETSC_SUCCESS);
}
