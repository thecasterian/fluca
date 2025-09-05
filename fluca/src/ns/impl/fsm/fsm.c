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

static PetscErrorCode CreateKSPWithDMClone_Private(DM dm, KSP *ksp)
{
  MPI_Comm comm;
  DM       dmclone, cdm;
  PC       pc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(KSPCreate(comm, ksp));

  PetscCall(DMClone(dm, &dmclone));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMSetCoordinateDM(dmclone, cdm));
  PetscCall(KSPSetDM(*ksp, dmclone));
  PetscCall(DMDestroy(&dmclone));

  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));

  PetscCall(KSPSetFromOptions(*ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetup_FSM(NS ns)
{
  NS_FSM   *fsm = (NS_FSM *)ns->data;
  MPI_Comm  comm;
  DM        sdm, Vdm;
  PetscInt  dim, d;
  PetscBool iscart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));

  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));
  PetscCall(MeshGetDimension(ns->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));

  /* Create solution */
  for (d = 0; d < dim; ++d) {
    PetscCall(DMCreateGlobalVector(sdm, &fsm->v[d]));
    PetscCall(DMCreateGlobalVector(sdm, &fsm->v_star[d]));
    PetscCall(DMCreateGlobalVector(sdm, &fsm->N[d]));
    PetscCall(DMCreateGlobalVector(sdm, &fsm->N_prev[d]));
  }
  PetscCall(DMCreateGlobalVector(Vdm, &fsm->V));
  PetscCall(DMCreateGlobalVector(Vdm, &fsm->V_star));
  PetscCall(DMCreateGlobalVector(sdm, &fsm->p));
  PetscCall(DMCreateGlobalVector(sdm, &fsm->p_half));
  PetscCall(DMCreateGlobalVector(sdm, &fsm->p_prime));
  PetscCall(DMCreateGlobalVector(sdm, &fsm->p_half_prev));

  /* Create operators */
  for (d = 0; d < dim; ++d) {
    PetscCall(DMCreateMatrix(sdm, &fsm->Gp[d]));
    PetscCall(CreateOperatorFromDMToDM_Private(sdm, Vdm, &fsm->Tv[d]));
    PetscCall(CreateOperatorFromDMToDM_Private(Vdm, sdm, &fsm->Gstv[d]));
    PetscCall(CreateOperatorFromDMToDM_Private(sdm, Vdm, &fsm->Gstp[d]));
  }
  PetscCall(DMCreateMatrix(sdm, &fsm->Lv));

  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSFSMComputeSpatialOperators2d_Cart_Internal(ns));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }

  /* Create KSP */
  for (d = 0; d < dim; ++d) PetscCall(CreateKSPWithDMClone_Private(sdm, &fsm->kspv[d]));
  PetscCall(CreateKSPWithDMClone_Private(sdm, &fsm->kspp));

  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSFSMSetKSPComputeFunctions2d_Cart_Internal(ns));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }
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
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  PetscInt d;

  PetscFunctionBegin;
  for (d = 0; d < 3; ++d) {
    PetscCall(VecDestroy(&fsm->v[d]));
    PetscCall(VecDestroy(&fsm->v_star[d]));
    PetscCall(VecDestroy(&fsm->N[d]));
    PetscCall(VecDestroy(&fsm->N_prev[d]));
  }
  PetscCall(VecDestroy(&fsm->V));
  PetscCall(VecDestroy(&fsm->V_star));
  PetscCall(VecDestroy(&fsm->p));
  PetscCall(VecDestroy(&fsm->p_half));
  PetscCall(VecDestroy(&fsm->p_prime));
  PetscCall(VecDestroy(&fsm->p_half_prev));

  for (d = 0; d < 3; ++d) {
    PetscCall(MatDestroy(&fsm->Gp[d]));
    PetscCall(MatDestroy(&fsm->Tv[d]));
    PetscCall(MatDestroy(&fsm->Gstv[d]));
    PetscCall(MatDestroy(&fsm->Gstp[d]));
  }
  PetscCall(MatDestroy(&fsm->Lv));
  PetscCall(MatDestroy(&fsm->Dstv));

  for (d = 0; d < 3; ++d) PetscCall(KSPDestroy(&fsm->kspv[d]));
  PetscCall(KSPDestroy(&fsm->kspp));

  PetscCall(PetscFree(ns->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView_FSM(NS ns, PetscViewer v)
{
  PetscFunctionBegin;
  // TODO: add view
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution_FSM(NS ns, PetscViewer v)
{
  PetscBool iscart;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));
  if (iscart) PetscCall(NSViewSolution_FSM_Cart_Internal(ns, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSLoadSolutionCGNS_FSM(NS ns, PetscInt file_num)
{
  PetscBool iscart;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));
  if (iscart) PetscCall(NSLoadSolutionCGNS_FSM_Cart_Internal(ns, file_num));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCreate_FSM(NS ns)
{
  NS_FSM  *fsm;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fsm));
  ns->data = (void *)fsm;

  for (d = 0; d < 3; ++d) {
    fsm->v[d]      = NULL;
    fsm->v_star[d] = NULL;
    fsm->N[d]      = NULL;
    fsm->N_prev[d] = NULL;
  }
  fsm->V           = NULL;
  fsm->V_star      = NULL;
  fsm->p           = NULL;
  fsm->p_half      = NULL;
  fsm->p_prime     = NULL;
  fsm->p_half_prev = NULL;

  for (d = 0; d < 3; ++d) {
    fsm->Gp[d]   = NULL;
    fsm->Tv[d]   = NULL;
    fsm->Gstv[d] = NULL;
    fsm->Gstp[d] = NULL;
  }
  fsm->Lv   = NULL;
  fsm->Dstv = NULL;

  for (d = 0; d < 3; ++d) {
    fsm->kspv[d]         = NULL;
    fsm->kspvctx[d].ns   = ns;
    fsm->kspvctx[d].axis = d;
  }
  fsm->kspp = NULL;

  ns->ops->setfromoptions   = NSSetFromOptions_FSM;
  ns->ops->setup            = NSSetup_FSM;
  ns->ops->iterate          = NSIterate_FSM;
  ns->ops->destroy          = NSDestroy_FSM;
  ns->ops->view             = NSView_FSM;
  ns->ops->viewsolution     = NSViewSolution_FSM;
  ns->ops->loadsolutioncgns = NSLoadSolutionCGNS_FSM;
  PetscFunctionReturn(PETSC_SUCCESS);
}
