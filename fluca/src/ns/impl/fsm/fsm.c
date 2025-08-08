#include <fluca/private/nsfsmimpl.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

PetscErrorCode NSSetFromOptions_FSM(NS ns, PetscOptionItems PetscOptionsObject)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "NSFSM Options");
  // TODO: Add options
  (void)fsm;
  PetscOptionsHeadEnd();
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
  DM        dm;
  PetscInt  dim, d;
  PetscBool iscart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));

  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetDimension(ns->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));

  /* Create solution */
  for (d = 0; d < dim; ++d) {
    PetscCall(DMCreateGlobalVector(dm, &fsm->v[d]));
    PetscCall(DMCreateGlobalVector(dm, &fsm->v_star[d]));
    PetscCall(DMCreateGlobalVector(dm, &fsm->N[d]));
    PetscCall(DMCreateGlobalVector(dm, &fsm->N_prev[d]));
  }
  PetscCall(DMCreateGlobalVector(dm, &fsm->p));
  PetscCall(DMCreateGlobalVector(dm, &fsm->p_half));
  PetscCall(DMCreateGlobalVector(dm, &fsm->p_prime));
  PetscCall(DMCreateGlobalVector(dm, &fsm->p_half_prev));

  /* Create operators */
  for (d = 0; d < dim; ++d) {
    PetscCall(DMCreateMatrix(dm, &fsm->Gp[d]));
    PetscCall(DMCreateMatrix(dm, &fsm->Gv[d]));
  }
  PetscCall(DMCreateMatrix(dm, &fsm->Lv));

  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSFSMComputeSpatialOperators2d_Cart_Internal(ns));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }

  /* Create KSP */
  for (d = 0; d < dim; ++d) PetscCall(CreateKSPWithDMClone_Private(dm, &fsm->kspv[d]));
  PetscCall(CreateKSPWithDMClone_Private(dm, &fsm->kspp));

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
  PetscCall(VecDestroy(&fsm->p));
  PetscCall(VecDestroy(&fsm->p_half));
  PetscCall(VecDestroy(&fsm->p_prime));
  PetscCall(VecDestroy(&fsm->p_half_prev));

  for (d = 0; d < 3; ++d) {
    PetscCall(MatDestroy(&fsm->Gp[d]));
    PetscCall(MatDestroy(&fsm->Gv[d]));
  }
  PetscCall(MatDestroy(&fsm->Lv));

  for (d = 0; d < 3; ++d) PetscCall(KSPDestroy(&fsm->kspv[d]));
  PetscCall(KSPDestroy(&fsm->kspp));

  PetscCall(PetscFree(ns->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView_FSM(NS ns, PetscViewer v)
{
  PetscFunctionBegin;
  // TODO: add view
  (void)ns;
  (void)v;
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
  fsm->p           = NULL;
  fsm->p_half      = NULL;
  fsm->p_prime     = NULL;
  fsm->p_half_prev = NULL;

  for (d = 0; d < 3; ++d) {
    fsm->Gp[d] = NULL;
    fsm->Gv[d] = NULL;
  }
  fsm->Lv = NULL;

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
