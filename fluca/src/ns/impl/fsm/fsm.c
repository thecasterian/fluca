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
  DM        sdm, vdm, Vdm;
  PetscInt  dim, nb, d, i;
  PetscBool neednullspace, iscart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));

  /* Compute initial Jacobian as it is used in computing initial RHS also */
  PetscCall(NSFSMFormJacobian_Cart_Internal(ns->snes, ns->x, ns->J, ns->J, ns));

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
    IS  is;
    Vec vecs[1], subvec;

    PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, &is));
    PetscCall(MatCreateVecs(ns->J, NULL, &vecs[0]));
    PetscCall(VecGetSubVector(vecs[0], is, &subvec));
    PetscCall(VecSet(subvec, 1.));
    PetscCall(VecRestoreSubVector(vecs[0], is, &subvec));
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, 1, vecs, &ns->nullspace));
    PetscCall(VecDestroy(&vecs[0]));
  }

  /* Set solver functions */
  if (iscart) PetscCall(SNESSetPicard(ns->snes, ns->r, NSFSMFormFunction_Cart_Internal, ns->J, ns->J, NSFSMFormJacobian_Cart_Internal, ns));
  if (neednullspace) PetscCall(SNESSetFunction(ns->snes, ns->r, NSFSMPicardComputeFunction_Internal, ns));

  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));
  PetscCall(MeshGetDimension(ns->mesh, &dim));

  /* Create intermediate solution vectors */
  PetscCall(DMCreateGlobalVector(vdm, &fsm->v_star));
  PetscCall(DMCreateGlobalVector(vdm, &fsm->N));
  PetscCall(DMCreateGlobalVector(vdm, &fsm->N_prev));
  PetscCall(DMCreateGlobalVector(Vdm, &fsm->V_star));
  PetscCall(DMCreateGlobalVector(sdm, &fsm->p_half));
  PetscCall(DMCreateGlobalVector(sdm, &fsm->p_prime));
  PetscCall(DMCreateGlobalVector(sdm, &fsm->p_half_prev));

  /* Create operators */
  PetscCall(CreateOperatorFromDMToDM_Private(sdm, vdm, &fsm->Gp));
  PetscCall(CreateOperatorFromDMToDM_Private(vdm, Vdm, &fsm->Tv));
  PetscCall(CreateOperatorFromDMToDM_Private(sdm, Vdm, &fsm->Gstp));
  PetscCall(CreateOperatorFromDMToDM_Private(Vdm, sdm, &fsm->Dstv));
  PetscCall(DMCreateMatrix(vdm, &fsm->Lv));
  for (d = 0; d < 3; ++d) PetscCall(CreateOperatorFromDMToDM_Private(vdm, Vdm, &fsm->TvN[d]));

  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSFSMComputeSpatialOperators2d_Cart_Internal(ns));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }

  /* Create KSP */
  PetscCall(CreateKSPWithDMClone_Private(vdm, &fsm->kspv));
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
  PetscCall(VecDestroy(&fsm->v_star));
  PetscCall(VecDestroy(&fsm->N));
  PetscCall(VecDestroy(&fsm->N_prev));
  PetscCall(VecDestroy(&fsm->V_star));
  PetscCall(VecDestroy(&fsm->p_half));
  PetscCall(VecDestroy(&fsm->p_prime));
  PetscCall(VecDestroy(&fsm->p_half_prev));

  PetscCall(MatDestroy(&fsm->Gp));
  PetscCall(MatDestroy(&fsm->Tv));
  PetscCall(MatDestroy(&fsm->Gstp));
  PetscCall(MatDestroy(&fsm->Lv));
  PetscCall(MatDestroy(&fsm->Dstv));
  for (d = 0; d < 3; ++d) PetscCall(MatDestroy(&fsm->TvN[d]));

  PetscCall(KSPDestroy(&fsm->kspv));
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

  fsm->v_star      = NULL;
  fsm->N           = NULL;
  fsm->N_prev      = NULL;
  fsm->V_star      = NULL;
  fsm->p_half      = NULL;
  fsm->p_prime     = NULL;
  fsm->p_half_prev = NULL;

  fsm->Gp   = NULL;
  fsm->Tv   = NULL;
  fsm->Gstp = NULL;
  fsm->Lv   = NULL;
  fsm->Dstv = NULL;
  for (d = 0; d < 3; ++d) fsm->TvN[d] = NULL;

  fsm->kspv = NULL;
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
