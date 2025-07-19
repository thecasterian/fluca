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

static PetscErrorCode CreateDMToDMOperator_Private(DM dmfrom, DM dmto, Mat *A)
{
  PetscInt               entriesfrom, entriesto;
  ISLocalToGlobalMapping ltogfrom, ltogto;
  MatType                mattype;

  PetscFunctionBegin;
  PetscCall(DMStagGetEntries(dmfrom, &entriesfrom));
  PetscCall(DMStagGetEntries(dmto, &entriesto));
  PetscCall(DMGetLocalToGlobalMapping(dmfrom, &ltogfrom));
  PetscCall(DMGetLocalToGlobalMapping(dmto, &ltogto));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmfrom), A));
  PetscCall(MatSetSizes(*A, entriesto, entriesfrom, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(DMGetMatType(dmfrom, &mattype));
  PetscCall(MatSetType(*A, mattype));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltogto, ltogfrom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetup_FSM(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  MPI_Comm comm;
  DM       dm, fdm;
  PC       pc;
  PetscInt dim, d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));

  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));
  PetscCall(MeshGetDimension(ns->mesh, &dim));

  /* Create solution */
  for (d = 0; d < dim; ++d) {
    PetscCall(DMCreateLocalVector(dm, &fsm->v[d]));
    PetscCall(DMCreateLocalVector(dm, &fsm->v_star[d]));
    PetscCall(DMCreateLocalVector(dm, &fsm->N[d]));
    PetscCall(DMCreateLocalVector(dm, &fsm->N_prev[d]));
  }
  PetscCall(DMCreateLocalVector(fdm, &fsm->fv));
  PetscCall(DMCreateLocalVector(fdm, &fsm->fv_star));
  PetscCall(DMCreateLocalVector(dm, &fsm->p));
  PetscCall(DMCreateLocalVector(dm, &fsm->p_half));
  PetscCall(DMCreateLocalVector(dm, &fsm->p_prime));
  PetscCall(DMCreateLocalVector(dm, &fsm->p_half_prev));

  /* Create operators */
  for (d = 0; d < dim; ++d) {
    PetscCall(DMCreateMatrix(dm, &fsm->grad_p[d]));
    PetscCall(DMCreateMatrix(dm, &fsm->grad_p_prime[d]));
    PetscCall(CreateDMToDMOperator_Private(dm, fdm, &fsm->interp_v[d]));
  }
  PetscCall(CreateDMToDMOperator_Private(dm, fdm, &fsm->grad_p_prime_f));
  PetscCall(DMCreateMatrix(dm, &fsm->helm_v));
  PetscCall(DMCreateMatrix(dm, &fsm->lap_p_prime));
  PetscCall(CreateDMToDMOperator_Private(fdm, dm, &fsm->div_fv));

  PetscCall(NSFSMComputePressureGradientOperators2d_Cart_Internal(dm, ns->bcs, fsm->grad_p));
  PetscCall(NSFSMComputePressureCorrectionGradientOperators2d_Cart_Internal(dm, ns->bcs, fsm->grad_p_prime));
  PetscCall(NSFSMComputePressureCorrectionFaceGradientOperator2d_Cart_Internal(dm, fdm, ns->bcs, fsm->grad_p_prime_f));
  PetscCall(NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(dm, ns->bcs, 1., 0.5 * ns->mu * ns->dt / ns->rho, fsm->helm_v));
  PetscCall(NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(dm, ns->bcs, fsm->lap_p_prime));
  PetscCall(NSFSMComputeVelocityInterpolationOperators2d_Cart_Internal(dm, fdm, ns->bcs, fsm->interp_v));

  /* Create KSP */
  for (d = 0; d < dim; ++d) {
    PetscCall(KSPCreate(comm, &fsm->kspv[d]));
    PetscCall(KSPSetDM(fsm->kspv[d], dm));
    PetscCall(KSPGetPC(fsm->kspv[d], &pc));
    PetscCall(PCSetType(pc, PCMG));
    PetscCall(KSPSetFromOptions(fsm->kspv[d]));
  }
  PetscCall(KSPCreate(comm, &fsm->kspp));
  PetscCall(KSPSetDM(fsm->kspp, dm));
  PetscCall(KSPGetPC(fsm->kspp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(KSPSetFromOptions(fsm->kspp));
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
    if (iscart) {
      PetscCall(NSFSMCalculateIntermediateVelocity2d_Cart_Internal(ns));
      PetscCall(NSFSMCalculatePressureCorrection2d_Cart_Internal(ns));
      PetscCall(NSFSMUpdate2d_Cart_Internal(ns));
    }
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
  PetscCall(VecDestroy(&fsm->fv));
  PetscCall(VecDestroy(&fsm->fv_star));
  PetscCall(VecDestroy(&fsm->p));
  PetscCall(VecDestroy(&fsm->p_half));
  PetscCall(VecDestroy(&fsm->p_prime));
  PetscCall(VecDestroy(&fsm->p_half_prev));

  for (d = 0; d < 3; ++d) {
    PetscCall(MatDestroy(&fsm->grad_p[d]));
    PetscCall(MatDestroy(&fsm->grad_p_prime[d]));
    PetscCall(MatDestroy(&fsm->interp_v[d]));
  }
  PetscCall(MatDestroy(&fsm->grad_p_prime_f));
  PetscCall(MatDestroy(&fsm->helm_v));
  PetscCall(MatDestroy(&fsm->lap_p_prime));
  PetscCall(MatDestroy(&fsm->div_fv));

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
  fsm->fv          = NULL;
  fsm->fv_star     = NULL;
  fsm->p           = NULL;
  fsm->p_half      = NULL;
  fsm->p_prime     = NULL;
  fsm->p_half_prev = NULL;

  for (d = 0; d < 3; ++d) fsm->kspv[d] = NULL;
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
