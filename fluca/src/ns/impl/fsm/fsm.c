#include <fluca/private/ns_fsm.h>

extern PetscErrorCode NSFSMInterpolateVelocity2d_MeshCart(NS ns);
extern PetscErrorCode NSFSMCalculateConvection2d_MeshCart(NS ns);
extern PetscErrorCode NSFSMCalculateIntermediateVelocity2d_MeshCart(NS ns);
extern PetscErrorCode NSFSMCalculatePressureCorrection2d_MeshCart(NS ns);
extern PetscErrorCode NSFSMUpdate2d_MeshCart(NS ns);

PetscErrorCode NSSetFromOptions_FSM(NS ns, PetscOptionItems *PetscOptionsObject)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;

  PetscOptionsHeadBegin(PetscOptionsObject, "NSFSM Options");

  // TODO: Add options
  (void)fsm;

  PetscOptionsHeadEnd();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetup_FSM(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  MPI_Comm comm;
  DM       dm;
  PC       pc;
  PetscInt dim, d;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));

  /* Create solution */
  PetscCall(SolCreate(comm, &ns->sol));
  PetscCall(SolSetType(ns->sol, SOLFSM));
  PetscCall(SolSetMesh(ns->sol, ns->mesh));

  /* Create KSP */
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetDim(ns->mesh, &dim));
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

  ns->state = NS_STATE_SETUP;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSolveInit_FSM(NS ns)
{
  PetscInt dim;

  PetscFunctionBegin;

  PetscCall(MeshGetDim(ns->mesh, &dim));
  switch (dim) {
  case 2:
    PetscCall(NSFSMInterpolateVelocity2d_MeshCart(ns));
    PetscCall(NSFSMCalculateConvection2d_MeshCart(ns));
    break;

  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSolveIter_FSM(NS ns)
{
  PetscInt dim;

  PetscFunctionBegin;

  PetscCall(MeshGetDim(ns->mesh, &dim));
  switch (dim) {
  case 2:
    PetscCall(NSFSMCalculateIntermediateVelocity2d_MeshCart(ns));
    PetscCall(NSFSMCalculatePressureCorrection2d_MeshCart(ns));
    PetscCall(NSFSMUpdate2d_MeshCart(ns));
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

  for (d = 0; d < 3; ++d) PetscCall(KSPDestroy(&fsm->kspv[d]));
  PetscCall(KSPDestroy(&fsm->kspp));

  PetscCall(PetscFree(ns->data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView_FSM(NS ns, PetscViewer v)
{
  (void)ns;
  (void)v;

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCreate_FSM(NS ns)
{
  NS_FSM  *fsm;
  PetscInt d;

  PetscFunctionBegin;

  PetscCall(PetscNew(&fsm));
  ns->data = (void *)fsm;

  for (d = 0; d < 3; ++d) fsm->kspv[d] = NULL;
  fsm->kspp = NULL;

  ns->ops->setup      = NSSetup_FSM;
  ns->ops->solve_init = NSSolveInit_FSM;
  ns->ops->solve_iter = NSSolveIter_FSM;
  ns->ops->destroy    = NSDestroy_FSM;
  ns->ops->view       = NSView_FSM;

  PetscFunctionReturn(PETSC_SUCCESS);
}
