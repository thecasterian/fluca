#include <fluca/private/physinsimpl.h>

static PetscErrorCode PhysCreateSolutionDM_INS(Phys phys)
{
  DM cdm;

  PetscFunctionBegin;
  /* Create cell-centered DMStag: dim+1 element DOFs (velocity components + pressure) */
  switch (phys->dim) {
  case 2:
    PetscCall(DMStagCreateCompatibleDMStag(phys->base_dm, 0, 0, phys->dim + 1, 0, &phys->sol_dm));
    break;
  case 3:
    PetscCall(DMStagCreateCompatibleDMStag(phys->base_dm, 0, 0, 0, phys->dim + 1, &phys->sol_dm));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, phys->dim);
  }
  /* Share coordinates from base DM */
  PetscCall(DMStagSetCoordinateDMType(phys->sol_dm, DMPRODUCT));
  PetscCall(DMGetCoordinateDM(phys->base_dm, &cdm));
  PetscCall(DMSetCoordinateDM(phys->sol_dm, cdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysSetFromOptions_INS(Phys phys, PetscOptionItems PetscOptionsObject)
{
  Phys_INS *ins = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "INS Options");
  PetscCall(PetscOptionsReal("-phys_ins_density", "Density", "PhysINSSetDensity", ins->rho, &ins->rho, NULL));
  PetscCall(PetscOptionsReal("-phys_ins_viscosity", "Dynamic viscosity", "PhysINSSetViscosity", ins->mu, &ins->mu, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysSetUp_INS(Phys phys)
{
  PetscFunctionBegin;
  PetscCall(PhysINSBuildOperators_Internal(phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysDestroy_INS(Phys phys)
{
  Phys_INS *ins = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  PetscCall(PhysINSDestroyOperators_Internal(phys));
  PetscCall(MatDestroy(&ins->J));
  PetscCall(ISDestroy(&ins->is_vel));
  PetscCall(ISDestroy(&ins->is_p));
  PetscCall(MatNullSpaceDestroy(&ins->nullspace));
  PetscCall(PetscFree(phys->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysView_INS(Phys phys, PetscViewer viewer)
{
  Phys_INS *ins = (Phys_INS *)phys->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Density: %g\n", (double)ins->rho));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Viscosity: %g\n", (double)ins->mu));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreate_INS(Phys phys)
{
  Phys_INS *ins;
  PetscInt  f;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ins));
  ins->rho = 1.0;
  ins->mu  = 1.0;

  /* Initialize BCs to NONE */
  for (f = 0; f < PHYS_INS_MAX_FACES; f++) {
    ins->bcs[f].type = PHYS_INS_BC_NONE;
    ins->bcs[f].fn   = NULL;
    ins->bcs[f].ctx  = NULL;
  }

  /* Initialize operators to NULL */
  for (f = 0; f < PHYS_INS_MAX_DIM; f++) {
    ins->fd_laplacian[f] = NULL;
    ins->fd_grad_p[f]    = NULL;
    ins->fd_div[f]       = NULL;
  }
  ins->fd_pstab            = NULL;
  ins->J                   = NULL;
  ins->is_vel              = NULL;
  ins->is_p                = NULL;
  ins->nullspace           = NULL;
  ins->temp                = NULL;
  ins->alpha               = 0.0;
  ins->has_pressure_outlet = PETSC_FALSE;

  phys->data                  = ins;
  phys->ops->createsolutiondm = PhysCreateSolutionDM_INS;
  phys->ops->setfromoptions   = PhysSetFromOptions_INS;
  phys->ops->setup            = PhysSetUp_INS;
  phys->ops->destroy          = PhysDestroy_INS;
  phys->ops->view             = PhysView_INS;
  phys->ops->setupts          = PhysSetUpTS_INS;
  phys->ops->computeifuncion  = PhysComputeIFunction_INS;
  phys->ops->computeijacobian = PhysComputeIJacobian_INS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSSetDensity(Phys phys, PetscReal rho)
{
  Phys_INS *ins;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscValidLogicalCollectiveReal(phys, rho, 2);
  PetscCheck(rho > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Density must be positive, got %g", (double)rho);
  ins      = (Phys_INS *)phys->data;
  ins->rho = rho;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSGetDensity(Phys phys, PetscReal *rho)
{
  Phys_INS *ins;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscAssertPointer(rho, 2);
  ins  = (Phys_INS *)phys->data;
  *rho = ins->rho;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSSetViscosity(Phys phys, PetscReal mu)
{
  Phys_INS *ins;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscValidLogicalCollectiveReal(phys, mu, 2);
  PetscCheck(mu >= 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Viscosity must be non-negative, got %g", (double)mu);
  ins     = (Phys_INS *)phys->data;
  ins->mu = mu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSGetViscosity(Phys phys, PetscReal *mu)
{
  Phys_INS *ins;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscAssertPointer(mu, 2);
  ins = (Phys_INS *)phys->data;
  *mu = ins->mu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSSetBoundaryCondition(Phys phys, PetscInt face, PhysINSBC bc)
{
  Phys_INS *ins;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscCheck(face >= 0 && face < PHYS_INS_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_INS_MAX_FACES);
  ins            = (Phys_INS *)phys->data;
  ins->bcs[face] = bc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSGetBoundaryCondition(Phys phys, PetscInt face, PhysINSBC *bc)
{
  Phys_INS *ins;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscAssertPointer(bc, 3);
  PetscCheck(face >= 0 && face < PHYS_INS_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_INS_MAX_FACES);
  ins = (Phys_INS *)phys->data;
  *bc = ins->bcs[face];
  PetscFunctionReturn(PETSC_SUCCESS);
}
