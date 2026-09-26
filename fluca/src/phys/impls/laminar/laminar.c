#include <fluca/private/physimpl.h>

/* One element-located DMStag holding the velocity components and the pressure */
static PetscErrorCode PhysCreateSolutionDM_Laminar(Phys phys)
{
  DM cdm;

  PetscFunctionBegin;
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
  PetscCall(DMStagSetCoordinateDMType(phys->sol_dm, DMPRODUCT));
  PetscCall(DMGetCoordinateDM(phys->base_dm, &cdm));
  PetscCall(DMSetCoordinateDM(phys->sol_dm, cdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreate_Laminar(Phys phys)
{
  PetscFunctionBegin;
  phys->data                  = NULL;
  phys->ops->createsolutiondm = PhysCreateSolutionDM_Laminar;
  PetscFunctionReturn(PETSC_SUCCESS);
}
