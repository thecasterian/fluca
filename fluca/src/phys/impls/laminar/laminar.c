#include <fluca/private/physimpl.h>

/* PhysSetUp() has declared the common fields. The boundary conditions of PHYSLAMINAR prescribe the
   velocity only, so the pressure is determined only up to a constant. */
static PetscErrorCode PhysSetUp_Laminar(Phys phys)
{
  PetscFunctionBegin;
  PetscCall(PhysDeclareConstantNullSpace_Internal(phys, PHYS_FIELD_PRESSURE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreate_Laminar(Phys phys)
{
  PetscFunctionBegin;
  phys->data       = NULL;
  phys->ops->setup = PhysSetUp_Laminar;
  PetscFunctionReturn(PETSC_SUCCESS);
}
