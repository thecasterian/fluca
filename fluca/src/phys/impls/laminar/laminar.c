#include <fluca/private/physimpl.h>

/* Cell velocity, face-normal velocity and cell pressure */
static PetscErrorCode PhysSetUp_Laminar(Phys phys)
{
  PetscFunctionBegin;
  PetscCall(PhysDeclareField_Internal(phys, PHYS_FIELD_VELOCITY, PHYS_FIELD_ELEMENT, phys->dim));
  PetscCall(PhysDeclareField_Internal(phys, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_FACE, 1));
  PetscCall(PhysDeclareField_Internal(phys, PHYS_FIELD_PRESSURE, PHYS_FIELD_ELEMENT, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreate_Laminar(Phys phys)
{
  PetscFunctionBegin;
  phys->data       = NULL;
  phys->ops->setup = PhysSetUp_Laminar;
  PetscFunctionReturn(PETSC_SUCCESS);
}
