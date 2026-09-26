#include <fluca/private/physimpl.h>

/* Cell velocity, face-normal velocity and cell pressure */
PetscErrorCode PhysCreate_Laminar(Phys phys)
{
  PetscFunctionBegin;
  phys->data = NULL;
  PetscCall(PhysDeclareField(phys, PHYS_FIELD_VELOCITY, PHYS_FIELD_ELEMENT, PETSC_DECIDE));
  PetscCall(PhysDeclareField(phys, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_FACE, 1));
  PetscCall(PhysDeclareField(phys, PHYS_FIELD_PRESSURE, PHYS_FIELD_ELEMENT, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
