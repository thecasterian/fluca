#include <fluca/private/physimpl.h>

FLUCA_EXTERN PetscErrorCode PhysCreate_Laminar(Phys);

PetscErrorCode PhysRegister(const char sname[], PetscErrorCode (*function)(Phys))
{
  PetscFunctionBegin;
  PetscCall(PhysInitializePackage());
  PetscCall(PetscFunctionListAdd(&PhysList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysRegisterAll(void)
{
  PetscFunctionBegin;
  if (PhysRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PhysRegisterAllCalled = PETSC_TRUE;

  PetscCall(PhysRegister(PHYSLAMINAR, PhysCreate_Laminar));
  PetscFunctionReturn(PETSC_SUCCESS);
}
