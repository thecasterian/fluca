#include <fluca/private/physimpl.h>

FLUCA_EXTERN PetscErrorCode PhysCreate_INS(Phys);

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
  PetscCall(PhysRegister(PHYSINS, PhysCreate_INS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
