#include <fluca/private/physimpl.h>

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
  /* No subtypes registered yet -- Phase 2 will add PHYSINS here */
  PetscFunctionReturn(PETSC_SUCCESS);
}
