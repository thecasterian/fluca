#include <fluca/private/flucaibimpl.h>

FLUCA_EXTERN PetscErrorCode FlucaIBCreate_None(FlucaIB);

PetscErrorCode FlucaIBRegister(const char sname[], PetscErrorCode (*function)(FlucaIB))
{
  PetscFunctionBegin;
  PetscCall(FlucaIBInitializePackage());
  PetscCall(PetscFunctionListAdd(&FlucaIBList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBRegisterAll(void)
{
  PetscFunctionBegin;
  if (FlucaIBRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  FlucaIBRegisterAllCalled = PETSC_TRUE;

  PetscCall(FlucaIBRegister(FLUCAIB_NONE, FlucaIBCreate_None));
  PetscFunctionReturn(PETSC_SUCCESS);
}
