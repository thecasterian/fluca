#include <fluca/private/nsimpl.h>

FLUCA_EXTERN PetscErrorCode NSCreate_CNLinear(NS);

PetscErrorCode NSRegister(const char type[], PetscErrorCode (*function)(NS))
{
  PetscFunctionBegin;
  PetscCall(NSInitializePackage());
  PetscCall(PetscFunctionListAdd(&NSList, type, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSRegisterAll(void)
{
  PetscFunctionBegin;
  if (NSRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(NSRegister(NSCNLINEAR, NSCreate_CNLinear));
  NSRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
