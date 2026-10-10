#include <fluca/private/ibbodyimpl.h>

FLUCA_EXTERN PetscErrorCode IBBodyCreate_LevelSet(IBBody);

PetscErrorCode IBBodyRegister(const char type[], PetscErrorCode (*function)(IBBody))
{
  PetscFunctionBegin;
  PetscCall(IBBodyInitializePackage());
  PetscCall(PetscFunctionListAdd(&IBBodyList, type, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyRegisterAll(void)
{
  PetscFunctionBegin;
  if (IBBodyRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  IBBodyRegisterAllCalled = PETSC_TRUE;
  PetscCall(IBBodyRegister(IBBODYLEVELSET, IBBodyCreate_LevelSet));
  PetscFunctionReturn(PETSC_SUCCESS);
}
