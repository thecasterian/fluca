#include <fluca/private/ibbodyimpl.h>

FLUCA_EXTERN PetscErrorCode IBBodyCreate_LevelSet(IBBody);
FLUCA_EXTERN PetscErrorCode IBBodyCreate_Surface(IBBody);

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
  PetscCall(IBBodyRegister(IBBODYSURFACE, IBBodyCreate_Surface));
  PetscFunctionReturn(PETSC_SUCCESS);
}
