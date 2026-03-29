#include <fluca/private/segimpl.h>

FLUCA_EXTERN PetscErrorCode SegCreate_SRK(Seg);

PetscErrorCode SegRegister(const char sname[], PetscErrorCode (*function)(Seg))
{
  PetscFunctionBegin;
  PetscCall(SegInitializePackage());
  PetscCall(PetscFunctionListAdd(&SegList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegRegisterAll(void)
{
  PetscFunctionBegin;
  if (SegRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  SegRegisterAllCalled = PETSC_TRUE;

  PetscCall(SegRegister(SEGSRK, SegCreate_SRK));
  PetscFunctionReturn(PETSC_SUCCESS);
}
