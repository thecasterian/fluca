#include <fluca/private/solimpl.h>

FLUCA_EXTERN PetscErrorCode SolCreate_FSM(Sol);

PetscErrorCode SolRegister(const char type[], PetscErrorCode (*function)(Sol))
{
  PetscFunctionBegin;
  PetscCall(SolInitializePackage());
  PetscCall(PetscFunctionListAdd(&SolList, type, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolRegisterAll(void)
{
  PetscFunctionBegin;
  PetscCall(SolRegister(SOLFSM, SolCreate_FSM));
  SolRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
