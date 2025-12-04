#include <petscpc.h>
#include <flucans.h>

FLUCA_EXTERN PetscErrorCode PCCreate_ABF(PC);

static PetscBool NSPCRegisterAllCalled = PETSC_FALSE;

PetscErrorCode NSPCRegisterAll(void)
{
  PetscFunctionBegin;
  if (NSPCRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PCRegister(PCABF, PCCreate_ABF));
  NSPCRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
