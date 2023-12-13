#include <fluca/private/nsimpl.h>

extern PetscErrorCode NSCreate_FSM(NS);

PetscErrorCode NSRegister(const char *type, PetscErrorCode (*function)(NS)) {
    PetscFunctionBegin;
    PetscCall(NSInitializePackage());
    PetscCall(PetscFunctionListAdd(&NSList, type, function));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSRegisterAll(void) {
    PetscFunctionBegin;
    PetscCall(NSRegister(NSFSM, NSCreate_FSM));
    NSRegisterAllCalled = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
}
