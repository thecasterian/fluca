#include <fluca/private/meshimpl.h>

FLUCA_EXTERN PetscErrorCode MeshCreate_Cart(Mesh);

PetscErrorCode MeshRegister(const char *type, PetscErrorCode (*function)(Mesh)) {
    PetscFunctionBegin;
    PetscCall(MeshInitializePackage());
    PetscCall(PetscFunctionListAdd(&MeshList, type, function));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshRegisterAll(void) {
    PetscFunctionBegin;
    PetscCall(MeshRegister(MESHCART, MeshCreate_Cart));
    MeshRegisterAllCalled = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
}
