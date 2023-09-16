#include <fluca/private/meshimpl.h>

FLUCA_EXTERN PetscErrorCode MeshCreate_Cartesian(Mesh);

PetscErrorCode MeshRegister(const char *type, PetscErrorCode (*function)(Mesh)) {
    PetscFunctionBegin;
    PetscCall(MeshInitializePackage());
    PetscCall(PetscFunctionListAdd(&MeshList, type, function));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshRegisterAll(void) {
    PetscFunctionBegin;
    PetscCall(MeshRegister(MESHCARTESIAN, MeshCreate_Cartesian));
    MeshRegisterAllCalled = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
}
