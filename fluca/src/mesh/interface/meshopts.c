#include <impl/meshimpl.h>

PetscErrorCode MeshSetDim(Mesh mesh, PetscInt dim) {
    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    PetscCheck(dim > 0, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Dimension must be positive");

    mesh->dim = dim;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetDim(Mesh mesh, PetscInt *dim) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    *dim = mesh->dim;
    PetscFunctionReturn(PETSC_SUCCESS);
}
