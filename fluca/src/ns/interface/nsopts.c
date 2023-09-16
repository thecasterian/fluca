#include <fluca/private/nsimpl.h>

PetscErrorCode NSSetMesh(NS ns, Mesh mesh) {
    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 2);
    PetscCheckSameComm(ns, 1, mesh, 2);

    if (ns->mesh == mesh)
        PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(MeshDestroy(&ns->mesh));
    PetscCall(SolDestroy(&ns->sol));

    ns->mesh = mesh;
    PetscCall(PetscObjectReference((PetscObject)mesh));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetMesh(NS ns, Mesh *mesh) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    *mesh = ns->mesh;
    PetscFunctionReturn(PETSC_SUCCESS);
}
