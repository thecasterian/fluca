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
    if (mesh)
        *mesh = ns->mesh;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetDensity(NS ns, PetscReal rho) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    ns->rho = rho;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetDensity(NS ns, PetscReal *rho) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    if (rho)
        *rho = ns->rho;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetViscosity(NS ns, PetscReal mu) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    ns->mu = mu;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetViscosity(NS ns, PetscReal *mu) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    if (mu)
        *mu = ns->mu;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTimeStepSize(NS ns, PetscReal dt) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    ns->dt = dt;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTimeStepSize(NS ns, PetscReal *dt) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    if (dt)
        *dt = ns->dt;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTimeStep(NS ns, PetscInt step) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    ns->step = step;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTimeStep(NS ns, PetscInt *step) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    if (step)
        *step = ns->step;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTime(NS ns, PetscReal t) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    ns->t = t;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTime(NS ns, PetscReal *t) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    if (t)
        *t = ns->t;
    PetscFunctionReturn(PETSC_SUCCESS);
}
