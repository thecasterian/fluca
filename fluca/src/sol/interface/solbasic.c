#include <impl/solimpl.h>

extern PetscErrorCode SolView_CGNSCartesian(Sol, PetscViewer);

PetscClassId SOL_CLASSID = 0;

PetscFunctionList SolList = NULL;
PetscBool SolRegisterAllCalled = PETSC_FALSE;

PetscErrorCode SolCreate(MPI_Comm comm, Sol *sol) {
    Sol s;

    PetscFunctionBegin;

    *sol = NULL;
    PetscCall(SolInitializePackage());

    PetscCall(FlucaHeaderCreate(s, SOL_CLASSID, "Sol", "Solver Solution", "Sol", comm, SolDestroy, SolView));

    s->mesh = NULL;
    s->data = NULL;

    *sol = s;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolSetType(Sol sol, SolType type) {
    SolType old_type;
    PetscErrorCode (*impl_create)(Sol);
    PetscBool match;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);

    PetscCall(SolGetType(sol, &old_type));

    PetscCall(PetscObjectTypeCompare((PetscObject)sol, type, &match));
    if (match)
        PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(PetscFunctionListFind(SolList, type, &impl_create));
    PetscCheck(impl_create, PetscObjectComm((PetscObject)sol), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown mesh type: %s",
               type);

    if (old_type) {
        PetscTryTypeMethod(sol, destroy);
        PetscCall(PetscMemzero(sol->ops, sizeof(struct _SolOps)));
    }

    PetscCall(PetscObjectChangeTypeName((PetscObject)sol, type));
    PetscCall((*impl_create)(sol));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetType(Sol sol, SolType *type) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
    PetscCall(SolRegisterAll());
    *type = ((PetscObject)sol)->type_name;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolSetMesh(Sol sol, Mesh mesh) {
    PetscInt dim;
    DM dm;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 2);

    if (sol->mesh == mesh)
        PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(MeshDestroy(&sol->mesh));
    PetscCall(VecDestroy(&sol->u));
    PetscCall(VecDestroy(&sol->v));
    PetscCall(VecDestroy(&sol->w));
    PetscCall(VecDestroy(&sol->p));

    sol->mesh = mesh;
    PetscCall(PetscObjectReference((PetscObject)mesh));

    PetscCall(MeshGetDim(mesh, &dim));
    PetscCall(MeshGetDM(mesh, &dm));

    PetscCall(DMCreateLocalVector(dm, &sol->u));
    PetscCall(DMCreateLocalVector(dm, &sol->v));
    if (dim > 2)
        PetscCall(DMCreateLocalVector(dm, &sol->w));
    PetscCall(DMCreateLocalVector(dm, &sol->p));

    PetscTryTypeMethod(sol, setmesh, mesh);

    PetscCall(SolViewFromOptions(sol, NULL, "-sol_view"));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetMesh(Sol sol, Mesh *mesh) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
    *mesh = sol->mesh;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetVelocity(Sol sol, Vec *u, Vec *v, Vec *w) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
    if (u)
        *u = sol->u;
    if (v)
        *v = sol->v;
    if (w)
        *w = sol->w;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetPressure(Sol sol, Vec *p) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
    if (p)
        *p = sol->p;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolView(Sol sol, PetscViewer v) {
    PetscViewerFormat format;
    PetscMPIInt size;
    PetscBool iscgns, iscart;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
    if (!v)
        PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sol), &v));
    PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
    PetscCheckSameComm(sol, 1, v, 2);

    PetscCall(PetscViewerGetFormat(v, &format));
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sol), &size));
    if (format == PETSC_VIEWER_LOAD_BALANCE && size == 1)
        PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERCGNS, &iscgns));
    PetscCall(PetscObjectTypeCompare((PetscObject)sol->mesh, MESHCARTESIAN, &iscart));
    if (iscgns) {
        if (iscart)
            PetscCall(SolView_CGNSCartesian(sol, v));
    }

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sol, v));
    PetscTryTypeMethod(sol, view, v);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolViewFromOptions(Sol sol, PetscObject obj, const char *name) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
    PetscCall(PetscObjectViewFromOptions((PetscObject)sol, obj, name));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolDestroy(Sol *sol) {
    PetscFunctionBegin;

    if (!*sol)
        PetscFunctionReturn(PETSC_SUCCESS);
    PetscValidHeaderSpecific(*sol, SOL_CLASSID, 1);

    if (--((PetscObject)(*sol))->refct > 0) {
        *sol = NULL;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(MeshDestroy(&(*sol)->mesh));
    PetscCall(VecDestroy(&(*sol)->u));
    PetscCall(VecDestroy(&(*sol)->v));
    PetscCall(VecDestroy(&(*sol)->w));
    PetscCall(VecDestroy(&(*sol)->p));

    PetscTryTypeMethod((*sol), destroy);
    PetscCall(PetscHeaderDestroy(sol));

    PetscFunctionReturn(PETSC_SUCCESS);
}
