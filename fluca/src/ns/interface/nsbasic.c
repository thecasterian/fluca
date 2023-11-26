#include <fluca/private/nsimpl.h>

PetscClassId NS_CLASSID = 0;
PetscLogEvent NS_SetUp = 0;
PetscLogEvent NS_Solve = 0;

PetscFunctionList NSList = NULL;
PetscBool NSRegisterAllCalled = PETSC_FALSE;

extern PetscErrorCode NSView_CGNS(NS, PetscViewer);

PetscErrorCode NSCreate(MPI_Comm comm, NS *ns) {
    NS n;

    PetscFunctionBegin;

    *ns = NULL;
    PetscCall(NSInitializePackage());

    PetscCall(FlucaHeaderCreate(n, NS_CLASSID, "NS", "Navier-Stokes solver", "NS", comm, NSDestroy, NSView));

    n->rho = 0.0;
    n->mu = 0.0;
    n->dt = 0.0;

    n->step = 0;
    n->t = 0.0;
    n->mesh = NULL;
    n->sol = NULL;
    n->data = NULL;

    n->state = NS_STATE_INITIAL;

    *ns = n;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetType(NS ns, NSType type) {
    NSType old_type;
    PetscErrorCode (*impl_create)(NS);
    PetscBool match;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    PetscCall(NSGetType(ns, &old_type));

    PetscCall(PetscObjectTypeCompare((PetscObject)ns, type, &match));
    if (match)
        PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(PetscFunctionListFind(NSList, type, &impl_create));
    PetscCheck(impl_create, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown ns type: %s", type);

    if (old_type) {
        PetscTryTypeMethod(ns, destroy);
        PetscCall(PetscMemzero(ns->ops, sizeof(struct _NSOps)));
    }

    ns->state = NS_STATE_INITIAL;
    PetscCall(PetscObjectChangeTypeName((PetscObject)ns, type));
    PetscCall((*impl_create)(ns));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetType(NS ns, NSType *type) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    PetscCall(NSRegisterAll());
    *type = ((PetscObject)ns)->type_name;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetUp(NS ns) {
    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    if (ns->state >= NS_STATE_SETUP)
        PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(PetscLogEventBegin(NS_SetUp, (PetscObject)ns, 0, 0, 0));

    /* Set default type */
    if (!((PetscObject)ns)->type_name)
        PetscCall(NSSetType(ns, NSFSM));

    /* Validate */
    PetscCheck(ns->mesh, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Mesh not set");
    PetscCheck(!ns->sol, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Solution already set");

    /* Call specific type setup */
    PetscTryTypeMethod(ns, setup);

    PetscCall(PetscLogEventEnd(NS_SetUp, (PetscObject)ns, 0, 0, 0));

    /* Viewers */
    PetscCall(NSViewFromOptions(ns, NULL, "-ns_view"));

    ns->state = NS_STATE_SETUP;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSolve(NS ns, PetscInt num_iters) {
    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    PetscCall(PetscLogEventBegin(NS_Solve, (PetscObject)ns, 0, 0, 0));

    PetscTryTypeMethod(ns, solve, num_iters);

    PetscCall(PetscLogEventEnd(NS_Solve, (PetscObject)ns, 0, 0, 0));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetSol(NS ns, Sol *sol) {
    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    PetscCheck(ns->state >= NS_STATE_SETUP, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE,
               "NS not set up");

    if (sol)
        *sol = ns->sol;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSDestroy(NS *ns) {
    PetscFunctionBegin;

    if (!*ns)
        PetscFunctionReturn(PETSC_SUCCESS);
    PetscValidHeaderSpecific((*ns), NS_CLASSID, 1);

    if (--((PetscObject)(*ns))->refct > 0) {
        *ns = NULL;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(MeshDestroy(&(*ns)->mesh));
    PetscCall(SolDestroy(&(*ns)->sol));

    PetscTryTypeMethod((*ns), destroy);
    PetscCall(PetscHeaderDestroy(ns));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView(NS ns, PetscViewer v) {
    PetscBool isascii, iscgns;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    if (!v)
        PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ns), &v));
    PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
    PetscCheckSameComm(ns, 1, v, 2);

    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));
    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERCGNS, &iscgns));

    if (isascii) {
        PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ns, v));
        PetscCall(
            PetscViewerASCIIPrintf(v, "Density: %g, Viscosity: %g, Time step size: %g\n", ns->rho, ns->mu, ns->dt));
        PetscCall(PetscViewerASCIIPrintf(v, "Current time step: %d, Current time: %g\n", ns->step, ns->t));
        PetscCall(PetscViewerASCIIPushTab(v));
        PetscTryTypeMethod(ns, view, v);
        PetscCall(PetscViewerASCIIPopTab(v));
    } else if (iscgns) {
        PetscCall(NSView_CGNS(ns, v));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewFromOptions(NS ns, PetscObject obj, const char *name) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    PetscCall(PetscObjectViewFromOptions((PetscObject)ns, obj, name));
    PetscFunctionReturn(PETSC_SUCCESS);
}
