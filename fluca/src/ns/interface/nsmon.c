#include <fluca/private/nsimpl.h>

FLUCA_EXTERN PetscErrorCode NSMonitorSet(NS ns, PetscErrorCode (*mon)(NS, PetscInt, PetscReal, Sol, void *),
                                         void *mon_ctx, PetscErrorCode (*mon_ctx_destroy)(void **)) {
    PetscInt i;
    PetscBool identical;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    for (i = 0; i < ns->num_mons; i++) {
        PetscCall(PetscMonitorCompare((PetscErrorCode(*)(void))mon, mon_ctx, mon_ctx_destroy,
                                      (PetscErrorCode(*)(void))ns->mons[i], ns->mon_ctxs[i], ns->mon_ctx_destroys[i],
                                      &identical));
        if (identical)
            PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCheck(ns->num_mons < MAXNSMONITORS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many monitors set");

    ns->mons[ns->num_mons] = mon;
    ns->mon_ctxs[ns->num_mons] = mon_ctx;
    ns->mon_ctx_destroys[ns->num_mons] = mon_ctx_destroy;
    ns->num_mons++;

    PetscFunctionReturn(PETSC_SUCCESS);
}

FLUCA_EXTERN PetscErrorCode NSMonitorCancel(NS ns) {
    PetscInt i;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    for (i = 0; i < ns->num_mons; i++)
        if (ns->mon_ctx_destroys[i])
            PetscCall((*ns->mon_ctx_destroys[i])(&ns->mon_ctxs[i]));
    ns->num_mons = 0;

    PetscFunctionReturn(PETSC_SUCCESS);
}

FLUCA_EXTERN PetscErrorCode NSMonitor(NS ns, PetscInt step, PetscReal time, Sol sol) {
    PetscInt i;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    for (i = 0; i < ns->num_mons; i++)
        PetscCall((*ns->mons[i])(ns, step, time, sol, ns->mon_ctxs[i]));

    PetscFunctionReturn(PETSC_SUCCESS);
}

FLUCA_EXTERN PetscErrorCode NSMonitorDefault(NS ns, PetscInt step, PetscReal time, Sol sol, void *v) {
    PetscViewer viewer = (PetscViewer)v;
    PetscBool isascii;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    PetscValidHeaderSpecific(sol, SOL_CLASSID, 4);
    PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 5);

    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

    if (isascii)
        PetscCall(PetscViewerASCIIPrintf(viewer, "step %" PetscInt_FMT ", time %g\n", step, (double)time));

    PetscFunctionReturn(PETSC_SUCCESS);
}
