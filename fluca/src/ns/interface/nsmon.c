#include <fluca/private/nsimpl.h>

PetscErrorCode NSMonitorSet(NS ns, PetscErrorCode (*mon)(NS, PetscInt, PetscReal, Sol, void *), void *mon_ctx,
                            PetscErrorCode (*mon_ctx_destroy)(void **)) {
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

PetscErrorCode NSMonitorCancel(NS ns) {
    PetscInt i;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    for (i = 0; i < ns->num_mons; i++)
        if (ns->mon_ctx_destroys[i])
            PetscCall((*ns->mon_ctx_destroys[i])(&ns->mon_ctxs[i]));
    ns->num_mons = 0;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitor(NS ns, PetscInt step, PetscReal time, Sol sol) {
    PetscInt i;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    for (i = 0; i < ns->num_mons; i++)
        PetscCall((*ns->mons[i])(ns, step, time, sol, ns->mon_ctxs[i]));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorSetFrequency(NS ns, PetscInt freq) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    ns->mon_freq = freq;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorSetFromOptions(NS ns, const char *name, const char *help, const char *manual,
                                       PetscErrorCode (*mon)(NS, PetscInt, PetscReal, Sol, PetscViewerAndFormat *),
                                       PetscErrorCode (*mon_setup)(NS, PetscViewerAndFormat *)) {
    PetscViewer viewer;
    PetscViewerFormat format;
    PetscBool flg;

    (void)help;
    (void)manual;

    PetscFunctionBegin;

    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)ns), ((PetscObject)ns)->options,
                                    ((PetscObject)ns)->prefix, name, &viewer, &format, &flg));
    if (flg) {
        PetscViewerAndFormat *vf;
        PetscCall(PetscViewerAndFormatCreate(viewer, format, &vf));
        if (mon_setup)
            PetscCall((*mon_setup)(ns, vf));
        PetscCall(NSMonitorSet(ns, (PetscErrorCode(*)(NS, PetscInt, PetscReal, Sol, void *))mon, vf,
                               (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorDefault(NS ns, PetscInt step, PetscReal time, Sol sol, PetscViewerAndFormat *vf) {
    PetscBool isascii;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
    PetscValidHeaderSpecific(sol, SOL_CLASSID, 4);

    PetscCall(PetscObjectTypeCompare((PetscObject)vf->viewer, PETSCVIEWERASCII, &isascii));

    if (isascii)
        PetscCall(PetscViewerASCIIPrintf(vf->viewer, "step %" PetscInt_FMT ", time %g\n", step, (double)time));

    PetscFunctionReturn(PETSC_SUCCESS);
}
