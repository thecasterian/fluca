#include <fluca/private/nsimpl.h>

PetscErrorCode NSMonitorSet(NS ns, PetscErrorCode (*mon)(NS, void *), void *mon_ctx, PetscErrorCode (*mon_ctx_destroy)(void **))
{
  PetscInt  i;
  PetscBool identical;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

  for (i = 0; i < ns->num_mons; ++i) {
    PetscCall(PetscMonitorCompare((PetscErrorCode(*)(void))mon, mon_ctx, mon_ctx_destroy, (PetscErrorCode(*)(void))ns->mons[i], ns->mon_ctxs[i], ns->mon_ctx_destroys[i], &identical));
    if (identical) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(ns->num_mons < MAXNSMONITORS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many monitors set");

  ns->mons[ns->num_mons]             = mon;
  ns->mon_ctxs[ns->num_mons]         = mon_ctx;
  ns->mon_ctx_destroys[ns->num_mons] = mon_ctx_destroy;
  ++ns->num_mons;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorCancel(NS ns)
{
  PetscInt i;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

  for (i = 0; i < ns->num_mons; ++i)
    if (ns->mon_ctx_destroys[i]) PetscCall((*ns->mon_ctx_destroys[i])(&ns->mon_ctxs[i]));
  ns->num_mons = 0;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitor(NS ns)
{
  DM       dm;
  PetscInt i;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(DMSetOutputSequenceNumber(dm, ns->step, ns->t));

  for (i = 0; i < ns->num_mons; ++i) PetscCall((*ns->mons[i])(ns, ns->mon_ctxs[i]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorSetFrequency(NS ns, PetscInt freq)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->mon_freq = freq;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorSetFromOptions(NS ns, const char name[], const char help[], const char manual[], PetscErrorCode (*mon)(NS, PetscViewerAndFormat *), PetscErrorCode (*mon_setup)(NS, PetscViewerAndFormat *))
{
  (void)help;
  (void)manual;

  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;

  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)ns), ((PetscObject)ns)->options, ((PetscObject)ns)->prefix, name, &viewer, &format, &flg));
  if (flg) {
    PetscViewerAndFormat *vf;

    PetscCall(PetscViewerAndFormatCreate(viewer, format, &vf));
    if (mon_setup) PetscCall((*mon_setup)(ns, vf));
    PetscCall(NSMonitorSet(ns, (PetscErrorCode(*)(NS, void *))mon, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorDefault(NS ns, PetscViewerAndFormat *vf)
{
  (void)ns;

  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscBool         isascii;

  PetscFunctionBegin;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  PetscCall(PetscViewerPushFormat(viewer, format));

  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "step %" PetscInt_FMT ", time %g\n", ns->step, (double)ns->t));

  PetscCall(PetscViewerPopFormat(viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSMonitorSolution(NS ns, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;

  PetscFunctionBegin;
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(SolView(ns->sol, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
