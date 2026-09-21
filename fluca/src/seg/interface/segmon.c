#include <fluca/private/segimpl.h>
#include <flucaviewer.h>

PetscErrorCode SegMonitorSet(Seg seg, PetscErrorCode (*mon)(Seg, void *), void *mon_ctx, PetscErrorCode (*mon_ctx_destroy)(void **))
{
  PetscInt  i;
  PetscBool identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  for (i = 0; i < seg->num_mons; ++i) {
    PetscCall(PetscMonitorCompare((PetscErrorCode(*)(void))mon, mon_ctx, mon_ctx_destroy, (PetscErrorCode(*)(void))seg->mons[i], seg->mon_ctxs[i], seg->mon_ctx_destroys[i], &identical));
    if (identical) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(seg->num_mons < MAXSEGMONITORS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many monitors set");

  seg->mons[seg->num_mons]             = mon;
  seg->mon_ctxs[seg->num_mons]         = mon_ctx;
  seg->mon_ctx_destroys[seg->num_mons] = mon_ctx_destroy;
  ++seg->num_mons;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegMonitorCancel(Seg seg)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  for (i = 0; i < seg->num_mons; ++i)
    if (seg->mon_ctx_destroys[i]) PetscCall((*seg->mon_ctx_destroys[i])(&seg->mon_ctxs[i]));
  seg->num_mons = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegMonitor(Seg seg)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  for (i = 0; i < seg->num_mons; ++i) PetscCall((*seg->mons[i])(seg, seg->mon_ctxs[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegMonitorSetFromOptions(Seg seg, const char name[], const char help[], const char manual[], PetscErrorCode (*mon)(Seg, PetscViewerAndFormat *), PetscErrorCode (*mon_setup)(Seg, PetscViewerAndFormat *))
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(FlucaOptionsCreateViewer(PetscObjectComm((PetscObject)seg), ((PetscObject)seg)->options, ((PetscObject)seg)->prefix, name, &viewer, &format, &flg));
  if (flg) {
    PetscViewerAndFormat *vf;
    char                  interval_key[1024];

    PetscCall(PetscSNPrintf(interval_key, sizeof(interval_key), "%s_interval", name));
    PetscCall(PetscViewerAndFormatCreate(viewer, format, &vf));
    vf->view_interval = 1;
    PetscCall(PetscOptionsGetInt(((PetscObject)seg)->options, ((PetscObject)seg)->prefix, interval_key, &vf->view_interval, NULL));

    PetscCall(PetscViewerDestroy(&viewer));
    if (mon_setup) PetscCall((*mon_setup)(seg, vf));
    PetscCall(SegMonitorSet(seg, (PetscErrorCode(*)(Seg, void *))mon, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegMonitorDefault(Seg seg, PetscViewerAndFormat *vf)
{
  PetscViewer viewer = vf->viewer;
  PetscBool   isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (vf->view_interval > 0 && seg->step % vf->view_interval == 0) {
    PetscCall(PetscViewerPushFormat(viewer, vf->format));
    if (isascii) {
      PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject)seg)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " Seg dt %g time %g\n", seg->step, (double)seg->dt, (double)seg->t));
      PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)seg)->tablevel));
    }
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
