#include <fluca/private/segimpl.h>
#include <flucaviewer.h>
#include <flucasys.h>

PetscClassId  SEG_CLASSID = 0;
PetscLogEvent SEG_SetUp   = 0;
PetscLogEvent SEG_Step    = 0;

PetscFunctionList SegList              = NULL;
PetscBool         SegRegisterAllCalled = PETSC_FALSE;

PetscErrorCode SegCreate(MPI_Comm comm, Seg *seg)
{
  Seg s;

  PetscFunctionBegin;
  PetscAssertPointer(seg, 2);

  PetscCall(SegInitializePackage());
  PetscCall(FlucaHeaderCreate(s, SEG_CLASSID, "Seg", "Segregated Time Integrator", "Seg", comm, SegDestroy, SegView));
  s->sol         = NULL;
  s->ib          = NULL;
  s->dm          = NULL;
  s->t           = 0.;
  s->dt          = PETSC_DECIDE;
  s->max_time    = PETSC_MAX_REAL;
  s->step_number = 0;
  s->max_steps   = PETSC_INT_MAX;
  s->rhsfn       = NULL;
  s->rhsfn_ctx   = NULL;
  s->setupcalled = PETSC_FALSE;
  s->data        = NULL;

  *seg = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetType(Seg seg, SegType type)
{
  SegType old_type;
  PetscErrorCode (*impl_create)(Seg);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);

  PetscCall(SegGetType(seg, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)seg, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(SegList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Seg type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(seg, destroy);
    PetscCall(PetscMemzero(seg->ops, sizeof(struct _SegOps)));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)seg, type));
  PetscCall((*impl_create)(seg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetType(Seg seg, SegType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(SegRegisterAll());
  *type = ((PetscObject)seg)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetUp(Seg seg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  if (seg->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(SEG_SetUp, (PetscObject)seg, 0, 0, 0));

  PetscCheck(((PetscObject)seg)->type_name, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Seg type not set. Call SegSetType() first");
  PetscCheck(seg->dm, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "DM not set. Call SegSetDM() first");
  PetscCheck(seg->sol, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Solution not set. Call SegSetSolution() first");

  PetscTryTypeMethod(seg, setup);

  PetscCall(PetscLogEventEnd(SEG_SetUp, (PetscObject)seg, 0, 0, 0));

  seg->setupcalled = PETSC_TRUE;

  PetscCall(SegViewFromOptions(seg, NULL, "-seg_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegReset(Seg seg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  if (!seg->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscTryTypeMethod(seg, reset);
  seg->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegDestroy(Seg *seg)
{
  PetscFunctionBegin;
  if (!*seg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*seg), SEG_CLASSID, 1);

  if (--((PetscObject)(*seg))->refct > 0) {
    *seg = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(SegReset(*seg));
  PetscTryTypeMethod((*seg), destroy);
  PetscCall(FlucaIBDestroy(&(*seg)->ib));

  PetscCall(PetscHeaderDestroy(seg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegView(Seg seg, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)seg), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(seg, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)seg, viewer));
    if (seg->setupcalled) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Time step size: %g\n", (double)seg->dt));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Maximum time: %g\n", (double)seg->max_time));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Maximum steps: %" PetscInt_FMT "\n", seg->max_steps));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }

  PetscTryTypeMethod(seg, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegViewFromOptions(Seg seg, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)seg, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetIB(Seg seg, FlucaIB ib)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 2);
  PetscCheckSameComm(seg, 1, ib, 2);
  PetscCall(FlucaIBDestroy(&seg->ib));
  seg->ib = ib;
  PetscCall(PetscObjectReference((PetscObject)ib));
  PetscCall(FlucaIBGetDM(ib, &seg->dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetIB(Seg seg, FlucaIB *ib)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(ib, 2);
  *ib = seg->ib;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetDM(Seg seg, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  *dm = seg->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetSolution(Seg seg, Vec sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidHeaderSpecific(sol, VEC_CLASSID, 2);
  PetscCheckSameComm(seg, 1, sol, 2);
  seg->sol = sol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetSolution(Seg seg, Vec *sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(sol, 2);
  *sol = seg->sol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetRHSFunction(Seg seg, SegRHSFn *fn, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  seg->rhsfn     = fn;
  seg->rhsfn_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}
