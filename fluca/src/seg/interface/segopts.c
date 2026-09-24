#include <fluca/private/segimpl.h>

PetscErrorCode SegSetPhys(Seg seg, Phys phys)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 2);
  PetscCheckSameComm(seg, 1, phys, 2);
  if (seg->phys == phys) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!seg->setupcalled, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the Phys after SegSetUp()");

  PetscCall(PetscObjectReference((PetscObject)phys));
  PetscCall(PhysDestroy(&seg->phys));
  seg->phys = phys;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetPhys(Seg seg, Phys *phys)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(phys, 2);
  *phys = seg->phys;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetTimeStep(Seg seg, PetscReal dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidLogicalCollectiveReal(seg, dt, 2);
  seg->dt = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetTimeStep(Seg seg, PetscReal *dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(dt, 2);
  *dt = seg->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetMaxTime(Seg seg, PetscReal max_time)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidLogicalCollectiveReal(seg, max_time, 2);
  seg->max_time = max_time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetMaxTime(Seg seg, PetscReal *max_time)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(max_time, 2);
  *max_time = seg->max_time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetMaxSteps(Seg seg, PetscInt max_steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidLogicalCollectiveInt(seg, max_steps, 2);
  seg->max_steps = max_steps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetMaxSteps(Seg seg, PetscInt *max_steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(max_steps, 2);
  *max_steps = seg->max_steps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetTime(Seg seg, PetscReal t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidLogicalCollectiveReal(seg, t, 2);
  seg->t = t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetTime(Seg seg, PetscReal *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(t, 2);
  *t = seg->t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetStepNumber(Seg seg, PetscInt step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidLogicalCollectiveInt(seg, step, 2);
  seg->step = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetStepNumber(Seg seg, PetscInt *step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(step, 2);
  *step = seg->step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetConvergedReason(Seg seg, SegConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  seg->reason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetConvergedReason(Seg seg, SegConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(reason, 2);
  *reason = seg->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetErrorIfStepFailed(Seg seg, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidLogicalCollectiveBool(seg, flg, 2);
  seg->errorifstepfailed = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetErrorIfStepFailed(Seg seg, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  *flg = seg->errorifstepfailed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetPreStep(Seg seg, PetscErrorCode (*prestep)(Seg, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  seg->prestep     = prestep;
  seg->prestep_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetOptionsPrefix(Seg seg, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)seg, prefix));
  if (seg->snes) {
    PetscCall(SNESSetOptionsPrefix(seg->snes, prefix));
    PetscCall(SNESAppendOptionsPrefix(seg->snes, "seg_"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegAppendOptionsPrefix(Seg seg, const char prefix[])
{
  const char *full_prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)seg, prefix));
  if (seg->snes) {
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)seg, &full_prefix));
    PetscCall(SNESSetOptionsPrefix(seg->snes, full_prefix));
    PetscCall(SNESAppendOptionsPrefix(seg->snes, "seg_"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetOptionsPrefix(Seg seg, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(prefix, 2);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)seg, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetFromOptions(Seg seg)
{
  char      type[256];
  PetscBool flg, opt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(SegRegisterAll());

  PetscObjectOptionsBegin((PetscObject)seg);

  PetscCall(PetscOptionsFList("-seg_type", "Seg type", "SegSetType", SegList, (char *)(((PetscObject)seg)->type_name ? ((PetscObject)seg)->type_name : SEGCNLINEAR), type, sizeof(type), &flg));
  if (flg) PetscCall(SegSetType(seg, type));
  else if (!((PetscObject)seg)->type_name) PetscCall(SegSetType(seg, SEGCNLINEAR));

  PetscCall(PetscOptionsReal("-seg_dt", "Time step size", "SegSetTimeStep", seg->dt, &seg->dt, NULL));
  PetscCall(PetscOptionsReal("-seg_max_time", "Final time", "SegSetMaxTime", seg->max_time, &seg->max_time, NULL));
  PetscCall(PetscOptionsInt("-seg_max_steps", "Maximum number of steps", "SegSetMaxSteps", seg->max_steps, &seg->max_steps, NULL));
  PetscCall(PetscOptionsBool("-seg_error_if_step_failed", "Error if a step fails", "SegSetErrorIfStepFailed", seg->errorifstepfailed, &seg->errorifstepfailed, NULL));

  PetscCall(SegMonitorSetFromOptions(seg, "-seg_monitor", "Monitor the current step and time", "SegMonitorDefault", SegMonitorDefault, NULL));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-seg_monitor_cancel", "Remove all monitors", "SegMonitorCancel", flg, &flg, &opt));
  if (opt && flg) PetscCall(SegMonitorCancel(seg));

  PetscTryTypeMethod(seg, setfromoptions, PetscOptionsObject);

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
