#include <fluca/private/segimpl.h>

PetscErrorCode SegSetFromOptions(Seg seg)
{
  const char *default_type;
  char        type[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  if (!((PetscObject)seg)->type_name) default_type = SEGSRK;
  else default_type = ((PetscObject)seg)->type_name;
  PetscCall(SegRegisterAll());

  PetscObjectOptionsBegin((PetscObject)seg);
  PetscCall(PetscOptionsFList("-seg_type", "Segregated integrator type", "SegSetType", SegList, default_type, type, sizeof(type), &flg));
  if (flg) PetscCall(SegSetType(seg, type));
  else if (!((PetscObject)seg)->type_name) PetscCall(SegSetType(seg, default_type));
  PetscCall(PetscOptionsReal("-seg_dt", "Time step size", "SegSetTimeStepSize", seg->dt, &seg->dt, NULL));
  PetscCall(PetscOptionsReal("-seg_max_time", "Maximum simulation time", "SegSetMaxTime", seg->max_time, &seg->max_time, NULL));
  PetscCall(PetscOptionsInt("-seg_max_steps", "Maximum number of steps", "SegSetMaxSteps", seg->max_steps, &seg->max_steps, NULL));
  PetscTryTypeMethod(seg, setfromoptions, PetscOptionsObject);
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)seg, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetOptionsPrefix(Seg seg, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)seg, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegAppendOptionsPrefix(Seg seg, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)seg, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetOptionsPrefix(Seg seg, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)seg, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetTimeStepSize(Seg seg, PetscReal dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCheck(dt > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Time step size must be positive, got %g", (double)dt);
  seg->dt = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegGetTimeStepSize(Seg seg, PetscReal *dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(dt, 2);
  *dt = seg->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSetTime(Seg seg, PetscReal t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
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

PetscErrorCode SegSetMaxTime(Seg seg, PetscReal max_time)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
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
  PetscCheck(max_steps >= 0, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Maximum steps must be non-negative, got %" PetscInt_FMT, max_steps);
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

PetscErrorCode SegGetStepNumber(Seg seg, PetscInt *step_number)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(step_number, 2);
  *step_number = seg->step_number;
  PetscFunctionReturn(PETSC_SUCCESS);
}
