#include <fluca/private/segimpl.h>

PetscErrorCode SegStep(Seg seg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCheck(seg->setupcalled, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Must call SegSetUp() before SegStep()");
  PetscCheck(seg->dt > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Time step size not set. Call SegSetTimeStepSize() first");

  PetscCall(PetscLogEventBegin(SEG_Step, (PetscObject)seg, 0, 0, 0));
  PetscUseTypeMethod(seg, step);
  PetscCall(PetscLogEventEnd(SEG_Step, (PetscObject)seg, 0, 0, 0));

  seg->step_number++;
  seg->t += seg->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSolve(Seg seg)
{
  PetscReal dt_orig;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCheck(seg->setupcalled, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Must call SegSetUp() before SegSolve()");

  dt_orig = seg->dt;

  while (seg->t < seg->max_time && seg->step_number < seg->max_steps) {
    /* Clamp dt to not overshoot max_time */
    if (seg->t + seg->dt > seg->max_time) seg->dt = seg->max_time - seg->t;

    PetscCall(SegStep(seg));
    PetscCall(PetscInfo(seg, "Step %" PetscInt_FMT ", t = %g\n", seg->step_number, (double)seg->t));
  }

  /* Restore original dt (may have been clamped on the last step) */
  seg->dt = dt_orig;
  PetscFunctionReturn(PETSC_SUCCESS);
}
