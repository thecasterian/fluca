#include <fluca/private/nsimpl.h>

PetscErrorCode NSSetPhys(NS ns, Phys phys)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 2);
  PetscCheckSameComm(ns, 1, phys, 2);
  PetscCheck(!ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the Phys after NSSetUp()");
  PetscCall(PetscObjectReference((PetscObject)phys));
  PetscCall(PhysDestroy(&ns->phys));
  ns->phys = phys;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetPhys(NS ns, Phys *phys)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(phys, 2);
  *phys = ns->phys;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTimeStepSize(NS ns, PetscReal dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->dt = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTimeStepSize(NS ns, PetscReal *dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (dt) *dt = ns->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTimeStep(NS ns, PetscInt step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->step = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTimeStep(NS ns, PetscInt *step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (step) *step = ns->step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTime(NS ns, PetscReal t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->t = t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTime(NS ns, PetscReal *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (t) *t = ns->t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetMaxTime(NS ns, PetscReal max_time)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->max_time = max_time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetMaxTime(NS ns, PetscReal *max_time)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (max_time) *max_time = ns->max_time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetMaxSteps(NS ns, PetscInt max_steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->max_steps = max_steps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetMaxSteps(NS ns, PetscInt *max_steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (max_steps) *max_steps = ns->max_steps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetFromOptions(NS ns)
{
  char      type[256];
  PetscBool flg, opt;
  SNES      snes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(NSRegisterAll());

  PetscObjectOptionsBegin((PetscObject)ns);

  PetscCall(PetscOptionsFList("-ns_type", "NS type", "NSSetType", NSList, (char *)(((PetscObject)ns)->type_name ? ((PetscObject)ns)->type_name : NSCNLINEAR), type, sizeof(type), &flg));
  if (flg) PetscCall(NSSetType(ns, type));
  else if (!((PetscObject)ns)->type_name) PetscCall(NSSetType(ns, NSCNLINEAR));

  PetscCall(PetscOptionsReal("-ns_time_step_size", "Time step size", "NSSetTimeStepSize", ns->dt, &ns->dt, NULL));
  PetscCall(PetscOptionsReal("-ns_max_time", "Maximum time", "NSSetMaxTime", ns->max_time, &ns->max_time, NULL));
  PetscCall(PetscOptionsInt("-ns_max_steps", "Maximum number of steps", "NSSetMaxSteps", ns->max_steps, &ns->max_steps, NULL));
  PetscCall(PetscOptionsBool("-ns_error_if_step_failed", "Error if step fails", "NSSetErrorIfStepFailed", ns->errorifstepfailed, &ns->errorifstepfailed, NULL));

  PetscCall(NSMonitorSetFromOptions(ns, "-ns_monitor", "Monitor current step and time", "NSMonitorDefault", NSMonitorDefault, NULL));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ns_monitor_cancel", "Remove all monitors", "NSMonitorCancel", flg, &flg, &opt));
  if (opt && flg) PetscCall(NSMonitorCancel(ns));

  PetscTryTypeMethod(ns, setfromoptions, PetscOptionsObject);

  PetscOptionsEnd();

  PetscCall(NSGetSNES(ns, &snes));
  PetscCall(SNESSetFromOptions(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetConvergedReason(NS ns, NSConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->reason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetConvergedReason(NS ns, NSConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(reason, 2);
  *reason = ns->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetErrorIfStepFailed(NS ns, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->errorifstepfailed = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetErrorIfStepFailed(NS ns, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  *flg = ns->errorifstepfailed;
  PetscFunctionReturn(PETSC_SUCCESS);
}
