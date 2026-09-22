#include <fluca/private/segimpl.h>
#include <flucaviewer.h>

const char *const  SegConvergedReasons_Shifted[] = {"DIVERGED_LINEAR_SOLVE", "CONVERGED_ITERATING", "CONVERGED_TIME", "CONVERGED_ITS", "SegConvergedReason", "", NULL};
const char *const *SegConvergedReasons           = SegConvergedReasons_Shifted + 1;

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
  PetscCall(FlucaHeaderCreate(s, SEG_CLASSID, "Seg", "Segregated solver", "Seg", comm, SegDestroy, SegView));

  s->phys              = NULL;
  s->dt                = 0.;
  s->max_time          = PETSC_MAX_REAL;
  s->max_steps         = PETSC_INT_MAX;
  s->sol               = NULL;
  s->data              = NULL;
  s->ksp               = NULL;
  s->errorifstepfailed = PETSC_TRUE;
  s->reason            = SEG_CONVERGED_ITERATING;
  s->t                 = 0.;
  s->step              = 0;
  s->setupcalled       = PETSC_FALSE;
  s->prestep           = NULL;
  s->prestep_ctx       = NULL;
  s->num_mons          = 0;

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

PetscErrorCode SegGetKSP(Seg seg, KSP *ksp)
{
  const char *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscAssertPointer(ksp, 2);
  if (!seg->ksp) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)seg), &seg->ksp));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)seg->ksp, (PetscObject)seg, 1));
    PetscCall(PetscObjectSetOptions((PetscObject)seg->ksp, ((PetscObject)seg)->options));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)seg, &prefix));
    PetscCall(KSPSetOptionsPrefix(seg->ksp, prefix));
    PetscCall(KSPAppendOptionsPrefix(seg->ksp, "seg_"));
  }
  *ksp = seg->ksp;
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

PetscErrorCode SegSetUp(Seg seg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  if (seg->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(SEG_SetUp, (PetscObject)seg, 0, 0, 0));

  if (!((PetscObject)seg)->type_name) PetscCall(SegSetType(seg, SEGCNLINEAR));
  PetscCheck(seg->phys, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "No Phys attached to Seg; call SegSetPhys() first");

  PetscTryTypeMethod(seg, setup);

  PetscCall(PetscLogEventEnd(SEG_SetUp, (PetscObject)seg, 0, 0, 0));

  seg->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The time step to use after the current one, shortened or slightly lengthened so that the last
   step lands exactly on the maximum time. This reproduces what TSAdaptChoose() does for
   TS_EXACTFINALTIME_MATCHSTEP with its default match-step factors. */
static PetscErrorCode SegNextTimeStep_Private(Seg seg, PetscReal *next_dt)
{
  PetscReal a = 1. + 0.01; /* allow a 1% step size increase in the last step */
  PetscReal b = 2.;        /* halve the last step if it is greater than what remains divided by this */
  PetscReal t, tend, hmax;

  PetscFunctionBegin;
  t    = seg->t + seg->dt;
  tend = t + seg->dt;
  hmax = seg->max_time - t;

  *next_dt = seg->dt;
  if (t < seg->max_time && tend > seg->max_time) *next_dt = hmax;
  if (t < seg->max_time && tend < seg->max_time && seg->dt * b > hmax) *next_dt = hmax / 2.;
  if (t < seg->max_time && tend < seg->max_time && seg->dt * a > hmax) *next_dt = hmax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegStep(Seg seg)
{
  PetscReal next_dt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(SegSetUp(seg));
  PetscCheck(seg->sol, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "No solution vector; call SegSolve() instead of SegStep()");
  PetscCheck(seg->dt > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Time step must be positive, got %g", (double)seg->dt);

  seg->reason = SEG_CONVERGED_ITERATING;

  PetscCall(PetscLogEventBegin(SEG_Step, (PetscObject)seg, 0, 0, 0));
  PetscUseTypeMethod(seg, step);
  PetscCall(PetscLogEventEnd(SEG_Step, (PetscObject)seg, 0, 0, 0));

  if (seg->reason >= 0) {
    PetscCall(SegNextTimeStep_Private(seg, &next_dt));
    seg->t += seg->dt;
    seg->dt = next_dt;
    ++seg->step;
  }

  if (seg->reason < 0 && seg->errorifstepfailed) {
    PetscCall(SegMonitorCancel(seg));
    SETERRQ(PetscObjectComm((PetscObject)seg), PETSC_ERR_NOT_CONVERGED, "SegStep has failed due to %s", SegConvergedReasons[seg->reason]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSolve(Seg seg, Vec sol)
{
  PetscReal maxdt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscValidHeaderSpecific(sol, VEC_CLASSID, 2);
  PetscCheck(seg->max_time < PETSC_MAX_REAL || seg->max_steps != PETSC_INT_MAX, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "At least one of the maximum time or the maximum number of steps must be specified");

  PetscCall(PetscObjectReference((PetscObject)sol));
  PetscCall(VecDestroy(&seg->sol));
  seg->sol = sol;

  PetscCall(SegSetUp(seg));
  PetscCall(SegViewFromOptions(seg, NULL, "-seg_view_pre"));

  /* Keep the first step from overshooting the final time */
  maxdt   = seg->max_time - seg->t;
  seg->dt = seg->dt >= maxdt ? maxdt : (PetscIsCloseAtTol(seg->dt, maxdt, 10 * PETSC_MACHINE_EPSILON, 0) ? maxdt : seg->dt);

  seg->reason = SEG_CONVERGED_ITERATING;
  if (seg->step >= seg->max_steps) seg->reason = SEG_CONVERGED_ITS;
  else if (seg->t >= seg->max_time) seg->reason = SEG_CONVERGED_TIME;

  /* Project the initial face velocity once per solve. Seg owns the time loop, so nothing can
     modify the state between here and the first step; a caller that refills the solution vector
     and solves again gets a fresh projection because it goes through SegSolve() again. */
  if (seg->reason == SEG_CONVERGED_ITERATING) PetscTryTypeMethod(seg, presolve);

  while (seg->reason == SEG_CONVERGED_ITERATING) {
    PetscCall(SegMonitor(seg));
    if (seg->prestep) PetscCall((*seg->prestep)(seg, seg->prestep_ctx));
    PetscCall(SegStep(seg));

    if (seg->reason == SEG_CONVERGED_ITERATING) {
      if (seg->step >= seg->max_steps) seg->reason = SEG_CONVERGED_ITS;
      else if (seg->t >= seg->max_time) seg->reason = SEG_CONVERGED_TIME;
    }
  }
  PetscCall(SegMonitor(seg));

  PetscCall(SegViewFromOptions(seg, NULL, "-seg_view"));
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "Time step size: %g, Maximum time: %g, Maximum steps: %" PetscInt_FMT "\n", (double)seg->dt, (double)seg->max_time, seg->max_steps));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Current step: %" PetscInt_FMT ", Current time: %g\n", seg->step, (double)seg->t));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(seg, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegViewFromOptions(Seg seg, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)seg, obj, name));
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

  PetscCall(SegMonitorCancel(*seg));
  PetscCall(KSPDestroy(&(*seg)->ksp));

  PetscTryTypeMethod((*seg), destroy);

  PetscCall(VecDestroy(&(*seg)->sol));
  PetscCall(PhysDestroy(&(*seg)->phys));

  PetscCall(PetscHeaderDestroy(seg));
  PetscFunctionReturn(PETSC_SUCCESS);
}
