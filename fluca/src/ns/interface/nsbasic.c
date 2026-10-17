#include <fluca/private/nsimpl.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

const char *const  NSConvergedReasons_Shifted[] = {"DIVERGED_NONLINEAR_SOLVE", "CONVERGED_ITERATING", "CONVERGED_TIME", "CONVERGED_ITS", "NSConvergedReason", "", NULL};
const char *const *NSConvergedReasons           = NSConvergedReasons_Shifted + 1;

PetscClassId  NS_CLASSID      = 0;
PetscLogEvent NS_SetUp        = 0;
PetscLogEvent NS_Step         = 0;
PetscLogEvent NS_FormJacobian = 0;
PetscLogEvent NS_FormFunction = 0;

PetscFunctionList NSList              = NULL;
PetscBool         NSRegisterAllCalled = PETSC_FALSE;

PetscErrorCode NSCreate(MPI_Comm comm, NS *ns)
{
  NS n;

  PetscFunctionBegin;
  PetscAssertPointer(ns, 2);

  PetscCall(NSInitializePackage());
  /* The header is zero-initialized, so every spatial operator starts NULL */
  PetscCall(FlucaHeaderCreate(n, NS_CLASSID, "NS", "Navier-Stokes solver", "NS", comm, NSDestroy, NSView));
  n->dt                = 0.0;
  n->max_time          = PETSC_MAX_REAL;
  n->max_steps         = PETSC_INT_MAX;
  n->phys              = NULL;
  n->step              = 0;
  n->t                 = 0.0;
  n->data              = NULL;
  n->sol               = NULL;
  n->sol0              = NULL;
  n->snes              = NULL;
  n->J                 = NULL;
  n->r                 = NULL;
  n->x                 = NULL;
  n->nullspace         = NULL;
  n->errorifstepfailed = PETSC_TRUE;
  n->reason            = NS_CONVERGED_ITERATING;
  n->setupcalled       = PETSC_FALSE;
  n->num_mons          = 0;

  *ns = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetType(NS ns, NSType type)
{
  NSType old_type;
  PetscErrorCode (*impl_create)(NS);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

  PetscCall(NSGetType(ns, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(NSList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown ns type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(ns, destroy);
    PetscCall(PetscMemzero(ns->ops, sizeof(struct _NSOps)));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)ns, type));
  PetscCall((*impl_create)(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetType(NS ns, NSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(NSRegisterAll());
  *type = ((PetscObject)ns)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian_Private(SNES snes, Vec x, Mat J, Mat Jpre, void *ctx)
{
  NS ns = (NS)ctx;

  PetscFunctionBegin;
  PetscCall(NSFormJacobian(ns, x, Jpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunction_Private(SNES snes, Vec x, Vec f, void *ctx)
{
  NS ns = (NS)ctx;

  PetscFunctionBegin;
  PetscCall(NSFormFunction(ns, x, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PicardComputeFunction_Private(SNES snes, Vec x, Vec f, void *ctx)
{
  NS ns = (NS)ctx;

  PetscFunctionBegin;
  PetscCall(SNESPicardComputeFunction(snes, x, f, ctx));
  PetscCall(MatNullSpaceRemove(ns->nullspace, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormInitialGuess_Private(SNES snes, Vec x, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckField_Private(NS ns, const char name[], PhysFieldLocation loc, PetscInt ncomp)
{
  PhysFieldLocation floc;
  PetscInt          fncomp;

  PetscFunctionBegin;
  PetscCall(PhysGetField(ns->phys, name, &floc, NULL, &fncomp));
  PetscCheck(floc == loc && fncomp == ncomp, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONG, "NS requires field %s at %s with %" PetscInt_FMT " component(s); the Phys declares it at %s with %" PetscInt_FMT, name, PhysFieldLocations[loc], ncomp, PhysFieldLocations[floc], fncomp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The pressure null space from the boundary condition types. Only velocity boundaries and periodic
   directions are supported; neither prescribes the pressure, so the pressure is determined up to a
   constant and the null space is the constant pressure. */
static PetscErrorCode CreateNullSpace_Private(NS ns)
{
  DMBoundaryType bt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PhysBC         bc;
  IS             is_p;
  Vec            nullvec, sub;
  PetscInt       dim, d, s, np;
  DM             dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetBoundaryTypes(dm, &bt[0], &bt[1], &bt[2]));
  for (d = 0; d < dim; ++d) {
    if (bt[d] == DM_BOUNDARY_PERIODIC) continue;
    for (s = 0; s < 2; ++s) {
      PetscCall(PhysGetBoundaryCondition(ns->phys, 2 * d + s, &bc));
      switch (bc.type) {
      case PHYS_BC_VELOCITY:
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "NS does not support boundary condition type %s on face %" PetscInt_FMT, PhysBCTypes[bc.type], 2 * d + s);
      }
    }
  }
  PetscCall(PhysGetFieldIS(ns->phys, PHYS_FIELD_PRESSURE, &is_p));
  PetscCall(DMCreateGlobalVector(dm, &nullvec));
  PetscCall(VecZeroEntries(nullvec));
  PetscCall(VecGetSubVector(nullvec, is_p, &sub));
  PetscCall(VecGetSize(sub, &np));
  PetscCall(VecSet(sub, 1. / PetscSqrtReal((PetscReal)np)));
  PetscCall(VecRestoreSubVector(nullvec, is_p, &sub));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)ns), PETSC_FALSE, 1, &nullvec, &ns->nullspace));
  PetscCall(VecDestroy(&nullvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetUp(NS ns)
{
  PetscInt  dim, sw;
  PetscBool isabf;
  IS        is_vel, is_U, is_p;
  SNES      snes;
  KSP       ksp;
  PC        pc;
  DM        dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (ns->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(NS_SetUp, (PetscObject)ns, 0, 0, 0));

  if (!((PetscObject)ns)->type_name) PetscCall(NSSetType(ns, NSCNLINEAR));
  PetscCheck(ns->phys, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Phys not set. Call NSSetPhys() first");
  PetscCall(PhysSetUp(ns->phys));
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(CheckField_Private(ns, PHYS_FIELD_VELOCITY, PHYS_FIELD_ELEMENT, dim));
  PetscCall(CheckField_Private(ns, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_FACE, 1));
  PetscCall(CheckField_Private(ns, PHYS_FIELD_PRESSURE, PHYS_FIELD_ELEMENT, 1));
  /* The one-sided wall gradient and T G_c reach two cells */
  PetscCall(DMStagGetStencilWidth(dm, &sw));
  PetscCheck(sw >= 2, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_OUTOFRANGE, "NS requires a base DM stencil width of at least 2, got %" PetscInt_FMT, sw);

  PetscCall(CreateNullSpace_Private(ns));
  PetscCall(NSSetUpSpatialOperators_Internal(ns));

  PetscCall(PhysCreateSolutionVector(ns->phys, &ns->sol));
  PetscCall(VecZeroEntries(ns->sol));
  PetscCall(PetscObjectSetName((PetscObject)ns->sol, "Solution"));
  PetscCall(VecDuplicate(ns->sol, &ns->sol0));
  PetscCall(VecDuplicate(ns->sol, &ns->x));
  PetscCall(VecDuplicate(ns->sol, &ns->r));
  PetscCall(DMCreateMatrix(dm, &ns->J));
  PetscCall(MatSetOption(ns->J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetNullSpace(ns->J, ns->nullspace));

  PetscTryTypeMethod(ns, setup);

  PetscCall(NSGetSNES(ns, &snes));
  PetscCall(SNESSetPicard(snes, ns->r, FormFunction_Private, ns->J, ns->J, FormJacobian_Private, ns));
  PetscCall(SNESSetFunction(snes, ns->r, PicardComputeFunction_Private, ns));
  /* Need zero initial guess to ensure least-square solution of pressure */
  PetscCall(SNESSetComputeInitialGuess(snes, FormInitialGuess_Private, NULL));

  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCABF, &isabf));
  if (isabf) {
    PetscCall(NSGetField(ns, PHYS_FIELD_VELOCITY, &is_vel));
    PetscCall(NSGetField(ns, PHYS_FIELD_FACE_VELOCITY, &is_U));
    PetscCall(NSGetField(ns, PHYS_FIELD_PRESSURE, &is_p));
    PetscCall(PCABFSetFieldIS(pc, is_vel, is_U, is_p));
  }

  PetscCall(PetscLogEventEnd(NS_SetUp, (PetscObject)ns, 0, 0, 0));

  /* NSViewFromOptions() is called in NSSolve(). */

  ns->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSStep(NS ns)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCheck(ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Must call NSSetUp() before NSStep()");

  PetscCall(VecCopy(ns->sol, ns->sol0));

  PetscCall(PetscLogEventBegin(NS_Step, (PetscObject)ns, 0, 0, 0));
  PetscUseTypeMethod(ns, step);
  PetscCall(PetscLogEventEnd(NS_Step, (PetscObject)ns, 0, 0, 0));

  if (ns->reason >= 0) {
    ++ns->step;
    ns->t += ns->dt;
  }

  if (ns->reason < 0 && ns->errorifstepfailed) {
    PetscCall(NSMonitorCancel(ns));
    PetscCall(SNESMonitorCancel(ns->snes));
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_NOT_CONVERGED, "NSStep has failed due to %s", NSConvergedReasons[ns->reason]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFormJacobian(NS ns, Vec x, Mat J)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (x) PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(J, MAT_CLASSID, 3);
  PetscCall(PetscLogEventBegin(NS_FormJacobian, ns, x, J, NULL));
  PetscUseTypeMethod(ns, formjacobian, x, J);
  PetscCall(PetscLogEventEnd(NS_FormJacobian, ns, x, J, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFormFunction(NS ns, Vec x, Vec f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (x) PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(f, VEC_CLASSID, 3);
  PetscCall(PetscLogEventBegin(NS_FormFunction, ns, x, f, NULL));
  PetscUseTypeMethod(ns, formfunction, x, f);
  PetscCall(PetscLogEventEnd(NS_FormFunction, ns, x, f, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSolve(NS ns)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCheck(ns->max_time < PETSC_MAX_REAL || ns->max_steps != PETSC_INT_MAX, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "At least one of max time or max steps must be specified");

  PetscCall(NSViewFromOptions(ns, NULL, "-ns_view_pre"));

  if (ns->step >= ns->max_steps) ns->reason = NS_CONVERGED_ITS;
  else if (ns->t >= ns->max_time) ns->reason = NS_CONVERGED_TIME;

  while (ns->reason == NS_CONVERGED_ITERATING) {
    PetscCall(NSMonitor(ns));
    PetscCall(NSStep(ns));

    if (ns->reason == NS_CONVERGED_ITERATING) {
      if (ns->step >= ns->max_steps) ns->reason = NS_CONVERGED_ITS;
      else if (ns->t >= ns->max_time) ns->reason = NS_CONVERGED_TIME;
    }
  }
  PetscCall(NSMonitor(ns));

  PetscCall(NSViewFromOptions(ns, NULL, "-ns_view"));
  PetscCall(NSViewSolutionFromOptions(ns, NULL, "-ns_view_solution"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView(NS ns, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ns), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ns, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ns, viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Time step size: %g\n", (double)ns->dt));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Current time step: %" PetscInt_FMT ", Current time: %g\n", ns->step, (double)ns->t));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (ns->phys) PetscCall(PhysView(ns->phys, viewer));
    PetscTryTypeMethod(ns, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewFromOptions(NS ns, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)ns, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSDestroy(NS *ns)
{
  PetscFunctionBegin;
  if (!*ns) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*ns), NS_CLASSID, 1);

  if (--((PetscObject)(*ns))->refct > 0) {
    *ns = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(NSDestroySpatialOperators_Internal(*ns));
  PetscCall(VecDestroy(&(*ns)->sol));
  PetscCall(VecDestroy(&(*ns)->sol0));

  PetscCall(SNESDestroy(&(*ns)->snes));
  PetscCall(MatDestroy(&(*ns)->J));
  PetscCall(VecDestroy(&(*ns)->r));
  PetscCall(VecDestroy(&(*ns)->x));
  PetscCall(MatNullSpaceDestroy(&(*ns)->nullspace));

  PetscCall(NSMonitorCancel(*ns));

  PetscTryTypeMethod((*ns), destroy);
  PetscCall(PhysDestroy(&(*ns)->phys));
  PetscCall(PetscHeaderDestroy(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCheckDiverged(NS ns)
{
  SNESConvergedReason snesreason;

  PetscFunctionBegin;
  PetscCall(SNESGetConvergedReason(ns->snes, &snesreason));
  if (snesreason < 0) {
    PetscCall(PetscInfo(ns, "Step=%" PetscInt_FMT ", nonlinear solve failure: %s\n", ns->step, SNESConvergedReasons[snesreason]));
    ns->reason = NS_DIVERGED_NONLINEAR_SOLVE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
