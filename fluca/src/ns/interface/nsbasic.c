#include <fluca/private/nsimpl.h>
#include <flucaviewer.h>
#include <petscdmcomposite.h>
#include <petscdmstag.h>

const char *const  NSSolvers[]                  = {"FSM", "NSSolver", "", NULL};
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
  PetscCall(FlucaHeaderCreate(n, NS_CLASSID, "NS", "Navier-Stokes solver", "NS", comm, NSDestroy, NSView));
  n->rho               = 0.0;
  n->mu                = 0.0;
  n->dt                = 0.0;
  n->max_time          = PETSC_MAX_REAL;
  n->max_steps         = PETSC_INT_MAX;
  n->step              = 0;
  n->t                 = 0.0;
  n->mesh              = NULL;
  n->bcs               = NULL;
  n->data              = NULL;
  n->fieldlink         = NULL;
  n->soldm             = NULL;
  n->sol               = NULL;
  n->sol0              = NULL;
  n->solver            = NS_FSM;
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

static PetscErrorCode AddField_Private(NS ns, const char *fieldname, MeshDMType dmtype)
{
  NSFieldLink newlink, lastlink;

  PetscFunctionBegin;
  /* Create new field */
  PetscCall(PetscNew(&newlink));
  PetscCall(PetscStrallocpy(fieldname, &newlink->fieldname));
  newlink->dmtype = dmtype;
  newlink->is     = NULL;
  newlink->prev   = NULL;
  newlink->next   = NULL;

  /* Append to end of list */
  if (!ns->fieldlink) {
    ns->fieldlink = newlink;
  } else {
    lastlink = ns->fieldlink;
    while (lastlink->next) lastlink = lastlink->next;
    lastlink->next = newlink;
    newlink->prev  = lastlink;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian_Private(SNES snes, Vec x, Mat J, Mat Jpre, void *ctx)
{
  NS ns = (NS)ctx;

  PetscFunctionBegin;
  PetscCall(NSFormJacobian(ns, x, Jpre, NS_UPDATE_JACOBIAN));
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

  /* Remove null space */
  PetscAssert(ns->nullspace, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Null space must be set");
  PetscCall(MatNullSpaceRemove(ns->nullspace, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormInitialGuess_Private(SNES snes, Vec x, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetUp(NS ns)
{
  MPI_Comm    comm;
  DM          dm;
  NSFieldLink link;
  IS         *is;
  Vec        *subvecs;
  Mat        *submats;
  SNES        snes;
  PetscInt    nf, nb, i;
  PetscBool   neednullspace;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (ns->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(NS_SetUp, (PetscObject)ns, 0, 0, 0));

  /* Set default type */
  if (!((PetscObject)ns)->type_name) PetscCall(NSSetType(ns, NSCNLINEAR));

  /* Validate */
  PetscCheck(ns->mesh, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Mesh not set");

  /* Create fields and solution vector */
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));

  PetscCall(AddField_Private(ns, NS_FIELD_VELOCITY, MESH_DM_VECTOR));
  PetscCall(AddField_Private(ns, NS_FIELD_FACE_NORMAL_VELOCITY, MESH_DM_STAG_SCALAR));
  PetscCall(AddField_Private(ns, NS_FIELD_PRESSURE, MESH_DM_SCALAR));

  PetscCall(DMCompositeCreate(comm, &ns->soldm));
  for (link = ns->fieldlink; link; link = link->next) {
    PetscCall(MeshGetDM(ns->mesh, link->dmtype, &dm));
    PetscCall(DMCompositeAddDM(ns->soldm, dm));
  }
  PetscCall(DMSetUp(ns->soldm));

  PetscCall(DMCompositeGetGlobalISs(ns->soldm, &is));
  for (link = ns->fieldlink, i = 0; link; link = link->next, ++i) link->is = is[i];

  PetscCall(NSGetNumFields(ns, &nf));
  PetscCall(PetscMalloc1(nf, &subvecs));
  for (link = ns->fieldlink, i = 0; link; link = link->next, ++i) {
    PetscCall(MeshCreateGlobalVector(ns->mesh, link->dmtype, &subvecs[i]));
    PetscCall(PetscObjectSetName((PetscObject)subvecs[i], link->fieldname));
  }
  PetscCall(VecCreateNest(comm, nf, is, subvecs, &ns->sol));

  /* Create solver */
  PetscCall(PetscMalloc1(nf * nf, &submats));
  for (i = 0; i < nf * nf; ++i) submats[i] = NULL;
  PetscCall(MatCreateNest(comm, nf, is, nf, is, submats, &ns->J));
  PetscCall(MatSetUp(ns->J));
  PetscCall(MatNestSetVecType(ns->J, VECNEST));
  /* Initialize Jacobian */
  PetscCall(NSFormJacobian(ns, ns->x, ns->J, NS_INIT_JACOBIAN));
  PetscCall(MatCreateVecs(ns->J, &ns->x, &ns->r));

  PetscCall(PetscFree(is));
  for (i = 0; i < nf; ++i) PetscCall(VecDestroy(&subvecs[i]));
  PetscCall(PetscFree(subvecs));
  PetscCall(PetscFree(submats));

  /* Create null space for pressure */
  neednullspace = PETSC_TRUE;
  PetscCall(MeshGetNumberBoundaries(ns->mesh, &nb));
  for (i = 0; i < nb; ++i) switch (ns->bcs[i].type) {
    case NS_BC_VELOCITY:
    case NS_BC_PERIODIC:
      /* Need null space */
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type");
    }
  if (neednullspace) {
    IS       is;
    Vec      vecs[1], subvec;
    PetscInt subvecsize;

    PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, NULL, &is));
    PetscCall(MatCreateVecs(ns->J, NULL, &vecs[0]));
    PetscCall(VecGetSubVector(vecs[0], is, &subvec));
    PetscCall(VecGetSize(subvec, &subvecsize));
    PetscCall(VecSet(subvec, 1. / PetscSqrtReal((PetscReal)subvecsize)));
    PetscCall(VecRestoreSubVector(vecs[0], is, &subvec));
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, 1, vecs, &ns->nullspace));
    PetscCall(VecDestroy(&vecs[0]));
    PetscCall(MatSetNullSpace(ns->J, ns->nullspace));
  }

  /* Set solver callbacks */
  PetscCall(NSGetSNES(ns, &snes));
  PetscCall(SNESSetPicard(snes, ns->r, FormFunction_Private, ns->J, ns->J, FormJacobian_Private, ns));
  if (neednullspace) PetscCall(SNESSetFunction(snes, ns->r, PicardComputeFunction_Private, ns));
  /* Need zero initial guess to ensure least-square solution of pressure */
  PetscCall(SNESSetComputeInitialGuess(snes, FormInitialGuess_Private, NULL));

  /* Call specific type setup */
  PetscTryTypeMethod(ns, setup);

  PetscCall(PetscLogEventEnd(NS_SetUp, (PetscObject)ns, 0, 0, 0));

  /* NSViewFromOptions() is called in NSSolve(). */

  ns->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSStep(NS ns)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

  if (!ns->sol0) PetscCall(VecDuplicate(ns->sol, &ns->sol0));
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

PetscErrorCode NSFormJacobian(NS ns, Vec x, Mat J, NSFormJacobianType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(J, MAT_CLASSID, 3);
  PetscLogEventBegin(NS_FormJacobian, ns, x, J, NULL);
  PetscUseTypeMethod(ns, formjacobian, x, J, type);
  PetscLogEventEnd(NS_FormJacobian, ns, x, J, NULL);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFormFunction(NS ns, Vec x, Vec f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(f, VEC_CLASSID, 3);
  PetscLogEventBegin(NS_FormFunction, ns, x, f, NULL);
  PetscUseTypeMethod(ns, formfunction, x, f);
  PetscLogEventEnd(NS_FormFunction, ns, x, f, NULL);
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

    /* Check convergence */
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "Density: %g, Viscosity: %g, Time step size: %g\n", ns->rho, ns->mu, ns->dt));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Current time step: %d, Current time: %g\n", ns->step, ns->t));
    PetscCall(PetscViewerASCIIPushTab(viewer));
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
  NSFieldLink link, nextlink;

  PetscFunctionBegin;
  if (!*ns) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*ns), NS_CLASSID, 1);

  if (--((PetscObject)(*ns))->refct > 0) {
    *ns = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(MeshDestroy(&(*ns)->mesh));
  PetscCall(PetscFree((*ns)->bcs));

  link = (*ns)->fieldlink;
  while (link) {
    PetscCall(PetscFree(link->fieldname));
    PetscCall(ISDestroy(&link->is));
    nextlink = link->next;
    PetscCall(PetscFree(link));
    link = nextlink;
  }
  PetscCall(DMDestroy(&(*ns)->soldm));
  PetscCall(VecDestroy(&(*ns)->sol));
  PetscCall(VecDestroy(&(*ns)->sol0));

  PetscCall(SNESDestroy(&(*ns)->snes));
  PetscCall(MatDestroy(&(*ns)->J));
  PetscCall(VecDestroy(&(*ns)->r));
  PetscCall(VecDestroy(&(*ns)->x));
  PetscCall(MatNullSpaceDestroy(&(*ns)->nullspace));

  PetscCall(NSMonitorCancel(*ns));

  PetscTryTypeMethod((*ns), destroy);
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
