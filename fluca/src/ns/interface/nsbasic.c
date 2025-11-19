#include <fluca/private/nsimpl.h>
#include <flucaviewer.h>
#include <petscdmcomposite.h>
#include <petscdmstag.h>

PetscClassId  NS_CLASSID              = 0;
PetscLogEvent NS_SetUp                = 0;
PetscLogEvent NS_LoadSolutionFromFile = 0;
PetscLogEvent NS_Solve                = 0;

PetscFunctionList NSList              = NULL;
PetscBool         NSRegisterAllCalled = PETSC_FALSE;

PetscErrorCode NSCreate(MPI_Comm comm, NS *ns)
{
  NS n;

  PetscFunctionBegin;
  PetscAssertPointer(ns, 2);

  PetscCall(NSInitializePackage());
  PetscCall(FlucaHeaderCreate(n, NS_CLASSID, "NS", "Navier-Stokes solver", "NS", comm, NSDestroy, NSView));
  n->rho         = 0.0;
  n->mu          = 0.0;
  n->dt          = 0.0;
  n->step        = 0;
  n->t           = 0.0;
  n->mesh        = NULL;
  n->bcs         = NULL;
  n->data        = NULL;
  n->fieldlink   = NULL;
  n->soldm       = NULL;
  n->sol         = NULL;
  n->sol0        = NULL;
  n->solver      = NS_FSM;
  n->snes        = NULL;
  n->J           = NULL;
  n->r           = NULL;
  n->x           = NULL;
  n->nullspace   = NULL;
  n->setupcalled = PETSC_FALSE;
  n->num_mons    = 0;
  n->mon_freq    = 1;

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

PetscErrorCode NSSetUp(NS ns)
{
  MPI_Comm    comm;
  DM          dm;
  NSFieldLink link;
  IS         *is;
  Vec        *subvecs;
  Mat        *submats;
  PetscInt    nf, i;

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
  PetscCall(SNESCreate(comm, &ns->snes));
  PetscCall(SNESSetDM(ns->snes, ns->soldm));
  PetscCall(SNESSetTolerances(ns->snes, PETSC_DEFAULT, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(SNESSetOptionsPrefix(ns->snes, "ns_"));
  PetscCall(SNESSetFromOptions(ns->snes));
  PetscCall(PetscMalloc1(nf * nf, &submats));
  for (i = 0; i < nf * nf; ++i) submats[i] = NULL;
  PetscCall(MatCreateNest(comm, nf, is, nf, is, submats, &ns->J));
  PetscCall(MatSetUp(ns->J));
  PetscCall(MatNestSetVecType(ns->J, VECNEST));
  PetscCall(VecDuplicate(ns->sol, &ns->x));
  PetscCall(VecDuplicate(ns->sol, &ns->r));

  PetscCall(PetscFree(is));
  for (i = 0; i < nf; ++i) PetscCall(VecDestroy(&subvecs[i]));
  PetscCall(PetscFree(subvecs));
  PetscCall(PetscFree(submats));

  /* Call specific type setup */
  PetscTryTypeMethod(ns, setup);

  /* Jacobian is built; create preconditioner */
  PetscCall(NSSetPreconditioner_FSM(ns));

  PetscCall(PetscLogEventEnd(NS_SetUp, (PetscObject)ns, 0, 0, 0));

  /* NSViewFromOptions() is called in NSSolve(). */

  ns->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSolve(NS ns, PetscInt num_iters)
{
  PetscReal t_init;
  PetscInt  i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscLogEventBegin(NS_Solve, (PetscObject)ns, 0, 0, 0));

  PetscCall(NSViewFromOptions(ns, NULL, "-ns_view_pre"));

  t_init = ns->t;
  for (i = 0; i < num_iters; ++i) {
    if (!ns->sol0) PetscCall(VecDuplicate(ns->sol, &ns->sol0));
    PetscCall(VecCopy(ns->sol, ns->sol0));

    PetscTryTypeMethod(ns, iterate);
    ++ns->step;
    ns->t = t_init + (i + 1) * ns->dt;

    if (ns->step % ns->mon_freq == 0) PetscCall(NSMonitor(ns));
  }

  PetscCall(NSViewFromOptions(ns, NULL, "-ns_view"));
  PetscCall(NSViewSolutionFromOptions(ns, NULL, "-ns_view_solution"));

  PetscCall(PetscLogEventEnd(NS_Solve, (PetscObject)ns, 0, 0, 0));
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
