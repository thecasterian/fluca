#include <fluca/private/nsimpl.h>
#include <flucaviewer.h>
#include <petscdmcomposite.h>

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
  n->snes        = NULL;
  n->b           = NULL;
  n->x           = NULL;
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

static PetscErrorCode AddField_Private(NS ns, const char *fieldname, DM dm)
{
  NSFieldLink newlink, lastlink;

  PetscFunctionBegin;
  /* Create new field */
  PetscCall(PetscNew(&newlink));
  PetscCall(PetscStrallocpy(fieldname, &newlink->fieldname));
  newlink->dm   = dm;
  newlink->is   = NULL;
  newlink->prev = NULL;
  newlink->next = NULL;

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
  DM          sdm, vdm, Vdm;
  NSFieldLink link;
  IS         *is;
  Vec        *subvecs;
  PetscInt    nf, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (ns->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(NS_SetUp, (PetscObject)ns, 0, 0, 0));

  /* Set default type */
  if (!((PetscObject)ns)->type_name) PetscCall(NSSetType(ns, NSFSM));

  /* Validate */
  PetscCheck(ns->mesh, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Mesh not set");

  /* Create fields and solution vector */
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));
  PetscCall(MeshGetScalarDM(ns->mesh, &sdm));
  PetscCall(MeshGetVectorDM(ns->mesh, &vdm));
  PetscCall(MeshGetStaggeredVectorDM(ns->mesh, &Vdm));

  PetscCall(AddField_Private(ns, "velocity", vdm));
  PetscCall(AddField_Private(ns, "facenormalvelocity", Vdm));
  PetscCall(AddField_Private(ns, "pressure", sdm));

  PetscCall(DMCompositeCreate(comm, &ns->soldm));
  for (link = ns->fieldlink; link; link = link->next) PetscCall(DMCompositeAddDM(ns->soldm, link->dm));
  PetscCall(DMSetUp(ns->soldm));

  PetscCall(DMCompositeGetGlobalISs(ns->soldm, &is));
  for (link = ns->fieldlink, i = 0; link; link = link->next, ++i) link->is = is[i];

  PetscCall(NSGetNumFields(ns, &nf));
  PetscCall(PetscMalloc1(nf, &subvecs));
  for (link = ns->fieldlink, i = 0; link; link = link->next, ++i) PetscCall(DMCreateGlobalVector(link->dm, &subvecs[i]));
  PetscCall(VecCreateNest(comm, nf, is, subvecs, &ns->sol));

  PetscCall(PetscFree(is));
  for (i = 0; i < nf; ++i) PetscCall(VecDestroy(&subvecs[i]));
  PetscCall(PetscFree(subvecs));

  /* Call specific type setup */
  PetscTryTypeMethod(ns, setup);

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

PetscErrorCode NSView(NS ns, PetscViewer v)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ns), &v));
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ns, 1, v, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ns, v));
    PetscCall(PetscViewerASCIIPrintf(v, "Density: %g, Viscosity: %g, Time step size: %g\n", ns->rho, ns->mu, ns->dt));
    PetscCall(PetscViewerASCIIPrintf(v, "Current time step: %d, Current time: %g\n", ns->step, ns->t));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscTryTypeMethod(ns, view, v);
    PetscCall(PetscViewerASCIIPopTab(v));
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

  PetscCall(NSMonitorCancel(*ns));

  PetscTryTypeMethod((*ns), destroy);
  PetscCall(PetscHeaderDestroy(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}
