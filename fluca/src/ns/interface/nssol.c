#include <fluca/private/nsimpl.h>
#include <flucaviewer.h>

PetscErrorCode NSGetSNES(NS ns, SNES *snes)
{
  const char *prefix;
  KSP         ksp;
  PC          pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(snes, 2);
  if (!ns->snes) {
    PetscCall(SNESCreate(PetscObjectComm((PetscObject)ns), &ns->snes));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)ns->snes, (PetscObject)ns, 1));
    PetscCall(PetscObjectSetOptions((PetscObject)ns->snes, ((PetscObject)ns)->options));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ns, &prefix));
    PetscCall(SNESSetOptionsPrefix(ns->snes, prefix));
    PetscCall(SNESAppendOptionsPrefix(ns->snes, "ns_"));

    /* Default SNES and KSP options */
    PetscCall(SNESSetTolerances(ns->snes, PETSC_DECIDE, 1.e-5, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(SNESGetKSP(ns->snes, &ksp));
    PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));

    /* Construct approximate block factorization preconditioner (ABF) */
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCABF));
  }
  *snes = ns->snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetSolution(NS ns, Vec *sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(sol, 2);
  *sol = ns->sol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The entries of a field in the solution vector; the IS is owned by the Phys, do not destroy it */
PetscErrorCode NSGetField(NS ns, const char name[], IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(is, 3);
  PetscCheck(ns->phys, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Phys not set. Call NSSetPhys() first");
  PetscCall(PhysGetFieldIS(ns->phys, name, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetSolutionSubVector(NS ns, const char name[], Vec *subvec)
{
  IS is;

  PetscFunctionBegin;
  PetscCall(NSGetField(ns, name, &is));
  PetscCall(VecGetSubVector(ns->sol, is, subvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSRestoreSolutionSubVector(NS ns, const char name[], Vec *subvec)
{
  IS is;

  PetscFunctionBegin;
  PetscCall(NSGetField(ns, name, &is));
  PetscCall(VecRestoreSubVector(ns->sol, is, subvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution(NS ns, PetscViewer viewer)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ns), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ns, 1, viewer, 2);
  PetscCheck(ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Must call NSSetUp() before NSViewSolution()");
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMSetOutputSequenceNumber(dm, ns->step, ns->t));
  PetscCall(VecView(ns->sol, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolutionFromOptions(NS ns, PetscObject obj, const char name[])
{
  PetscViewer       viewer;
  PetscBool         flg;
  PetscViewerFormat format;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (obj) PetscValidHeader(obj, 2);
  prefix = obj ? obj->prefix : ((PetscObject)ns)->prefix;
  PetscCall(FlucaOptionsCreateViewer(PetscObjectComm((PetscObject)ns), ((PetscObject)ns)->options, prefix, name, &viewer, &format, &flg));
  if (flg) {
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(NSViewSolution(ns, viewer));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSLoadSolution(NS ns, PetscViewer viewer)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ns, 1, viewer, 2);
  PetscCheck(ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Must call NSSetUp() before NSLoadSolution()");
  PetscCall(PetscViewerCheckReadable(viewer));
  PetscCall(VecLoad(ns->sol, viewer));
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetOutputSequenceNumber(dm, &ns->step, &ns->t));
  PetscFunctionReturn(PETSC_SUCCESS);
}
