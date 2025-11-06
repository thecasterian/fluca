#include <fluca/private/nsimpl.h>
#include <fluca/private/flucaviewercgnsimpl.h>

PetscErrorCode NSGetSolution(NS ns, Vec *sol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(sol, 2);
  *sol = ns->sol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetNumFields(NS ns, PetscInt *nfields)
{
  NSFieldLink link;
  PetscInt    count = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(nfields, 2);

  link = ns->fieldlink;
  while (link) {
    count++;
    link = link->next;
  }
  *nfields = count;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetField(NS ns, const char name[], MeshDMType *dmtype, IS *is)
{
  NSFieldLink link;
  PetscBool   found = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(name, 2);

  link = ns->fieldlink;
  while (link) {
    PetscCall(PetscStrcmp(link->fieldname, name, &found));
    if (found) break;
    link = link->next;
  }
  PetscCheck(found, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_OUTOFRANGE, "Field \"%s\" not found", name);

  if (dmtype) *dmtype = link->dmtype;
  if (is) *is = link->is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetFieldByIndex(NS ns, PetscInt index, const char *name[], MeshDMType *dmtype, IS *is)
{
  NSFieldLink link;
  PetscInt    count = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCheck(index >= 0, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_OUTOFRANGE, "Field index %" PetscInt_FMT " cannot be negative", index);

  link = ns->fieldlink;
  while (link && count < index) {
    count++;
    link = link->next;
  }
  PetscCheck(link, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_OUTOFRANGE, "Field index %" PetscInt_FMT " exceeds number of fields", index);

  if (name) *name = link->fieldname;
  if (dmtype) *dmtype = link->dmtype;
  if (is) *is = link->is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetSolutionSubVector(NS ns, const char name[], Vec *subvec)
{
  IS is;

  PetscFunctionBegin;
  PetscCall(NSGetField(ns, name, NULL, &is));
  PetscCall(VecGetSubVector(ns->sol, is, subvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSRestoreSolutionSubVector(NS ns, const char name[], Vec *subvec)
{
  IS is;

  PetscFunctionBegin;
  PetscCall(NSGetField(ns, name, NULL, &is));
  PetscCall(VecRestoreSubVector(ns->sol, is, subvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution(NS ns, PetscViewer viewer)
{
  NSFieldLink link;
  Vec         subvec;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ns), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ns, 1, viewer, 2);

  /* View fields */
  for (link = ns->fieldlink; link; link = link->next) {
    PetscCall(VecGetSubVector(ns->sol, link->is, &subvec));
    PetscCall(VecView(subvec, viewer));
    PetscCall(VecRestoreSubVector(ns->sol, link->is, &subvec));
  }

  PetscTryTypeMethod(ns, viewsolution, viewer);
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
  NSFieldLink link;
  Vec         subvec;
  PetscInt    step;
  PetscReal   time;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ns, 1, viewer, 2);
  PetscCheck(ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "This function must be called after NSSetUp()");
  PetscCall(PetscViewerCheckReadable(viewer));

  /* Output sequence number is reset here and will be set in VecLoad() */
  PetscCall(MeshSetOutputSequenceNumber(ns->mesh, -1, 0.));

  /* Load fields */
  for (link = ns->fieldlink; link; link = link->next) {
    PetscCall(VecGetSubVector(ns->sol, link->is, &subvec));
    PetscCall(FlucaVecLoad(subvec, viewer));
    PetscCall(VecRestoreSubVector(ns->sol, link->is, &subvec));
  }

  PetscUseTypeMethod(ns, loadsolution, viewer);

  PetscCall(MeshGetOutputSequenceNumber(ns->mesh, &step, &time));
  ns->step = step;
  ns->t    = time;
  PetscFunctionReturn(PETSC_SUCCESS);
}
