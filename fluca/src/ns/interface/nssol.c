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

static PetscErrorCode CheckExtension_Private(const char filename[], const char extension[], PetscBool *is_extension)
{
  size_t len_filename, len_extension;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(filename, &len_filename));
  PetscCall(PetscStrlen(extension, &len_extension));
  if (len_filename < len_extension) {
    *is_extension = PETSC_FALSE;
  } else {
    PetscCall(PetscStrcmp(filename + len_filename - len_extension, extension, is_extension));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSLoadSolutionFromFile(NS ns, const char filename[])
{
  const char *ext_cgns = ".cgns";
  PetscBool   iscgns;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(filename, 2);
  PetscCall(PetscLogEventBegin(NS_LoadSolutionFromFile, ns, 0, 0, 0));

  PetscCall(CheckExtension_Private(filename, ext_cgns, &iscgns));
  if (iscgns) PetscCall(NSLoadSolutionCGNSFromFile(ns, filename));
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot load file %s: unrecognized extension", filename);

  PetscCall(PetscLogEventEnd(NS_LoadSolutionFromFile, ns, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSLoadSolutionCGNS(NS ns, PetscInt file_num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCheck(ns->mesh, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Mesh is not set");
  PetscTryTypeMethod(ns, loadsolutioncgns, file_num);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSLoadSolutionCGNSFromFile(NS ns, const char filename[])
{
  int file_num = -1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(filename, 2);

  CGNSCall(cgp_mpi_comm(PetscObjectComm((PetscObject)ns)));
  CGNSCall(cgp_open(filename, CG_MODE_READ, &file_num));
  PetscCheck(file_num > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cgp_open(\"%s\", ...) did not return a valid file number", filename);
  PetscCall(NSLoadSolutionCGNS(ns, file_num));
  CGNSCall(cgp_close(file_num));
  PetscFunctionReturn(PETSC_SUCCESS);
}
