#include <fluca/private/nsimpl.h>
#include <fluca/private/flucaviewercgnsimpl.h>

PetscErrorCode NSViewSolution(NS ns, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ns), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ns, 1, viewer, 2);

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ns, viewer));
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
