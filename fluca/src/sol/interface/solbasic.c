#include <fluca/private/solimpl.h>
#include <flucaviewer.h>
#include <petscdraw.h>

PetscClassId  SOL_CLASSID      = 0;
PetscLogEvent SOL_LoadFromFile = 0;

PetscFunctionList SolList              = NULL;
PetscBool         SolRegisterAllCalled = PETSC_FALSE;

PetscErrorCode SolCreate(MPI_Comm comm, Sol *sol)
{
  Sol s;

  PetscFunctionBegin;
  *sol = NULL;
  PetscCall(SolInitializePackage());

  PetscCall(FlucaHeaderCreate(s, SOL_CLASSID, "Sol", "Solver Solution", "Sol", comm, SolDestroy, SolView));

  s->mesh = NULL;
  s->data = NULL;

  *sol = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolSetType(Sol sol, SolType type)
{
  SolType old_type;
  PetscErrorCode (*impl_create)(Sol);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);

  PetscCall(SolGetType(sol, &old_type));

  PetscCall(PetscObjectTypeCompare((PetscObject)sol, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(SolList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)sol), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown sol type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(sol, destroy);
    PetscCall(PetscMemzero(sol->ops, sizeof(struct _SolOps)));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)sol, type));
  PetscCall((*impl_create)(sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetType(Sol sol, SolType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  PetscCall(SolRegisterAll());
  *type = ((PetscObject)sol)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolSetMesh(Sol sol, Mesh mesh)
{
  DM       dm;
  PetscInt dim, d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 2);
  PetscCheckSameComm(sol, 1, mesh, 2);
  if (sol->mesh == mesh) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MeshDestroy(&sol->mesh));
  for (d = 0; d < 3; ++d) PetscCall(VecDestroy(&sol->v[d]));
  PetscCall(VecDestroy(&sol->p));

  sol->mesh = mesh;
  PetscCall(PetscObjectReference((PetscObject)mesh));

  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(MeshGetDimension(mesh, &dim));

  for (d = 0; d < dim; ++d) PetscCall(DMCreateLocalVector(dm, &sol->v[d]));
  PetscCall(DMCreateLocalVector(dm, &sol->p));

  PetscTryTypeMethod(sol, setmesh, mesh);

  PetscCall(SolViewFromOptions(sol, NULL, "-sol_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetMesh(Sol sol, Mesh *mesh)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  *mesh = sol->mesh;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetVelocity(Sol sol, Vec *u, Vec *v, Vec *w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  if (u) *u = sol->v[0];
  if (v) *v = sol->v[1];
  if (w) *w = sol->v[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolGetPressure(Sol sol, Vec *p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  if (p) *p = sol->p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolView(Sol sol, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sol), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(sol, 1, viewer, 2);

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sol, viewer));
  PetscTryTypeMethod(sol, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolViewFromOptions(Sol sol, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)sol, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolDestroy(Sol *sol)
{
  PetscInt d;

  PetscFunctionBegin;
  if (!*sol) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*sol, SOL_CLASSID, 1);

  if (--((PetscObject)(*sol))->refct > 0) {
    *sol = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(MeshDestroy(&(*sol)->mesh));
  for (d = 0; d < 3; ++d) PetscCall(VecDestroy(&(*sol)->v[d]));
  PetscCall(VecDestroy(&(*sol)->p));

  PetscTryTypeMethod((*sol), destroy);
  PetscCall(PetscHeaderDestroy(sol));
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

PetscErrorCode SolLoadFromFile(Sol sol, const char filename[])
{
  const char *ext_cgns = ".cgns";
  PetscBool   is_cgns;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  PetscAssertPointer(filename, 2);
  PetscCall(PetscLogEventBegin(SOL_LoadFromFile, sol, 0, 0, 0));

  PetscCall(CheckExtension_Private(filename, ext_cgns, &is_cgns));
  if (is_cgns) PetscCall(SolLoadCGNSFromFile(sol, filename));
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot load file %s: unrecognized extension", filename);

  PetscCall(PetscLogEventEnd(SOL_LoadFromFile, sol, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
