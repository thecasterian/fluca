#include <fluca/private/meshimpl.h>
#include <flucaviewer.h>

const char *MeshBoundaryTypes[] = {"NONE", "PERIODIC", "MeshBoundaryType", "", NULL};

PetscClassId  MESH_CLASSID = 0;
PetscLogEvent MESH_SetUp   = 0;

PetscFunctionList MeshList              = NULL;
PetscBool         MeshRegisterAllCalled = PETSC_FALSE;

PetscErrorCode MeshCreate(MPI_Comm comm, Mesh *mesh)
{
  Mesh m;

  PetscFunctionBegin;
  PetscAssertPointer(mesh, 2);

  PetscCall(MeshInitializePackage());
  PetscCall(FlucaHeaderCreate(m, MESH_CLASSID, "Mesh", "Mesh", "Mesh", comm, MeshDestroy, MeshView));
  m->dim         = PETSC_DETERMINE;
  m->dm          = NULL;
  m->fdm         = NULL;
  m->data        = NULL;
  m->setupcalled = PETSC_FALSE;

  *mesh = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetType(Mesh mesh, MeshType type)
{
  MeshType old_type;
  PetscErrorCode (*impl_create)(Mesh);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

  PetscCall(MeshGetType(mesh, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)mesh, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(MeshList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown mesh type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(mesh, destroy);
    PetscCall(PetscMemzero(mesh->ops, sizeof(struct _MeshOps)));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)mesh, type));
  PetscCall((*impl_create)(mesh));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetType(Mesh mesh, MeshType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(MeshRegisterAll());
  *type = ((PetscObject)mesh)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetUp(Mesh mesh)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (mesh->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(MESH_SetUp, (PetscObject)mesh, 0, 0, 0));

  /* Set default type */
  if (!((PetscObject)mesh)->type_name) PetscCall(MeshSetType(mesh, MESHCART));

  /* Validate */
  PetscCheck(MESH_MIN_DIM <= mesh->dim && mesh->dim <= MESH_MAX_DIM, PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension %d", mesh->dim);

  /* Call specific type setup */
  PetscTryTypeMethod(mesh, setup);

  PetscCall(PetscLogEventEnd(MESH_SetUp, (PetscObject)mesh, 0, 0, 0));

  /* Viewers */
  PetscCall(MeshViewFromOptions(mesh, NULL, "-mesh_view"));

  mesh->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView(Mesh mesh, PetscViewer v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mesh), &v));
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(mesh, 1, v, 2);

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mesh, v));
  PetscTryTypeMethod(mesh, view, v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshViewFromOptions(Mesh mesh, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)mesh, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy(Mesh *mesh)
{
  PetscFunctionBegin;
  if (!*mesh) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*mesh), MESH_CLASSID, 1);

  if (--((PetscObject)(*mesh))->refct > 0) {
    *mesh = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod((*mesh), destroy);
  PetscCall(PetscHeaderDestroy(mesh));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetDM(Mesh mesh, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Mesh not setup");
  *dm = mesh->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetFaceDM(Mesh mesh, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Mesh not setup");
  *dm = mesh->fdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetNumberBoundaries(Mesh mesh, PetscInt *nb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(nb, 2);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Mesh not setup");
  PetscTryTypeMethod(mesh, getnumberboundaries, nb);
  PetscFunctionReturn(PETSC_SUCCESS);
}
