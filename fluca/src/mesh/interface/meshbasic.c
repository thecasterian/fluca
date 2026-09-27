#include <fluca/private/meshimpl.h>
#include <flucaviewer.h>

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
  m->dm          = NULL;
  m->dim         = PETSC_DETERMINE;
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
  PetscAssertPointer(type, 2);
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

  if (!((PetscObject)mesh)->type_name) PetscCall(MeshSetType(mesh, MESHCARTESIAN));
  PetscCheck(mesh->dm, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "DM not set. Call MeshSetDM() or MeshLoad() first");
  PetscTryTypeMethod(mesh, setup);

  PetscCall(PetscLogEventEnd(MESH_SetUp, (PetscObject)mesh, 0, 0, 0));
  mesh->setupcalled = PETSC_TRUE;

  PetscCall(MeshViewFromOptions(mesh, NULL, "-mesh_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView(Mesh mesh, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mesh), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mesh, viewer));
  PetscTryTypeMethod(mesh, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshViewFromOptions(Mesh mesh, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)mesh, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshLoad(Mesh mesh, PetscViewer viewer)
{
  PetscBool iscgns;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Cannot load a mesh after MeshSetUp()");
  PetscCall(PetscViewerCheckReadable(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  PetscCheck(iscgns, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerFlucaCGNSOpen()");
  if (!((PetscObject)mesh)->type_name) PetscCall(MeshSetType(mesh, MESHCARTESIAN));
  PetscUseTypeMethod(mesh, load, viewer);
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
  PetscCall(DMDestroy(&(*mesh)->dm));
  PetscCall(PetscHeaderDestroy(mesh));
  PetscFunctionReturn(PETSC_SUCCESS);
}
