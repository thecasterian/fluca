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
  m->dim          = PETSC_DETERMINE;
  m->sdm          = NULL;
  m->vdm          = NULL;
  m->Sdm          = NULL;
  m->Vdm          = NULL;
  m->data         = NULL;
  m->outputseqnum = -1;
  m->outputseqval = 0.;
  m->setupcalled  = PETSC_FALSE;

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

PetscErrorCode MeshView(Mesh mesh, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mesh), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mesh, viewer));
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
  PetscCall(PetscViewerCheckReadable(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) PetscUseTypeMethod(mesh, load, viewer);
  else SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerFlucaCGNSOpen()");
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

PetscErrorCode MeshGetDM(Mesh mesh, MeshDMType type, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(dm, 3);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Mesh not setup");
  switch (type) {
  case MESH_DM_SCALAR:
    *dm = mesh->sdm;
    break;
  case MESH_DM_VECTOR:
    *dm = mesh->vdm;
    break;
  case MESH_DM_STAG_SCALAR:
    *dm = mesh->Sdm;
    break;
  case MESH_DM_STAG_VECTOR:
    *dm = mesh->Vdm;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Invalid MeshDMType %d", (int)type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCreateGlobalVector(Mesh mesh, MeshDMType type, Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(vec, 3);
  PetscUseTypeMethod(mesh, createglobalvector, type, vec);
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

PetscErrorCode MeshSetOutputSequenceNumber(Mesh mesh, PetscInt num, PetscReal val)
{
  PetscFunctionBegin;
  mesh->outputseqnum = num;
  mesh->outputseqval = val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetOutputSequenceNumber(Mesh mesh, PetscInt *num, PetscReal *val)
{
  PetscFunctionBegin;
  if (num) *num = mesh->outputseqnum;
  if (val) *val = mesh->outputseqval;
  PetscFunctionReturn(PETSC_SUCCESS);
}
