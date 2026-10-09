#include <fluca/private/meshimpl.h>

PetscErrorCode MeshSetDM(Mesh mesh, DM dm)
{
  PetscInt dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 2, DMSTAG);
  PetscCheckSameComm(mesh, 1, dm, 2);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the DM after MeshSetUp()");
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(MESH_MIN_DIM <= dim && dim <= MESH_MAX_DIM, PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(DMDestroy(&mesh->dm));
  mesh->dm  = dm;
  mesh->dim = dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetDM(Mesh mesh, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  PetscCheck(mesh->dm, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "DM not set. Call MeshSetDM() or MeshLoad() first");
  *dm = mesh->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetDimension(Mesh mesh, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(dim, 2);
  PetscCheck(mesh->dm, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "DM not set. Call MeshSetDM() or MeshLoad() first");
  *dim = mesh->dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetFromOptions(Mesh mesh)
{
  char      type[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(MeshRegisterAll());

  PetscObjectOptionsBegin((PetscObject)mesh);
  PetscCall(PetscOptionsFList("-mesh_type", "Mesh type", "MeshSetType", MeshList, (char *)(((PetscObject)mesh)->type_name ? ((PetscObject)mesh)->type_name : MESHCARTESIAN), type, sizeof(type), &flg));
  if (flg) PetscCall(MeshSetType(mesh, type));
  else if (!((PetscObject)mesh)->type_name) PetscCall(MeshSetType(mesh, MESHCARTESIAN));
  PetscTryTypeMethod(mesh, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
