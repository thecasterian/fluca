#include <fluca/private/meshimpl.h>

PetscErrorCode MeshSetDimension(Mesh mesh, PetscInt dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called before MeshSetUp()");
  mesh->dim = dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetDimension(Mesh mesh, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
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
  PetscCall(PetscOptionsFList("-mesh_type", "Mesh type", "MeshSetType", MeshList, (char *)(((PetscObject)mesh)->type_name ? ((PetscObject)mesh)->type_name : MESHCART), type, sizeof(type), &flg));
  if (flg) PetscCall(MeshSetType(mesh, type));
  else if (!((PetscObject)mesh)->type_name) PetscCall(MeshSetType(mesh, MESHCART));
  PetscTryTypeMethod(mesh, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
