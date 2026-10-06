#include <fluca/private/meshimpl.h>
#include <flucaviewer.h>
#include <petsc/private/vecimpl.h>

#define MESH_VEC_MAX_FIELDS     16
#define MESH_VEC_MAX_NAME_LEN   64
#define MESH_VEC_COMPOSED_MESH  "Fluca_Mesh"
#define MESH_VEC_COMPOSED_SPECS "Fluca_MeshVecFields"

/* The field description of a vector, composed on it in a PetscContainer. Fixed-size so that the container can be
   freed with PetscCtxDestroyDefault() */
typedef struct {
  PetscInt nfields;
  struct {
    char                  name[MESH_VEC_MAX_NAME_LEN];
    DMStagStencilLocation loc;
    PetscInt              c0;
    PetscInt              ncomp;
  } fields[MESH_VEC_MAX_FIELDS];
  PetscErrorCode (*view_default)(Vec, PetscViewer); /* VECOP_VIEW before MeshVecSetFields_Internal(), for non-CGNS viewers */
  PetscErrorCode (*load_default)(Vec, PetscViewer); /* VECOP_LOAD before MeshVecSetFields_Internal(), for non-CGNS viewers */
} MeshVecFields;

static PetscErrorCode VecGetMeshFields_Private(Vec v, Mesh *mesh, MeshVecFields **fields)
{
  PetscContainer container;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)v, MESH_VEC_COMPOSED_MESH, (PetscObject *)mesh));
  PetscCall(PetscObjectQuery((PetscObject)v, MESH_VEC_COMPOSED_SPECS, (PetscObject *)&container));
  PetscCheck(*mesh && container, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector has no Mesh field description");
  PetscCall(PetscContainerGetPointer(container, (void **)fields));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecView_Mesh_Private(Vec v, PetscViewer viewer)
{
  Mesh           mesh;
  MeshVecFields *mf;
  PetscInt       f;
  PetscBool      iscgns;

  PetscFunctionBegin;
  PetscCall(VecGetMeshFields_Private(v, &mesh, &mf));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (!iscgns) {
    PetscCall((*mf->view_default)(v, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (f = 0; f < mf->nfields; ++f) PetscUseTypeMethod(mesh, viewveccomponents, v, mf->fields[f].loc, mf->fields[f].c0, mf->fields[f].ncomp, mf->fields[f].name, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecLoad_Mesh_Private(Vec v, PetscViewer viewer)
{
  Mesh           mesh;
  MeshVecFields *mf;
  PetscInt       f;
  PetscBool      iscgns;

  PetscFunctionBegin;
  PetscCall(VecGetMeshFields_Private(v, &mesh, &mf));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (!iscgns) {
    PetscCall((*mf->load_default)(v, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscViewerCheckReadable(viewer));
  for (f = 0; f < mf->nfields; ++f) PetscUseTypeMethod(mesh, loadveccomponents, v, mf->fields[f].loc, mf->fields[f].c0, mf->fields[f].ncomp, mf->fields[f].name, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The vector must live on a DMStag with the Mesh DM's global sizes and boundary types that has the field's components */
static PetscErrorCode MeshCheckVecField_Private(Mesh mesh, Vec v, const MeshField *field)
{
  DM             dm;
  PetscBool      isstag;
  PetscInt       d, M[3], vM[3], dof[4], ndof;
  DMBoundaryType bt[3], vbt[3];
  size_t         len;

  PetscFunctionBegin;
  PetscAssertPointer(field->name, 3);
  PetscCall(PetscStrlen(field->name, &len));
  PetscCheck(len > 0 && len < MESH_VEC_MAX_NAME_LEN, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Field name \"%s\" must have 1 to %d characters", field->name, MESH_VEC_MAX_NAME_LEN - 1);
  PetscCheck(field->loc == DMSTAG_ELEMENT || field->loc == DMSTAG_LEFT, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Field %s: location must be DMSTAG_ELEMENT or DMSTAG_LEFT", field->name);
  PetscCheck(field->c0 >= 0 && field->ncomp > 0, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Field %s: invalid component range [%" PetscInt_FMT ", %" PetscInt_FMT ")", field->name, field->c0, field->c0 + field->ncomp);
  PetscCall(VecGetDM(v, &dm));
  PetscCheck(dm, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector has no DM");
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMSTAG, &isstag));
  PetscCheck(isstag, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_INCOMP, "Vector's DM is not a DMSTAG");
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M[0], &M[1], &M[2]));
  PetscCall(DMStagGetGlobalSizes(dm, &vM[0], &vM[1], &vM[2]));
  PetscCall(DMStagGetBoundaryTypes(mesh->dm, &bt[0], &bt[1], &bt[2]));
  PetscCall(DMStagGetBoundaryTypes(dm, &vbt[0], &vbt[1], &vbt[2]));
  for (d = 0; d < mesh->dim; ++d) {
    PetscCheck(vM[d] == M[d], PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_INCOMP, "Vector's DM global size %" PetscInt_FMT " in direction %" PetscInt_FMT " does not match the Mesh DM's %" PetscInt_FMT, vM[d], d, M[d]);
    PetscCheck(vbt[d] == bt[d], PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_INCOMP, "Vector's DM boundary type %s in direction %" PetscInt_FMT " does not match the Mesh DM's %s", DMBoundaryTypes[vbt[d]], d, DMBoundaryTypes[bt[d]]);
  }
  PetscCall(DMStagGetDOF(dm, &dof[0], &dof[1], &dof[2], &dof[3]));
  ndof = field->loc == DMSTAG_ELEMENT ? dof[mesh->dim] : dof[mesh->dim - 1];
  PetscCheck(field->c0 + field->ncomp <= ndof, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_OUTOFRANGE, "Field %s: component range [%" PetscInt_FMT ", %" PetscInt_FMT ") exceeds the %" PetscInt_FMT " DOF available at its location", field->name,
             field->c0, field->c0 + field->ncomp, ndof);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshVecSetFields_Internal(Mesh mesh, Vec v, PetscInt nfields, const MeshField fields[])
{
  PetscContainer container;
  MeshVecFields *mf;
  PetscInt       f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  if (nfields) PetscAssertPointer(fields, 4);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Must call MeshSetUp() first");
  PetscCheck(nfields >= 0 && nfields <= MESH_VEC_MAX_FIELDS, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %" PetscInt_FMT " out of range [0, %d]", nfields, MESH_VEC_MAX_FIELDS);
  for (f = 0; f < nfields; ++f) PetscCall(MeshCheckVecField_Private(mesh, v, &fields[f]));

  PetscCall(PetscNew(&mf));
  mf->nfields = nfields;
  for (f = 0; f < nfields; ++f) {
    PetscCall(PetscStrncpy(mf->fields[f].name, fields[f].name, sizeof(mf->fields[f].name)));
    mf->fields[f].loc   = fields[f].loc;
    mf->fields[f].c0    = fields[f].c0;
    mf->fields[f].ncomp = fields[f].ncomp;
  }
  /* A vector described before keeps the defaults saved the first time */
  PetscCall(PetscObjectQuery((PetscObject)v, MESH_VEC_COMPOSED_SPECS, (PetscObject *)&container));
  if (container) {
    MeshVecFields *old;

    PetscCall(PetscContainerGetPointer(container, (void **)&old));
    mf->view_default = old->view_default;
    mf->load_default = old->load_default;
  } else {
    mf->view_default = v->ops->view;
    mf->load_default = v->ops->load;
  }

  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)v), &container));
  PetscCall(PetscContainerSetPointer(container, mf));
  PetscCall(PetscContainerSetCtxDestroy(container, PetscCtxDestroyDefault));
  PetscCall(PetscObjectCompose((PetscObject)v, MESH_VEC_COMPOSED_SPECS, (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscCall(PetscObjectCompose((PetscObject)v, MESH_VEC_COMPOSED_MESH, (PetscObject)mesh));
  PetscCall(VecSetOperation(v, VECOP_VIEW, (void (*)(void))VecView_Mesh_Private));
  PetscCall(VecSetOperation(v, VECOP_LOAD, (void (*)(void))VecLoad_Mesh_Private));
  PetscFunctionReturn(PETSC_SUCCESS);
}
