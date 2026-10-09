#include <fluca/private/physimpl.h>
#include <fluca/private/flucaviewercgnsimpl.h>
#include <petsc/private/vecimpl.h>

#define PHYS_VEC_MAX_NAME_LEN   64
#define PHYS_VEC_COMPOSED_MESH  "Fluca_Mesh"
#define PHYS_VEC_COMPOSED_SPECS "Fluca_PhysVecFields"

/* The field layout of a solution vector, composed on it in a PetscContainer so that VecDuplicate() carries it. It
   describes the vector's own DM, so it stays valid if the Phys is set up again. Fixed-size so that the container can be
   freed with PetscCtxDestroyDefault(). */
typedef struct {
  PetscInt nfields;
  struct {
    char                  name[PHYS_VEC_MAX_NAME_LEN];
    DMStagStencilLocation loc; /* DMSTAG_ELEMENT, or DMSTAG_LEFT for every face orientation */
    PetscInt              c0;
    PetscInt              ncomp;
  } fields[PHYS_MAX_FIELDS];
  PetscErrorCode (*view_default)(Vec, PetscViewer); /* VECOP_VIEW of a plain solution-DM vector, for non-CGNS viewers */
  PetscErrorCode (*load_default)(Vec, PetscViewer); /* VECOP_LOAD of a plain solution-DM vector, for non-CGNS viewers */
} PhysVecFields;

static PetscErrorCode VecGetPhysFields_Private(Vec v, Mesh *mesh, PhysVecFields **pf)
{
  PetscContainer container;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)v, PHYS_VEC_COMPOSED_MESH, (PetscObject *)mesh));
  PetscCall(PetscObjectQuery((PetscObject)v, PHYS_VEC_COMPOSED_SPECS, (PetscObject *)&container));
  PetscCheck(*mesh && container, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not created by PhysCreateSolutionVector()");
  PetscCall(PetscContainerGetPointer(container, (void **)pf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecView_Phys_Private(Vec v, PetscViewer viewer)
{
  Mesh           mesh;
  PhysVecFields *pf;
  DM             dm;
  PetscInt       f, step;
  PetscReal      time;
  PetscBool      iscgns;

  PetscFunctionBegin;
  PetscCall(VecGetPhysFields_Private(v, &mesh, &pf));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (!iscgns) {
    PetscCall((*pf->view_default)(v, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetOutputSequenceNumber(dm, &step, &time));
  if (step < 0) {
    step = 0;
    time = 0.;
  }
  PetscCall(PetscViewerFlucaCGNSBeginStep_Internal(viewer, step, time));
  PetscCall(MeshView(mesh, viewer)); /* the grid zone, once per file */
  for (f = 0; f < pf->nfields; ++f) PetscCall(PetscViewerFlucaCGNSWriteDMStagComponents_Internal(viewer, v, pf->fields[f].loc, pf->fields[f].c0, pf->fields[f].ncomp, pf->fields[f].name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecLoad_Phys_Private(Vec v, PetscViewer viewer)
{
  Mesh           mesh;
  PhysVecFields *pf;
  PetscInt       f;
  PetscBool      iscgns;

  PetscFunctionBegin;
  PetscCall(VecGetPhysFields_Private(v, &mesh, &pf));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (!iscgns) {
    PetscCall((*pf->load_default)(v, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (f = 0; f < pf->nfields; ++f) PetscCall(PetscViewerFlucaCGNSReadDMStagComponents_Internal(viewer, v, pf->fields[f].loc, pf->fields[f].c0, pf->fields[f].ncomp, pf->fields[f].name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreateSolutionVector(Phys phys, Vec *v)
{
  PetscContainer    container;
  PhysVecFields    *pf;
  PhysFieldLocation loc;
  PetscInt          f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(v, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysCreateSolutionVector()");
  PetscCall(DMCreateGlobalVector(phys->sol_dm, v));

  PetscCall(PetscNew(&pf));
  pf->nfields = phys->nfields;
  for (f = 0; f < phys->nfields; ++f) {
    size_t len;

    PetscCall(PetscStrlen(phys->fields[f].name, &len));
    PetscCheck(len < PHYS_VEC_MAX_NAME_LEN, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Field name %s is longer than %d characters", phys->fields[f].name, PHYS_VEC_MAX_NAME_LEN - 1);
    PetscCall(PhysGetField(phys, phys->fields[f].name, &loc, &pf->fields[f].c0, &pf->fields[f].ncomp));
    PetscCall(PetscStrncpy(pf->fields[f].name, phys->fields[f].name, sizeof(pf->fields[f].name)));
    pf->fields[f].loc = loc == PHYS_FIELD_FACE ? DMSTAG_LEFT : DMSTAG_ELEMENT;
  }
  pf->view_default = (*v)->ops->view;
  pf->load_default = (*v)->ops->load;
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)*v), &container));
  PetscCall(PetscContainerSetPointer(container, pf));
  PetscCall(PetscContainerSetCtxDestroy(container, PetscCtxDestroyDefault));
  PetscCall(PetscObjectCompose((PetscObject)*v, PHYS_VEC_COMPOSED_SPECS, (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscCall(PetscObjectCompose((PetscObject)*v, PHYS_VEC_COMPOSED_MESH, (PetscObject)phys->mesh));
  PetscCall(VecSetOperation(*v, VECOP_VIEW, (void (*)(void))VecView_Phys_Private));
  PetscCall(VecSetOperation(*v, VECOP_LOAD, (void (*)(void))VecLoad_Phys_Private));
  PetscFunctionReturn(PETSC_SUCCESS);
}
