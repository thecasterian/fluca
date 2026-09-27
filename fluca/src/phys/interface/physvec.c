#include <fluca/private/physimpl.h>
#include <fluca/private/meshimpl.h>
#include <flucaviewer.h>
#include <petsc/private/vecimpl.h>

/* The Phys that created v; the layout must still be its current solution DM */
static PetscErrorCode VecGetPhys_Private(Vec v, Phys *phys)
{
  DM dm;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)v, "Fluca_Phys", (PetscObject *)phys));
  PetscCheck(*phys, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not created by PhysCreateSolutionVector()");
  PetscCall(VecGetDM(v, &dm));
  PetscCheck((*phys)->setupcalled && dm == (*phys)->sol_dm, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONGSTATE, "Vector layout is stale; the Phys was set up again after the vector was created");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecView_Phys_Private(Vec v, PetscViewer viewer)
{
  Phys              phys;
  PhysFieldLocation loc;
  PetscInt          f, c0, ncomp;
  PetscBool         iscgns;

  PetscFunctionBegin;
  PetscCall(VecGetPhys_Private(v, &phys));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (!iscgns) {
    PetscCall((*phys->vecview_default)(v, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PhysGetField(phys, phys->fields[f].name, &loc, &c0, &ncomp));
    PetscCall(MeshViewVecComponents_Internal(phys->mesh, v, loc == PHYS_FIELD_FACE ? DMSTAG_LEFT : DMSTAG_ELEMENT, c0, ncomp, phys->fields[f].name, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecLoad_Phys_Private(Vec v, PetscViewer viewer)
{
  Phys              phys;
  PhysFieldLocation loc;
  PetscInt          f, c0, ncomp;
  PetscBool         iscgns;

  PetscFunctionBegin;
  PetscCall(VecGetPhys_Private(v, &phys));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  PetscCheck(iscgns, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Solution vectors load only from a CGNS viewer; open it with PetscViewerFlucaCGNSOpen()");
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PhysGetField(phys, phys->fields[f].name, &loc, &c0, &ncomp));
    PetscCall(MeshLoadVecComponents_Internal(phys->mesh, v, loc == PHYS_FIELD_FACE ? DMSTAG_LEFT : DMSTAG_ELEMENT, c0, ncomp, phys->fields[f].name, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreateSolutionVector(Phys phys, Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(v, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysCreateSolutionVector()");
  PetscCall(DMCreateGlobalVector(phys->sol_dm, v));
  if (!phys->vecview_default) phys->vecview_default = (*v)->ops->view;
  PetscCall(PetscObjectCompose((PetscObject)*v, "Fluca_Phys", (PetscObject)phys));
  PetscCall(VecSetOperation(*v, VECOP_VIEW, (void (*)(void))VecView_Phys_Private));
  PetscCall(VecSetOperation(*v, VECOP_LOAD, (void (*)(void))VecLoad_Phys_Private));
  PetscFunctionReturn(PETSC_SUCCESS);
}
