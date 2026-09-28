#include <fluca/private/physimpl.h>
#include <fluca/private/meshimpl.h>
#include <flucaviewer.h>

PetscClassId  PHYS_CLASSID = 0;
PetscLogEvent PHYS_SetUp   = 0;

PetscFunctionList PhysList              = NULL;
PetscBool         PhysRegisterAllCalled = PETSC_FALSE;

const char *PhysBCTypes[] = {"NONE", "VELOCITY", "PhysBCType", "PHYS_BC_", NULL};

PetscErrorCode PhysCreate(MPI_Comm comm, Phys *phys)
{
  Phys     p;
  PetscInt f;

  PetscFunctionBegin;
  PetscAssertPointer(phys, 2);

  PetscCall(PhysInitializePackage());
  PetscCall(FlucaHeaderCreate(p, PHYS_CLASSID, "Phys", "Physical Model", "Phys", comm, PhysDestroy, PhysView));
  p->mesh            = NULL;
  p->bodyforce       = NULL;
  p->bodyforce_ctx   = NULL;
  p->nprops          = 0;
  p->nfields         = 0;
  p->sol_dm          = NULL;
  p->dim             = PETSC_DETERMINE;
  p->data            = NULL;
  p->vecview_default = NULL;
  p->vecload_default = NULL;
  p->setupcalled     = PETSC_FALSE;
  for (f = 0; f < PHYS_MAX_FACES; f++) {
    p->bcs[f].type       = PHYS_BC_NONE;
    p->bcs[f].fn         = NULL;
    p->bcs[f].ctx        = NULL;
    p->bcs[f].fn_dot     = NULL;
    p->bcs[f].fn_dot_ctx = NULL;
  }
  PetscCall(PhysRegisterProperty_Internal(p, PHYS_PROPERTY_DENSITY, 1.));
  PetscCall(PhysRegisterProperty_Internal(p, PHYS_PROPERTY_VISCOSITY, 1.));

  *phys = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetType(Phys phys, PhysType type)
{
  PhysType old_type;
  PetscErrorCode (*impl_create)(Phys);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);

  PetscCall(PhysGetType(phys, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)phys, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PhysList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Phys type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(phys, destroy);
    PetscCall(PetscMemzero(phys->ops, sizeof(struct _PhysOps)));
  }
  /* The fields and the solution DM belong to the old type; the new subtype declares into an empty table */
  PetscCall(PhysResetFields(phys));

  PetscCall(PetscObjectChangeTypeName((PetscObject)phys, type));
  PetscCall((*impl_create)(phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetType(Phys phys, PhysType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(PhysRegisterAll());
  *type = ((PetscObject)phys)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysDestroy(Phys *phys)
{
  PetscInt p;

  PetscFunctionBegin;
  if (!*phys) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*phys), PHYS_CLASSID, 1);

  if (--((PetscObject)(*phys))->refct > 0) {
    *phys = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod((*phys), destroy);

  for (p = 0; p < (*phys)->nprops; ++p) PetscCall(PetscFree((*phys)->props[p].name));
  PetscCall(PhysResetFields(*phys));
  PetscCall(MeshDestroy(&(*phys)->mesh));

  PetscCall(PetscHeaderDestroy(phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetUp(Phys phys)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (phys->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(PHYS_SetUp, (PetscObject)phys, 0, 0, 0));

  PetscCheck(phys->mesh, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Mesh not set. Call PhysSetMesh() first");
  PetscCheck(phys->mesh->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Mesh not set up. Call MeshSetUp() first");
  PetscCall(MeshGetDimension(phys->mesh, &phys->dim));

  PetscCheck(phys->nfields > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "No field declared. Call PhysDeclareField() first");
  PetscTryTypeMethod(phys, setup);
  PetscCall(PhysCreateSolutionDM_Internal(phys));

  PetscCall(PetscLogEventEnd(PHYS_SetUp, (PetscObject)phys, 0, 0, 0));

  phys->setupcalled = PETSC_TRUE;

  PetscCall(PhysViewFromOptions(phys, NULL, "-phys_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysView(Phys phys, PetscViewer viewer)
{
  PetscBool isascii;
  PetscReal rho, mu;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)phys), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(phys, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)phys, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PhysGetDensity(phys, &rho));
    PetscCall(PhysGetViscosity(phys, &mu));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Density: %g\n", (double)rho));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Viscosity: %g\n", (double)mu));
    if (phys->setupcalled) PetscCall(PetscViewerASCIIPrintf(viewer, "Dimension: %" PetscInt_FMT "\n", phys->dim));
    if (phys->mesh) PetscCall(MeshView(phys->mesh, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }

  PetscTryTypeMethod(phys, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysViewFromOptions(Phys phys, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)phys, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}
