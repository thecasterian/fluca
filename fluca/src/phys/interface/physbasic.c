#include <fluca/private/physimpl.h>
#include <flucaviewer.h>

PetscClassId  PHYS_CLASSID = 0;
PetscLogEvent PHYS_SetUp   = 0;

PetscFunctionList PhysList              = NULL;
PetscBool         PhysRegisterAllCalled = PETSC_FALSE;

const char *PhysINSBCTypes[] = {"NONE", "VELOCITY", "PhysINSBCType", "", NULL};

PetscErrorCode PhysCreate(MPI_Comm comm, Phys *phys)
{
  Phys p;

  PetscFunctionBegin;
  PetscAssertPointer(phys, 2);

  PetscCall(PhysInitializePackage());
  PetscCall(FlucaHeaderCreate(p, PHYS_CLASSID, "Phys", "Physical Model", "Phys", comm, PhysDestroy, PhysView));
  p->base_dm       = NULL;
  p->bodyforce     = NULL;
  p->bodyforce_ctx = NULL;
  p->sol_dm        = NULL;
  p->dim           = PETSC_DETERMINE;
  p->data          = NULL;
  p->setupcalled   = PETSC_FALSE;

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
  PetscFunctionBegin;
  if (!*phys) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*phys), PHYS_CLASSID, 1);

  if (--((PetscObject)(*phys))->refct > 0) {
    *phys = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Call type-specific destroy */
  PetscTryTypeMethod((*phys), destroy);

  PetscCall(DMDestroy(&(*phys)->sol_dm));
  PetscCall(DMDestroy(&(*phys)->base_dm));

  PetscCall(PetscHeaderDestroy(phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetUp(Phys phys)
{
  PetscBool isdmstag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (phys->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(PHYS_SetUp, (PetscObject)phys, 0, 0, 0));

  /* Validate base DM */
  PetscCheck(phys->base_dm, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Base DM not set. Call PhysSetBaseDM() first");
  PetscCall(PetscObjectTypeCompare((PetscObject)phys->base_dm, DMSTAG, &isdmstag));
  PetscCheck(isdmstag, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Base DM must be DMStag");

  /* Extract dimension */
  PetscCall(DMGetDimension(phys->base_dm, &phys->dim));

  /* Call subtype createsolutiondm */
  PetscCheck(phys->ops->createsolutiondm, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Phys type not set or subtype does not implement createsolutiondm");
  PetscCall((*phys->ops->createsolutiondm)(phys));

  /* Call subtype setup */
  PetscTryTypeMethod(phys, setup);

  PetscCall(PetscLogEventEnd(PHYS_SetUp, (PetscObject)phys, 0, 0, 0));

  phys->setupcalled = PETSC_TRUE;

  PetscCall(PhysViewFromOptions(phys, NULL, "-phys_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysView(Phys phys, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)phys), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(phys, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)phys, viewer));
    if (phys->setupcalled) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Dimension: %" PetscInt_FMT "\n", phys->dim));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
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

PetscErrorCode PhysSetUpTS(Phys phys, TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysSetUpTS()");
  PetscUseTypeMethod(phys, setupts, ts);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysComputeIFunction(Phys phys, PetscReal t, Vec y, Vec y_dot, Vec F)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysComputeIFunction()");
  PetscUseTypeMethod(phys, computeifunction, t, y, y_dot, F);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysComputeIJacobian(Phys phys, PetscReal t, Vec y, Vec y_dot, PetscReal shift, Mat J, Mat Jpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysComputeIJacobian()");
  PetscUseTypeMethod(phys, computeijacobian, t, y, y_dot, shift, J, Jpre);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysComputeRHSFunction(Phys phys, PetscReal t, Vec y, Vec G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysComputeRHSFunction()");
  PetscUseTypeMethod(phys, computerhsfunction, t, y, G);
  PetscFunctionReturn(PETSC_SUCCESS);
}
