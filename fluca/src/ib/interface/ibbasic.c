#include <fluca/private/flucaibimpl.h>
#include <flucaviewer.h>
#include <flucasys.h>

PetscClassId  FLUCAIB_CLASSID = 0;
PetscLogEvent FLUCAIB_SetUp   = 0;
PetscLogEvent FLUCAIB_ApplyBC = 0;

PetscFunctionList FlucaIBList              = NULL;
PetscBool         FlucaIBRegisterAllCalled = PETSC_FALSE;

PetscErrorCode FlucaIBCreate(MPI_Comm comm, FlucaIB *ib)
{
  FlucaIB i;

  PetscFunctionBegin;
  PetscAssertPointer(ib, 2);

  PetscCall(FlucaIBInitializePackage());
  PetscCall(FlucaHeaderCreate(i, FLUCAIB_CLASSID, "FlucaIB", "Immersed Boundary", "FlucaIB", comm, FlucaIBDestroy, FlucaIBView));
  i->dm          = NULL;
  i->setupcalled = PETSC_FALSE;
  i->data        = NULL;

  *ib = i;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBCreateNone(MPI_Comm comm, DM dm, FlucaIB *ib)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscAssertPointer(ib, 3);
  PetscCall(FlucaIBCreate(comm, ib));
  PetscCall(FlucaIBSetType(*ib, FLUCAIB_NONE));
  PetscCall(FlucaIBSetDM(*ib, dm));
  PetscCall(FlucaIBSetUp(*ib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBSetType(FlucaIB ib, FlucaIBType type)
{
  FlucaIBType old_type;
  PetscErrorCode (*impl_create)(FlucaIB);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);

  PetscCall(FlucaIBGetType(ib, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)ib, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(FlucaIBList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)ib), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown FlucaIB type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(ib, destroy);
    PetscCall(PetscMemzero(ib->ops, sizeof(struct _FlucaIBOps)));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)ib, type));
  PetscCall((*impl_create)(ib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBGetType(FlucaIB ib, FlucaIBType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(FlucaIBRegisterAll());
  *type = ((PetscObject)ib)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBSetUp(FlucaIB ib)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  if (ib->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(FLUCAIB_SetUp, (PetscObject)ib, 0, 0, 0));

  /* Default to FLUCAIB_NONE if no type was set explicitly. */
  if (!((PetscObject)ib)->type_name) PetscCall(FlucaIBSetType(ib, FLUCAIB_NONE));
  PetscCheck(ib->dm, PetscObjectComm((PetscObject)ib), PETSC_ERR_ARG_WRONGSTATE, "DM not set. Call FlucaIBSetDM() first");

  PetscTryTypeMethod(ib, setup);

  PetscCall(PetscLogEventEnd(FLUCAIB_SetUp, (PetscObject)ib, 0, 0, 0));

  ib->setupcalled = PETSC_TRUE;

  PetscCall(FlucaIBViewFromOptions(ib, NULL, "-ib_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBDestroy(FlucaIB *ib)
{
  PetscFunctionBegin;
  if (!*ib) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*ib), FLUCAIB_CLASSID, 1);

  if (--((PetscObject)(*ib))->refct > 0) {
    *ib = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod((*ib), destroy);

  PetscCall(PetscHeaderDestroy(ib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBView(FlucaIB ib, PetscViewer viewer)
{
  PetscBool   isascii;
  FlucaIBType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ib), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(ib, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ib, viewer));
    PetscCall(FlucaIBGetType(ib, &type));
    if (type) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Setup called: %s\n", ib->setupcalled ? "yes" : "no"));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }

  PetscTryTypeMethod(ib, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBViewFromOptions(FlucaIB ib, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)ib, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBSetDM(FlucaIB ib, DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCheckSameComm(ib, 1, dm, 2);
  ib->dm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBGetDM(FlucaIB ib, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  *dm = ib->dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBApplyBC(FlucaIB ib, Vec v, FlucaIBField field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscCheck(ib->setupcalled, PetscObjectComm((PetscObject)ib), PETSC_ERR_ARG_WRONGSTATE, "FlucaIBSetUp() must be called before FlucaIBApplyBC()");
  PetscCall(PetscLogEventBegin(FLUCAIB_ApplyBC, (PetscObject)ib, 0, 0, 0));
  PetscTryTypeMethod(ib, applybc, v, field);
  PetscCall(PetscLogEventEnd(FLUCAIB_ApplyBC, (PetscObject)ib, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
