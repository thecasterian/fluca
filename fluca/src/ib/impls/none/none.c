#include <fluca/private/flucaibimpl.h>

/* FLUCAIB_NONE: trivial subtype representing "no immersed body".
 * All IB-specific operations are no-ops. */

static PetscErrorCode FlucaIBApplyBC_None(FlucaIB ib, Vec v, FlucaIBField field)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaIBView_None(FlucaIB ib, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Trivial pass-through (no immersed body)\n"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBCreate_None(FlucaIB ib)
{
  PetscFunctionBegin;
  ib->data = NULL;

  ib->ops->setfromoptions = NULL;
  ib->ops->setup          = NULL;
  ib->ops->destroy        = NULL;
  ib->ops->view           = FlucaIBView_None;
  ib->ops->applybc        = FlucaIBApplyBC_None;
  PetscFunctionReturn(PETSC_SUCCESS);
}
