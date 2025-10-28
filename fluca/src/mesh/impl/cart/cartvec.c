#include <fluca/private/meshcartimpl.h>
#include <flucaviewer.h>

PetscErrorCode VecView_Cart(Vec v, PetscViewer viewer)
{
  DM        dm;
  PetscBool iscgns;

  PetscFunctionBegin;
  PetscCall(VecGetDM(v, &dm));
  PetscCheck(dm, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a Mesh");
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) {
    Vec         locv;
    const char *name;

    PetscCall(DMGetLocalVector(dm, &locv));
    PetscCall(PetscObjectGetName((PetscObject)v, &name));
    PetscCall(PetscObjectSetName((PetscObject)locv, name));
    PetscCall(DMGlobalToLocal(dm, v, INSERT_VALUES, locv));
    PetscCall(VecView_Cart_Local(v, viewer));
    PetscCall(DMRestoreLocalVector(dm, &locv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Cart_Local(Vec v, PetscViewer viewer)
{
  PetscBool iscgns;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) PetscCall(VecView_Cart_Local_CGNS(v, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
