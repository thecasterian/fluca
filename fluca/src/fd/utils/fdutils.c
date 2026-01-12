#include <fluca/private/flucafdimpl.h>

PetscErrorCode FlucaFDStencilLocationToDMStagStencilLocation_Internal(FlucaFDStencilLocation loc, DMStagStencilLocation *stag_loc)
{
  PetscFunctionBegin;
  switch (loc) {
  case FLUCAFD_ELEMENT:
    *stag_loc = DMSTAG_ELEMENT;
    break;
  case FLUCAFD_LEFT:
    *stag_loc = DMSTAG_LEFT;
    break;
  case FLUCAFD_DOWN:
    *stag_loc = DMSTAG_DOWN;
    break;
  case FLUCAFD_BACK:
    *stag_loc = DMSTAG_BACK;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported FlucaFDStencilLocation");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
