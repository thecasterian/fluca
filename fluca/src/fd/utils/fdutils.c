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

PetscErrorCode FlucaFDRemoveZeroStencilPoints_Internal(PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscScalar       v_abssum;
  PetscInt          ncols_new, c;
  PetscBool         remove;
  const PetscScalar atol = 1e-10, rtol = 1e-8;

  PetscFunctionBegin;
  v_abssum = 0.;
  for (c = 0; c < *ncols; ++c) v_abssum += PetscAbs(v[c]);
  ncols_new = 0;
  for (c = 0; c < *ncols; ++c) {
    remove = PetscAbs(v[c]) < atol || PetscAbs(v[c] / v_abssum) < rtol;
    if (!remove) {
      col[ncols_new] = col[c];
      v[ncols_new]   = v[c];
      ++ncols_new;
    }
  }
  *ncols = ncols_new;
  PetscFunctionReturn(PETSC_SUCCESS);
}
