#include <fluca/private/flucafdimpl.h>

PetscErrorCode FlucaFDSetCoordinateDM(FlucaFD fd, DM cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecific(cdm, DM_CLASSID, 2);
  PetscCheckSameComm(fd, 1, cdm, 2);
  PetscCall(DMDestroy(&fd->cdm));
  fd->cdm = cdm;
  PetscCall(PetscObjectReference((PetscObject)cdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetInputLocation(FlucaFD fd, FlucaFDStencilLocation loc, PetscInt c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  fd->input_loc = loc;
  fd->input_c   = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetOutputLocation(FlucaFD fd, FlucaFDStencilLocation loc, PetscInt c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  fd->output_loc = loc;
  fd->output_c   = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetBoundaryConditions(FlucaFD fd, const FlucaFDBoundaryCondition bcs[])
{
  PetscInt dim, nb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscAssertPointer(bcs, 2);

  PetscCheck(fd->cdm, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Coordinate DM must be set before boundary conditions");
  PetscCall(DMGetDimension(fd->cdm, &dim));
  nb = 2 * dim;
  PetscCall(PetscArraycpy(fd->bcs, bcs, nb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetFromOptions(FlucaFD fd)
{
  const char       *default_type;
  char              type[256];
  char              opt[PETSC_MAX_OPTION_NAME];
  char              text[PETSC_MAX_PATH_LEN];
  PetscBool         flg;
  PetscInt          dim, d;
  const char *const boundary_names[] = {"left", "right", "down", "up", "back", "front"};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  if (!((PetscObject)fd)->type_name) default_type = FLUCAFDDERIVATIVE;
  else default_type = ((PetscObject)fd)->type_name;
  PetscCall(FlucaFDRegisterAll());

  PetscObjectOptionsBegin((PetscObject)fd);
  PetscCall(PetscOptionsFList("-flucafd_type", "Finite difference discretization", "FlucaFDSetType", FlucaFDList, default_type, type, sizeof(type), &flg));
  if (flg) PetscCall(FlucaFDSetType(fd, type));
  else if (!((PetscObject)fd)->type_name) PetscCall(FlucaFDSetType(fd, default_type));
  PetscCall(PetscOptionsEnum("-flucafd_input_loc", "Input stencil location", "FlucaFDSetInputLocation", FlucaFDStencilLocations, (PetscEnum)fd->input_loc, (PetscEnum *)&fd->input_loc, NULL));
  PetscCall(PetscOptionsInt("-flucafd_input_c", "Input component", "FlucaFDSetInputLocation", fd->input_c, &fd->input_c, NULL));
  PetscCall(PetscOptionsEnum("-flucafd_output_loc", "Output stencil location", "FlucaFDSetOutputLocation", FlucaFDStencilLocations, (PetscEnum)fd->output_loc, (PetscEnum *)&fd->output_loc, NULL));
  PetscCall(PetscOptionsInt("-flucafd_output_c", "Output component", "FlucaFDSetOutputLocation", fd->output_c, &fd->output_c, NULL));
  PetscCall(DMGetDimension(fd->cdm, &dim));
  for (d = 0; d < 2 * dim; ++d) {
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-flucafd_%s_bc_type", boundary_names[d]));
    PetscCall(PetscSNPrintf(text, PETSC_MAX_PATH_LEN, "Boundary condition type on the %s boundary", boundary_names[d]));
    PetscCall(PetscOptionsEnum(opt, text, "FlucaFDSetBoundaryConditions", FlucaFDBoundaryConditionTypes, (PetscEnum)fd->bcs[d].type, (PetscEnum *)&fd->bcs[d].type, NULL));
  }
  PetscTryTypeMethod(fd, setfromoptions, PetscOptionsObject);
  /* Process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)fd, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetOptionsPrefix(FlucaFD fd, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fd, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDAppendOptionsPrefix(FlucaFD fd, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)fd, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetOptionsPrefix(FlucaFD fd, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)fd, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}
