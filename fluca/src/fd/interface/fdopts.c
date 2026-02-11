#include <fluca/private/flucafdimpl.h>

PetscErrorCode FlucaFDSetDM(FlucaFD fd, DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 2, DMSTAG);
  PetscCheckSameComm(fd, 1, dm, 2);
  PetscCall(DMDestroy(&fd->dm));
  fd->dm = dm;
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetInputLocation(FlucaFD fd, DMStagStencilLocation loc, PetscInt c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscCall(FlucaFDValidateStencilLocation_Internal(loc));
  fd->input_loc = loc;
  fd->input_c   = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetOutputLocation(FlucaFD fd, DMStagStencilLocation loc, PetscInt c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscCall(FlucaFDValidateStencilLocation_Internal(loc));
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

  PetscCheck(fd->dm, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Reference DM must be set before setting boundary conditions");
  PetscCall(DMGetDimension(fd->dm, &dim));
  nb = 2 * dim;
  PetscCall(PetscArraycpy(fd->bcs, bcs, nb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetBoundaryConditions(FlucaFD fd, FlucaFDBoundaryCondition bcs[])
{
  PetscInt dim, nb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscAssertPointer(bcs, 2);

  PetscCheck(fd->dm, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Reference DM must be set before getting boundary conditions");
  PetscCall(DMGetDimension(fd->dm, &dim));
  nb = 2 * dim;
  PetscCall(PetscArraycpy(bcs, fd->bcs, nb));
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
  PetscCall(PetscOptionsEnum("-flucafd_input_loc", "Input stencil location", "FlucaFDSetInputLocation", DMStagStencilLocations, (PetscEnum)fd->input_loc, (PetscEnum *)&fd->input_loc, NULL));
  PetscCall(PetscOptionsInt("-flucafd_input_c", "Input component", "FlucaFDSetInputLocation", fd->input_c, &fd->input_c, NULL));
  PetscCall(PetscOptionsEnum("-flucafd_output_loc", "Output stencil location", "FlucaFDSetOutputLocation", DMStagStencilLocations, (PetscEnum)fd->output_loc, (PetscEnum *)&fd->output_loc, NULL));
  PetscCall(PetscOptionsInt("-flucafd_output_c", "Output component", "FlucaFDSetOutputLocation", fd->output_c, &fd->output_c, NULL));
  PetscCall(DMGetDimension(fd->dm, &dim));
  for (d = 0; d < 2 * dim; ++d) {
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-flucafd_%s_bc_type", boundary_names[d]));
    PetscCall(PetscSNPrintf(text, PETSC_MAX_PATH_LEN, "Boundary condition type on the %s boundary", boundary_names[d]));
    PetscCall(PetscOptionsEnum(opt, text, "FlucaFDSetBoundaryConditions", FlucaFDBoundaryConditionTypes, (PetscEnum)fd->bcs[d].type, (PetscEnum *)&fd->bcs[d].type, NULL));
  }
  PetscTryTypeMethod(fd, setfromoptions, PetscOptionsObject);
  /* Process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)fd, PetscOptionsObject));
  PetscOptionsEnd();

  /* Validate stencil locations after options parsing */
  PetscCall(FlucaFDValidateStencilLocation_Internal(fd->input_loc));
  PetscCall(FlucaFDValidateStencilLocation_Internal(fd->output_loc));
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
