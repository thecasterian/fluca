#include <fluca/private/flucafdimpl.h>
#include <petscdmda.h>

static PetscErrorCode FlucaFDSetFromOptions_Scale(FlucaFD fd, PetscOptionItems PetscOptionsObject)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "FlucaFDScale Options");
  if (scale->is_constant) {
    PetscCall(PetscOptionsScalar("-flucafd_constant", "Scale constant", "FlucaFDScaleSetConstant", scale->constant, &scale->constant, NULL));
  } else {
    PetscCall(PetscOptionsEnum("-flucafd_vec_loc", "Scale vector location", "FlucaFDScaleSetVector", DMStagStencilLocations, (PetscEnum)scale->vec_loc, (PetscEnum *)&scale->vec_loc, NULL));
    PetscCall(PetscOptionsInt("-flucafd_vec_c", "Scale vector component", "FlucaFDScaleSetVector", scale->vec_c, &scale->vec_c, NULL));
    PetscCall(FlucaFDValidateStencilLocation_Internal(scale->vec_loc));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetUp_Scale(FlucaFD fd)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscCheck(scale->operand, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Operand not set");
  PetscCall(FlucaFDValidateOperand_Internal(fd, scale->operand));
  PetscCheck(scale->operand->output_c == fd->input_c && fd->input_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change component");
  PetscCheck(scale->operand->output_loc == fd->input_loc && fd->input_loc == fd->output_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change location");

  if (!scale->is_constant) {
    DM vec_dm;

    PetscCheck(scale->vec, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Neither constant nor vector scale specified");
    PetscCheck(scale->operand->output_loc == scale->vec_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Operand and vector must have the same location");

    PetscCall(VecGetDM(scale->vec, &vec_dm));
    PetscCheckTypeName(vec_dm, DMSTAG);
    scale->vec_dm = vec_dm;
    PetscCall(PetscObjectReference((PetscObject)scale->vec_dm));
    PetscCall(FlucaFDCreateDMStagToDAScatter_Internal(scale->vec_dm, fd->dim, scale->vec_loc, scale->vec_c, scale->vec, &scale->vec_da, &scale->vec_local, &scale->vec_scatter));

    /* Get array views on local DMDA vector (kept until destroy) */
    switch (fd->dim) {
    case 1:
      PetscCall(DMDAVecGetArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_1d));
      break;
    case 2:
      PetscCall(DMDAVecGetArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_2d));
      break;
    case 3:
      PetscCall(DMDAVecGetArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }

    /* Scatter data */
    PetscCall(VecScatterBegin(scale->vec_scatter, scale->vec, scale->vec_local, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scale->vec_scatter, scale->vec, scale->vec_local, INSERT_VALUES, SCATTER_FORWARD));
  }

  /* Copy term infos from operand */
  {
    FlucaFDTermLink src, dst;

    for (src = scale->operand->termlink; src; src = src->next) {
      PetscCall(FlucaFDTermLinkDuplicate_Internal(src, &dst));
      PetscCall(FlucaFDTermLinkAppend_Internal(&fd->termlink, dst));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_Scale(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;
  PetscInt       n;
  PetscScalar    scale_value;

  PetscFunctionBegin;
  PetscCall(FlucaFDGetStencilRaw(scale->operand, i, j, k, ncols, col, v));

  if (scale->is_constant) {
    scale_value = scale->constant;
  } else {
    switch (fd->dim) {
    case 1:
      scale_value = scale->arr_vec_1d[i];
      break;
    case 2:
      scale_value = scale->arr_vec_2d[j][i];
      break;
    case 3:
      scale_value = scale->arr_vec_3d[k][j][i];
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
  }

  for (n = 0; n < *ncols; n++) v[n] *= scale_value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_Scale(FlucaFD fd)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscCall(FlucaFDDestroy(&scale->operand));
  if (scale->vec_dm) {
    switch (fd->dim) {
    case 1:
      PetscCall(DMDAVecRestoreArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_1d));
      break;
    case 2:
      PetscCall(DMDAVecRestoreArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_2d));
      break;
    case 3:
      PetscCall(DMDAVecRestoreArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_3d));
      break;
    default:
      break;
    }
    PetscCall(VecScatterDestroy(&scale->vec_scatter));
    PetscCall(VecDestroy(&scale->vec_local));
    PetscCall(DMDestroy(&scale->vec_da));
    PetscCall(DMDestroy(&scale->vec_dm));
  }
  PetscCall(VecDestroy(&scale->vec));
  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_Scale(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (scale->is_constant) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Scale type: constant\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Scale value: %g\n", (double)PetscRealPart(scale->constant)));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Scale type: vector\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Vector stencil location: %s\n", DMStagStencilLocations[scale->vec_loc]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Vector component: %" PetscInt_FMT "\n", scale->vec_c));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Operand:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(FlucaFDView(scale->operand, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_Scale(FlucaFD fd)
{
  FlucaFD_Scale *scale;

  PetscFunctionBegin;
  PetscCall(PetscNew(&scale));
  scale->operand     = NULL;
  scale->constant    = 1.;
  scale->vec         = NULL;
  scale->vec_c       = 0;
  scale->vec_loc     = DMSTAG_ELEMENT;
  scale->is_constant = PETSC_TRUE;
  scale->vec_dm      = NULL;
  scale->vec_da      = NULL;
  scale->vec_local   = NULL;
  scale->vec_scatter = NULL;
  scale->arr_vec_1d  = NULL;
  scale->arr_vec_2d  = NULL;
  scale->arr_vec_3d  = NULL;

  fd->data                = (void *)scale;
  fd->ops->setfromoptions = FlucaFDSetFromOptions_Scale;
  fd->ops->setup          = FlucaFDSetUp_Scale;
  fd->ops->getstencilraw  = FlucaFDGetStencilRaw_Scale;
  fd->ops->destroy        = FlucaFDDestroy_Scale;
  fd->ops->view           = FlucaFDView_Scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDScaleCreateConstant(FlucaFD operand, PetscScalar constant, FlucaFD *fd)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(operand, FLUCAFD_CLASSID, 1);
  PetscCheck(operand->setupcalled, PetscObjectComm((PetscObject)operand), PETSC_ERR_ARG_WRONGSTATE, "Operand must be set up before calling FlucaFDScaleCreateConstant");
  PetscAssertPointer(fd, 3);

  PetscCall(PetscObjectGetComm((PetscObject)operand, &comm));
  PetscCall(FlucaFDCreate(comm, fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDSCALE));
  PetscCall(FlucaFDSetDM(*fd, operand->dm));
  PetscCall(FlucaFDSetInputLocation(*fd, operand->output_loc, operand->output_c));
  PetscCall(FlucaFDSetOutputLocation(*fd, operand->output_loc, operand->output_c));
  PetscCall(FlucaFDScaleSetOperand(*fd, operand));
  PetscCall(FlucaFDScaleSetConstant(*fd, constant));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDScaleCreateVector(FlucaFD operand, Vec vec, PetscInt vec_c, FlucaFD *fd)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(operand, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscCheckSameComm(operand, 1, vec, 2);
  PetscCheck(operand->setupcalled, PetscObjectComm((PetscObject)operand), PETSC_ERR_ARG_WRONGSTATE, "Operand must be set up before calling FlucaFDScaleCreateVector");
  PetscAssertPointer(fd, 4);

  PetscCall(PetscObjectGetComm((PetscObject)operand, &comm));
  PetscCall(FlucaFDCreate(comm, fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDSCALE));
  PetscCall(FlucaFDSetDM(*fd, operand->dm));
  PetscCall(FlucaFDSetInputLocation(*fd, operand->output_loc, operand->output_c));
  PetscCall(FlucaFDSetOutputLocation(*fd, operand->output_loc, operand->output_c));
  PetscCall(FlucaFDScaleSetOperand(*fd, operand));
  PetscCall(FlucaFDScaleSetVector(*fd, vec, operand->output_loc, vec_c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDScaleSetOperand(FlucaFD fd, FlucaFD operand)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSCALE);
  PetscValidHeaderSpecific(operand, FLUCAFD_CLASSID, 2);
  PetscCheckSameComm(fd, 1, operand, 2);
  scale->operand = operand;
  PetscCall(PetscObjectReference((PetscObject)operand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDScaleSetConstant(FlucaFD fd, PetscScalar constant)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSCALE);
  PetscCall(VecDestroy(&scale->vec));
  scale->constant    = constant;
  scale->is_constant = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDScaleSetVector(FlucaFD fd, Vec vec, DMStagStencilLocation loc, PetscInt c)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;
  DM             vec_dm;
  PetscBool      isstag;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSCALE);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscCall(FlucaFDValidateStencilLocation_Internal(loc));
  PetscCall(VecGetDM(vec, &vec_dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)vec_dm, DMSTAG, &isstag));
  PetscCheck(isstag, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Vector is not on DMStag");

  /* Recreate scatter if DM, location, or component changed */
  if (fd->setupcalled && (vec_dm != scale->vec_dm || loc != scale->vec_loc || c != scale->vec_c)) {
    if (scale->vec_dm) {
      switch (fd->dim) {
      case 1:
        PetscCall(DMDAVecRestoreArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_1d));
        break;
      case 2:
        PetscCall(DMDAVecRestoreArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_2d));
        break;
      case 3:
        PetscCall(DMDAVecRestoreArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_3d));
        break;
      default:
        break;
      }
    }
    PetscCall(VecScatterDestroy(&scale->vec_scatter));
    PetscCall(VecDestroy(&scale->vec_local));
    PetscCall(DMDestroy(&scale->vec_da));
    PetscCall(DMDestroy(&scale->vec_dm));
  }

  PetscCall(VecDestroy(&scale->vec));
  scale->vec         = vec;
  scale->vec_loc     = loc;
  scale->vec_c       = c;
  scale->is_constant = PETSC_FALSE;
  PetscCall(PetscObjectReference((PetscObject)vec));

  /* If not yet set up, defer infrastructure creation to SetUp */
  if (!fd->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  /* Create DMDA infrastructure if needed */
  if (!scale->vec_dm) {
    scale->vec_dm = vec_dm;
    PetscCall(PetscObjectReference((PetscObject)scale->vec_dm));
    PetscCall(FlucaFDCreateDMStagToDAScatter_Internal(scale->vec_dm, fd->dim, scale->vec_loc, scale->vec_c, vec, &scale->vec_da, &scale->vec_local, &scale->vec_scatter));

    /* Get array views on local DMDA vector (kept until destroy) */
    switch (fd->dim) {
    case 1:
      PetscCall(DMDAVecGetArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_1d));
      break;
    case 2:
      PetscCall(DMDAVecGetArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_2d));
      break;
    case 3:
      PetscCall(DMDAVecGetArrayRead(scale->vec_da, scale->vec_local, &scale->arr_vec_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
  }

  /* Scatter data */
  PetscCall(VecScatterBegin(scale->vec_scatter, vec, scale->vec_local, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scale->vec_scatter, vec, scale->vec_local, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}
