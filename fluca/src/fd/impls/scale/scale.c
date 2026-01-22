#include <fluca/private/flucafdimpl.h>

static PetscErrorCode FlucaFDSetFromOptions_Scale(FlucaFD fd, PetscOptionItems PetscOptionsObject)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "FlucaFDDerivative Options");
  if (scale->is_constant) {
    PetscCall(PetscOptionsReal("-flucafd_constant", "Scale constant", "FlucaFDScaleSetConstant", scale->constant, &scale->constant, NULL));
  } else {
    PetscCall(PetscOptionsEnum("-flucafd_vec_loc", "Scale vector location", "FlucaFDScaleSetVector", FlucaFDStencilLocations, (PetscEnum)scale->vec_loc, (PetscEnum *)&scale->vec_loc, NULL));
    PetscCall(PetscOptionsInt("-flucafd_vec_c", "Scale vector component", "FlucaFDScaleSetVector", scale->vec_c, &scale->vec_c, NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetUp_Scale(FlucaFD fd)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscCheck(scale->operand, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Operand not set");
  PetscCheck(scale->operand->output_c == fd->input_c && fd->input_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change component");
  PetscCheck(scale->operand->output_loc == fd->input_loc && fd->input_loc == fd->output_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change location");

  if (!scale->is_constant) {
    DMStagStencilLocation stag_loc;

    PetscCheck(scale->vec, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Neither constant nor vector scale specified");
    PetscCheck(scale->operand->output_loc == scale->vec_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Operand and vector must have the same location");

    PetscCall(VecGetDM(scale->vec, &scale->vec_dm));
    PetscCheckTypeName(scale->vec_dm, DMSTAG);
    PetscCall(DMCreateLocalVector(scale->vec_dm, &scale->vec_local));
    PetscCall(DMGlobalToLocal(scale->vec_dm, scale->vec, INSERT_VALUES, scale->vec_local));
    switch (fd->dim) {
    case 1:
      PetscCall(DMStagVecGetArrayRead(scale->vec_dm, scale->vec_local, &scale->arr_vec_1d));
      break;
    case 2:
      PetscCall(DMStagVecGetArrayRead(scale->vec_dm, scale->vec_local, &scale->arr_vec_2d));
      break;
    case 3:
      PetscCall(DMStagVecGetArrayRead(scale->vec_dm, scale->vec_local, &scale->arr_vec_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
    PetscCall(FlucaFDStencilLocationToDMStagStencilLocation_Internal(scale->vec_loc, &stag_loc));
    PetscCall(DMStagGetLocationSlot(scale->vec_dm, stag_loc, scale->vec_c, &scale->vec_slot));
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
      scale_value = scale->arr_vec_1d[i][scale->vec_slot];
      break;
    case 2:
      scale_value = scale->arr_vec_2d[j][i][scale->vec_slot];
      break;
    case 3:
      scale_value = scale->arr_vec_3d[k][j][i][scale->vec_slot];
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
  if (!scale->is_constant) {
    switch (fd->dim) {
    case 1:
      PetscCall(DMStagVecRestoreArrayRead(scale->vec_dm, scale->vec_local, &scale->arr_vec_1d));
      break;
    case 2:
      PetscCall(DMStagVecRestoreArrayRead(scale->vec_dm, scale->vec_local, &scale->arr_vec_2d));
      break;
    case 3:
      PetscCall(DMStagVecRestoreArrayRead(scale->vec_dm, scale->vec_local, &scale->arr_vec_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
    PetscCall(VecDestroy(&scale->vec));
    PetscCall(VecDestroy(&scale->vec_local));
  }
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
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Scale value: %g\n", scale->constant));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Scale type: vector\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Vector stencil location: %s\n", FlucaFDStencilLocations[scale->vec_loc]));
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
  scale->vec_loc     = FLUCAFD_ELEMENT;
  scale->is_constant = PETSC_TRUE;
  scale->vec_dm      = NULL;
  scale->vec_local   = NULL;
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

PetscErrorCode FlucaFDScaleSetVector(FlucaFD fd, Vec vec, FlucaFDStencilLocation loc, PetscInt c)
{
  FlucaFD_Scale *scale = (FlucaFD_Scale *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSCALE);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscCall(VecDestroy(&scale->vec));
  scale->vec         = vec;
  scale->vec_loc     = loc;
  scale->vec_c       = c;
  scale->is_constant = PETSC_FALSE;
  PetscCall(PetscObjectReference((PetscObject)vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}
