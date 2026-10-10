#include <fluca/private/flucafdimpl.h>

static PetscErrorCode FlucaFDSetUp_IBMask(FlucaFD fd)
{
  FlucaFD_IBMask *mask = (FlucaFD_IBMask *)fd->data;
  FlucaFDTermLink src, dst;
  PetscInt        nb, s[3];
  MeshIBPointType type;

  PetscFunctionBegin;
  PetscCheck(mask->operand, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Operand not set");
  PetscCall(FlucaFDValidateOperand_Internal(fd, mask->operand));
  PetscCheck(mask->operand->input_loc == fd->input_loc && mask->operand->input_c == fd->input_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Mask input must match the operand input");
  PetscCheck(mask->operand->output_loc == fd->output_loc && mask->operand->output_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Mask output must match the operand output");
  /* Fail in setup, not at the first stencil query, when the output location is not classified */
  PetscCall(MeshGetNumBodies(fd->mesh, &nb));
  if (nb) {
    PetscCall(DMStagGetCorners(fd->dm, &s[0], &s[1], &s[2], NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, s[0], fd->dim > 1 ? s[1] : 0, fd->dim > 2 ? s[2] : 0, &type));
  }

  /* The mask is transparent at fluid points: same domain boundary conditions and terms as the operand */
  PetscCall(PetscArraycpy(&fd->bcs[0][0], &mask->operand->bcs[0][0], FLUCAFD_MAX_COMPONENT * 2 * FLUCAFD_MAX_DIM));
  for (src = mask->operand->termlink; src; src = src->next) {
    PetscCall(FlucaFDTermLinkDuplicate_Internal(src, &dst));
    PetscCall(FlucaFDTermLinkAppend_Internal(&fd->termlink, dst));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_IBMask(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *npoints, FlucaFDStencilPoint points[])
{
  FlucaFD_IBMask *mask = (FlucaFD_IBMask *)fd->data;
  MeshIBPointType type;

  PetscFunctionBegin;
  PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, i, j, k, &type));
  if (type != MESH_IB_FLUID) {
    *npoints = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(FlucaFDGetStencilRaw(mask->operand, i, j, k, npoints, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_IBMask(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_IBMask *mask = (FlucaFD_IBMask *)fd->data;
  PetscBool       isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii && mask->operand) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Operand:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(FlucaFDView(mask->operand, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_IBMask(FlucaFD fd)
{
  FlucaFD_IBMask *mask = (FlucaFD_IBMask *)fd->data;

  PetscFunctionBegin;
  PetscCall(FlucaFDDestroy(&mask->operand));
  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_IBMask(FlucaFD fd)
{
  FlucaFD_IBMask *mask;

  PetscFunctionBegin;
  PetscCall(PetscNew(&mask));
  mask->operand = NULL;

  fd->data               = (void *)mask;
  fd->ops->setup         = FlucaFDSetUp_IBMask;
  fd->ops->getstencilraw = FlucaFDGetStencilRaw_IBMask;
  fd->ops->destroy       = FlucaFDDestroy_IBMask;
  fd->ops->view          = FlucaFDView_IBMask;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBMaskCreate(FlucaFD operand, FlucaFD *fd)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(operand, FLUCAFD_CLASSID, 1);
  PetscCheck(operand->setupcalled, PetscObjectComm((PetscObject)operand), PETSC_ERR_ARG_WRONGSTATE, "Operand must be set up before calling FlucaFDIBMaskCreate");
  PetscAssertPointer(fd, 2);

  PetscCall(PetscObjectGetComm((PetscObject)operand, &comm));
  PetscCall(FlucaFDCreate(comm, fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDIBMASK));
  PetscCall(FlucaFDSetMesh(*fd, operand->mesh));
  PetscCall(FlucaFDSetInputLocation(*fd, operand->input_loc, operand->input_c));
  PetscCall(FlucaFDSetOutputLocation(*fd, operand->output_loc, operand->output_c));
  PetscCall(FlucaFDIBMaskSetOperand(*fd, operand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBMaskSetOperand(FlucaFD fd, FlucaFD operand)
{
  FlucaFD_IBMask *mask;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDIBMASK);
  PetscValidHeaderSpecific(operand, FLUCAFD_CLASSID, 2);
  PetscCheckSameComm(fd, 1, operand, 2);
  PetscCheck(!fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the operand after FlucaFDSetUp()");
  mask = (FlucaFD_IBMask *)fd->data;
  PetscCall(PetscObjectReference((PetscObject)operand));
  PetscCall(FlucaFDDestroy(&mask->operand));
  mask->operand = operand;
  PetscFunctionReturn(PETSC_SUCCESS);
}
