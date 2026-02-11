#include <fluca/private/flucafdimpl.h>

static PetscErrorCode FlucaFDSetUp_Sum(FlucaFD fd)
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink op;

  PetscFunctionBegin;
  /* Validate that we have at least one operand */
  PetscCheck(sum->oplink, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "No operands set");
  /* Set up all operands and validate compatibility */
  PetscCheck(fd->input_loc == fd->output_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change location");
  PetscCheck(fd->input_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change component");
  for (op = sum->oplink; op != NULL; op = op->next) {
    PetscCheck(op->fd->output_loc == fd->output_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "All operands must have the same output stencil location");
    PetscCheck(op->fd->output_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "All operands must have the same output component");
    PetscCall(FlucaFDValidatePeriodicityMatch_Internal(fd, op->fd));
  }

  /* Concatenate terms */
  for (op = sum->oplink; op != NULL; op = op->next) {
    FlucaFDTermLink src, dst;
    PetscBool       found;

    for (src = op->fd->termlink; src; src = src->next) {
      PetscCall(FlucaFDTermLinkFind_Internal(fd->termlink, src, &found));
      if (!found) {
        PetscCall(FlucaFDTermLinkDuplicate_Internal(src, &dst));
        PetscCall(FlucaFDTermLinkAppend_Internal(&fd->termlink, dst));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_Sum(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink op;
  PetscInt              op_ncols;
  DMStagStencil         op_col[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar           op_v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt              c;

  PetscFunctionBegin;
  *ncols = 0;

  for (op = sum->oplink; op != NULL; op = op->next) {
    PetscCall(FlucaFDGetStencilRaw(op->fd, i, j, k, &op_ncols, op_col, op_v));
    for (c = 0; c < op_ncols; c++) PetscCall(FlucaFDAddStencilPoint_Internal(op_col[c], op_v[c], ncols, col, v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_Sum(FlucaFD fd)
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink op, next;

  PetscFunctionBegin;
  op = sum->oplink;
  while (op != NULL) {
    next = op->next;
    PetscCall(FlucaFDDestroy(&op->fd));
    PetscCall(PetscFree(op));
    op = next;
  }
  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_Sum(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  PetscBool             isascii;
  FlucaFDSumOperandLink op;
  PetscInt              numops, idx;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(FlucaFDSumGetNumOperands(fd, &numops));

  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Number of operands: %" PetscInt_FMT "\n", numops));

    for (op = sum->oplink, idx = 0; op != NULL; op = op->next, ++idx) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Operand %" PetscInt_FMT ":\n", idx));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(FlucaFDView(op->fd, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_Sum(FlucaFD fd)
{
  FlucaFD_Sum *sum;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sum));
  sum->oplink = NULL;

  fd->data               = (void *)sum;
  fd->ops->setup         = FlucaFDSetUp_Sum;
  fd->ops->getstencilraw = FlucaFDGetStencilRaw_Sum;
  fd->ops->destroy       = FlucaFDDestroy_Sum;
  fd->ops->view          = FlucaFDView_Sum;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSumCreate(PetscInt n, const FlucaFD ops[], FlucaFD *fd)
{
  MPI_Comm comm;
  PetscInt i;

  PetscFunctionBegin;
  PetscCheck(n > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of operands must be positive, got %" PetscInt_FMT, n);
  PetscAssertPointer(ops, 2);
  for (i = 0; i < n; ++i) {
    PetscValidHeaderSpecific(ops[i], FLUCAFD_CLASSID, 2);
    PetscCheck(ops[i]->setupcalled, PetscObjectComm((PetscObject)ops[i]), PETSC_ERR_ARG_WRONGSTATE, "Operand %" PetscInt_FMT " must be set up before calling FlucaFDSumCreate", i);
  }
  PetscAssertPointer(fd, 3);

  PetscCall(PetscObjectGetComm((PetscObject)ops[0], &comm));
  PetscCall(FlucaFDCreate(comm, fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDSUM));
  PetscCall(FlucaFDSetCoordinateDM(*fd, ops[0]->cdm));
  PetscCall(FlucaFDSetInputLocation(*fd, ops[0]->output_loc, ops[0]->output_c));
  PetscCall(FlucaFDSetOutputLocation(*fd, ops[0]->output_loc, ops[0]->output_c));
  for (i = 0; i < n; ++i) PetscCall(FlucaFDSumAddOperand(*fd, ops[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSumGetNumOperands(FlucaFD fd, PetscInt *n)
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSUM);
  PetscAssertPointer(n, 2);
  *n = 0;
  for (link = sum->oplink; link; link = link->next) ++(*n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSumAddOperand(FlucaFD fd, FlucaFD operand)
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink newlink, lastlink;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSUM);
  PetscValidHeaderSpecific(operand, FLUCAFD_CLASSID, 2);
  PetscCheckSameComm(fd, 1, operand, 2);

  PetscCall(PetscNew(&newlink));
  newlink->fd   = operand;
  newlink->next = NULL;
  PetscCall(PetscObjectReference((PetscObject)operand));

  if (!sum->oplink) {
    sum->oplink = newlink;
  } else {
    lastlink = sum->oplink;
    while (lastlink->next) lastlink = lastlink->next;
    lastlink->next = newlink;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
