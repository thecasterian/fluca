#include <fluca/private/flucafdimpl.h>

static PetscErrorCode FlucaFDGetStencil_Sum(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink op;
  PetscInt              temp_ncols;
  DMStagStencil         temp_col[64];
  PetscScalar           temp_v[64];
  PetscInt              n, idx;
  PetscBool             found;

  PetscFunctionBegin;
  *ncols = 0;

  for (op = sum->oplink; op != NULL; op = op->next) {
    PetscCall(FlucaFDGetStencil(op->fd, i, j, k, &temp_ncols, temp_col, temp_v));

    for (n = 0; n < temp_ncols; n++) {
      found = PETSC_FALSE;
      for (idx = 0; idx < *ncols; idx++) {
        if (col[idx].i == temp_col[n].i && col[idx].j == temp_col[n].j && col[idx].k == temp_col[n].k && col[idx].c == temp_col[n].c && col[idx].loc == temp_col[n].loc) {
          v[idx] += temp_v[n];
          found = PETSC_TRUE;
          break;
        }
      }

      if (!found) {
        col[*ncols] = temp_col[n];
        v[*ncols]   = temp_v[n];
        (*ncols)++;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetUp_Sum(FlucaFD fd)
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink op;

  PetscFunctionBegin;
  /* Validate that we have at least one operand */
  PetscCheck(sum->oplink, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "No operands set");
  /* Set up all operands and validate compatibility */
  for (op = sum->oplink; op != NULL; op = op->next) {
    PetscCheck(op->fd->output_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "All operands must have the same output component");
    PetscCheck(op->fd->output_loc == fd->output_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "All operands must have the same output stencil location");
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

static PetscErrorCode FlucaFDDestroy_Sum(FlucaFD fd)
{
  FlucaFD_Sum          *sum = (FlucaFD_Sum *)fd->data;
  FlucaFDSumOperandLink op, next;

  PetscFunctionBegin;
  op = sum->oplink;
  while (op != NULL) {
    next = op->next;
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

  fd->data            = (void *)sum;
  fd->ops->getstencil = FlucaFDGetStencil_Sum;
  fd->ops->setup      = FlucaFDSetUp_Sum;
  fd->ops->destroy    = FlucaFDDestroy_Sum;
  fd->ops->view       = FlucaFDView_Sum;
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

  if (!sum->oplink) {
    sum->oplink = newlink;
  } else {
    lastlink = sum->oplink;
    while (lastlink->next) lastlink = lastlink->next;
    lastlink->next = newlink;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
