#include <fluca/private/flucafdimpl.h>

static PetscErrorCode FlucaFDSetUp_Composition(FlucaFD fd)
{
  FlucaFD_Composition *comp = (FlucaFD_Composition *)fd->data;

  PetscFunctionBegin;
  PetscCheck(comp->inner, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Inner operator not set");
  PetscCheck(comp->outer, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Outer operator not set");
  PetscCheck(comp->inner->output_c == comp->outer->input_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Inner output component (%" PetscInt_FMT ") must match outer input component (%" PetscInt_FMT ")", comp->inner->output_c, comp->outer->input_c);
  PetscCheck(comp->inner->output_loc == comp->outer->input_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Inner output location must match outer input location");
  PetscCall(FlucaFDValidatePeriodicityMatch_Internal(fd, comp->inner));
  PetscCall(FlucaFDValidatePeriodicityMatch_Internal(fd, comp->outer));

  /*
    Multiply term by term:
    - deriv_order = sum of deriv_orders per direction
    - accu_order = min of accu_orders per direction
    - input_loc = inner's input_loc
    - input_c = inner's input_c
  */
  {
    FlucaFDTermLink inner_term, outer_term, new_term;
    PetscInt        d;
    PetscBool       found;

    for (outer_term = comp->outer->termlink; outer_term; outer_term = outer_term->next) {
      for (inner_term = comp->inner->termlink; inner_term; inner_term = inner_term->next) {
        PetscCall(FlucaFDTermLinkCreate_Internal(&new_term));
        for (d = 0; d < FLUCAFD_MAX_DIM; ++d) {
          if (inner_term->deriv_order[d] == -1) new_term->deriv_order[d] = outer_term->deriv_order[d];
          else if (outer_term->deriv_order[d] == -1) new_term->deriv_order[d] = inner_term->deriv_order[d];
          else new_term->deriv_order[d] = inner_term->deriv_order[d] + outer_term->deriv_order[d];
          new_term->accu_order[d] = PetscMin(inner_term->accu_order[d], outer_term->accu_order[d]);
        }
        new_term->input_loc = inner_term->input_loc;
        new_term->input_c   = inner_term->input_c;

        PetscCall(FlucaFDTermLinkFind_Internal(fd->termlink, new_term, &found));
        if (found) PetscCall(PetscFree(new_term));
        else PetscCall(FlucaFDTermLinkAppend_Internal(&fd->termlink, new_term));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_Composition(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  FlucaFD_Composition *comp = (FlucaFD_Composition *)fd->data;
  DMStagStencil        outer_col[FLUCAFD_MAX_STENCIL_SIZE];
  DMStagStencil        inner_col[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar          outer_v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar          inner_v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt             outer_ncols, inner_ncols;
  PetscInt             oc, ic;

  PetscFunctionBegin;
  PetscCall(FlucaFDGetStencilRaw(comp->outer, i, j, k, &outer_ncols, outer_col, outer_v));

  *ncols = 0;
  for (oc = 0; oc < outer_ncols; oc++) {
    if (outer_col[oc].c < 0) {
      /* Constant or boundary marker from outer; pass through directly */
      PetscCall(FlucaFDAddStencilPoint_Internal(outer_col[oc], outer_v[oc], ncols, col, v));
      continue;
    }
    PetscCall(FlucaFDGetStencilRaw(comp->inner, outer_col[oc].i, outer_col[oc].j, outer_col[oc].k, &inner_ncols, inner_col, inner_v));
    for (ic = 0; ic < inner_ncols; ic++) PetscCall(FlucaFDAddStencilPoint_Internal(inner_col[ic], outer_v[oc] * inner_v[ic], ncols, col, v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_Composition(FlucaFD fd)
{
  FlucaFD_Composition *comp = (FlucaFD_Composition *)fd->data;

  PetscFunctionBegin;
  PetscCall(FlucaFDDestroy(&comp->inner));
  PetscCall(FlucaFDDestroy(&comp->outer));
  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_Composition(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_Composition *comp = (FlucaFD_Composition *)fd->data;
  PetscBool            isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Inner operator:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(FlucaFDView(comp->inner, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));

    PetscCall(PetscViewerASCIIPrintf(viewer, "  Outer operator:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(FlucaFDView(comp->outer, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_Composition(FlucaFD fd)
{
  FlucaFD_Composition *comp;

  PetscFunctionBegin;
  PetscCall(PetscNew(&comp));
  comp->inner = NULL;
  comp->outer = NULL;

  fd->data               = (void *)comp;
  fd->ops->setup         = FlucaFDSetUp_Composition;
  fd->ops->getstencilraw = FlucaFDGetStencilRaw_Composition;
  fd->ops->destroy       = FlucaFDDestroy_Composition;
  fd->ops->view          = FlucaFDView_Composition;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCompositionCreate(FlucaFD inner, FlucaFD outer, FlucaFD *fd)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(inner, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecific(outer, FLUCAFD_CLASSID, 2);
  PetscCheckSameComm(inner, 1, outer, 2);
  PetscCheck(inner->setupcalled, PetscObjectComm((PetscObject)inner), PETSC_ERR_ARG_WRONGSTATE, "Inner operand must be set up before calling FlucaFDCompositionCreate");
  PetscCheck(outer->setupcalled, PetscObjectComm((PetscObject)outer), PETSC_ERR_ARG_WRONGSTATE, "Outer operand must be set up before calling FlucaFDCompositionCreate");
  PetscAssertPointer(fd, 3);

  PetscCall(PetscObjectGetComm((PetscObject)inner, &comm));
  PetscCall(FlucaFDCreate(comm, fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDCOMPOSITION));
  PetscCall(FlucaFDSetCoordinateDM(*fd, inner->cdm));
  PetscCall(FlucaFDSetInputLocation(*fd, inner->input_loc, inner->input_c));
  PetscCall(FlucaFDSetOutputLocation(*fd, outer->output_loc, outer->output_c));
  PetscCall(FlucaFDCompositionSetOperands(*fd, inner, outer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCompositionSetOperands(FlucaFD fd, FlucaFD inner, FlucaFD outer)
{
  FlucaFD_Composition *comp = (FlucaFD_Composition *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDCOMPOSITION);
  PetscValidHeaderSpecific(inner, FLUCAFD_CLASSID, 2);
  PetscValidHeaderSpecific(outer, FLUCAFD_CLASSID, 3);
  PetscCheck(fd->data, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FlucaFD type not set");
  comp->inner = inner;
  comp->outer = outer;
  PetscCall(PetscObjectReference((PetscObject)inner));
  PetscCall(PetscObjectReference((PetscObject)outer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
