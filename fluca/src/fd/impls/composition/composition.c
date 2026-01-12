#include <fluca/private/flucafdimpl.h>

static PetscErrorCode FlucaFDGetStencil_Composition(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  FlucaFD_Composition *comp = (FlucaFD_Composition *)fd->data;
  DMStagStencil        outer_col[64], temp_col[64];
  PetscScalar          outer_v[64], temp_v[64];
  PetscInt             outer_ncols, temp_ncols;
  PetscInt             n, m, idx;
  PetscBool            found;

  PetscFunctionBegin;
  PetscCall(FlucaFDGetStencil(comp->outer, i, j, k, &outer_ncols, outer_col, outer_v));

  *ncols = 0;

  for (n = 0; n < outer_ncols; n++) {
    PetscCall(FlucaFDGetStencil(comp->inner, outer_col[n].i, outer_col[n].j, outer_col[n].k, &temp_ncols, temp_col, temp_v));

    for (m = 0; m < temp_ncols; m++) {
      found = PETSC_FALSE;
      for (idx = 0; idx < *ncols; idx++) {
        if (col[idx].i == temp_col[m].i && col[idx].j == temp_col[m].j && col[idx].k == temp_col[m].k && col[idx].c == temp_col[m].c && col[idx].loc == temp_col[m].loc) {
          v[idx] += outer_v[n] * temp_v[m];
          found = PETSC_TRUE;
          break;
        }
      }

      if (!found) {
        col[*ncols] = temp_col[m];
        v[*ncols]   = outer_v[n] * temp_v[m];
        (*ncols)++;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetUp_Composition(FlucaFD fd)
{
  FlucaFD_Composition *comp = (FlucaFD_Composition *)fd->data;

  PetscFunctionBegin;
  PetscCheck(comp->inner, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Inner operator not set");
  PetscCheck(comp->outer, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Outer operator not set");
  PetscCheck(comp->inner->output_c == comp->outer->input_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Inner output component (%" PetscInt_FMT ") must match outer input component (%" PetscInt_FMT ")", comp->inner->output_c, comp->outer->input_c);
  PetscCheck(comp->inner->output_loc == comp->outer->input_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Inner output location must match outer input location");
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

  fd->data            = (void *)comp;
  fd->ops->getstencil = FlucaFDGetStencil_Composition;
  fd->ops->setup      = FlucaFDSetUp_Composition;
  fd->ops->destroy    = FlucaFDDestroy_Composition;
  fd->ops->view       = FlucaFDView_Composition;
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
