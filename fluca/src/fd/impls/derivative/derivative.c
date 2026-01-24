#include <fluca/private/flucafdimpl.h>

static PetscErrorCode FlucaFDSetFromOptions_Derivative(FlucaFD fd, PetscOptionItems PetscOptionsObject)
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "FlucaFDDerivative Options");
  PetscCall(PetscOptionsEnum("-flucafd_dir", "Direction of derivative", "FlucaFDDerivativeSetDirection", FlucaFDDirections, (PetscEnum)deriv->dir, (PetscEnum *)&deriv->dir, NULL));
  PetscCall(PetscOptionsBoundedInt("-flucafd_deriv_order", "Order of derivative", "FlucaFDDerivativeSetDerivativeOrder", deriv->deriv_order, &deriv->deriv_order, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-flucafd_accu_order", "Order of accuracy", "FlucaFDDerivativeSetAccuracyOrder", deriv->accu_order, &deriv->accu_order, NULL, 1));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetUp_Derivative(FlucaFD fd)
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;
  PetscBool           input_use_face, output_use_face, valid_transition;

  PetscFunctionBegin;
  PetscCheck((PetscInt)deriv->dir < fd->dim, PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Cannot compute derivative in %s direction on %" PetscInt_FMT "D DM", FlucaFDDirections[deriv->dir], fd->dim);
  /* Validate input and output locations: they must either be same, or differ only in the derivative direction's face component */
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->input_loc, deriv->dir, &input_use_face));
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, deriv->dir, &output_use_face));
  valid_transition = (input_use_face != output_use_face);
  for (PetscInt d = 0; d < fd->dim; ++d)
    if (d != (PetscInt)deriv->dir) {
      PetscBool input_face_d, output_face_d;

      /* All other directions must have matching face components */
      PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->input_loc, d, &input_face_d));
      PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, d, &output_face_d));
      if (input_face_d != output_face_d) valid_transition = PETSC_FALSE;
    }
  PetscCheck(fd->input_loc == fd->output_loc || valid_transition, PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Cannot compute derivative in %s direction from input location %s to output location %s", FlucaFDDirections[deriv->dir], DMStagStencilLocations[fd->input_loc], DMStagStencilLocations[fd->output_loc]);

  /* Pre-compute stencils */
  {
    const PetscScalar **arr_coord;
    PetscBool           periodic;
    PetscInt            xg, ng, extrag, stencil_size, offset_start;
    PetscInt            input_slot_coord, output_slot_coord, first_face_idx, last_face_idx;
    PetscScalar         h_prev, h_next;
    PetscInt            i, r, c, o;

    arr_coord = fd->arr_coord[deriv->dir];
    periodic  = fd->bcs[2 * deriv->dir].type == FLUCAFD_BC_PERIODIC;

    /* Local grid info */
    xg = fd->x[deriv->dir] - ((fd->is_first_rank[deriv->dir] && !periodic) ? 0 : fd->stencil_width);
    ng = fd->n[deriv->dir]                                                      //
       + ((fd->is_first_rank[deriv->dir] && !periodic) ? 0 : fd->stencil_width) //
       + ((fd->is_last_rank[deriv->dir] && !periodic) ? 0 : fd->stencil_width);
    extrag = (fd->is_last_rank[deriv->dir] && input_use_face && !periodic) ? 1 : 0;

    stencil_size = deriv->deriv_order + deriv->accu_order;
    PetscCheck(stencil_size <= FLUCAFD_MAX_STENCIL_SIZE, PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Required stencil size (%" PetscInt_FMT ") exceeds maximum (%" PetscInt_FMT ")", stencil_size, FLUCAFD_MAX_STENCIL_SIZE);

    /* Use central differencing scheme */
    offset_start = -(stencil_size - 1) / 2;
    if (!input_use_face && output_use_face) offset_start -= 1;

    deriv->ncols = stencil_size;
    for (c = 0; c < stencil_size; ++c) {
      deriv->col[c].i   = (deriv->dir == FLUCAFD_X) ? (offset_start + c) : 0;
      deriv->col[c].j   = (deriv->dir == FLUCAFD_Y) ? (offset_start + c) : 0;
      deriv->col[c].k   = (deriv->dir == FLUCAFD_Z) ? (offset_start + c) : 0;
      deriv->col[c].c   = fd->input_c;
      deriv->col[c].loc = fd->input_loc;
    }

    /* Allocate coefficient arrays */
    deriv->v_start = xg - (offset_start + stencil_size - 1);
    deriv->v_end   = xg + ng + extrag - offset_start;
    PetscCall(PetscMalloc1(deriv->v_end - deriv->v_start, &deriv->v));
    deriv->v -= deriv->v_start;

    /* Compute coefficients */
    input_slot_coord  = input_use_face ? fd->slot_coord_prev : fd->slot_coord_elem;
    output_slot_coord = output_use_face ? fd->slot_coord_prev : fd->slot_coord_elem;
    first_face_idx    = xg;
    last_face_idx     = xg + ng - ((fd->is_last_rank[deriv->dir] && !periodic) ? 0 : 1);
    h_prev            = arr_coord[first_face_idx + 1][fd->slot_coord_prev] - arr_coord[first_face_idx][fd->slot_coord_prev];
    h_next            = arr_coord[last_face_idx][fd->slot_coord_prev] - arr_coord[last_face_idx - 1][fd->slot_coord_prev];

    for (i = deriv->v_start - 1; i < deriv->v_end + 1; ++i) {
      PetscScalar *v;
      PetscScalar  input_coord, output_coord, h, factorial;
      PetscScalar  A[FLUCAFD_MAX_STENCIL_SIZE * FLUCAFD_MAX_STENCIL_SIZE];
      PetscScalar  b[FLUCAFD_MAX_STENCIL_SIZE];

      if (i < deriv->v_start) v = deriv->v_prev;
      else if (i >= deriv->v_end) v = deriv->v_next;
      else v = deriv->v[i];

      /* Build Vandermonde-like matrix using coordinates */
      PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, i, output_slot_coord, xg, ng, h_prev, h_next, &output_coord));
      for (c = 0; c < stencil_size; ++c) {
        PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, i + offset_start + c, input_slot_coord, xg, ng + extrag, h_prev, h_next, &input_coord));
        h = input_coord - output_coord;
        for (r = 0; r < stencil_size; ++r) A[r * stencil_size + c] = PetscPowScalarInt(h, r);
      }

      factorial = 1.;
      for (o = 1; o <= deriv->deriv_order; ++o) factorial *= o;
      for (c = 0; c < stencil_size; ++c) b[c] = (c == deriv->deriv_order) ? factorial : 0.;

      PetscCall(FlucaFDSolveLinearSystem_Internal(stencil_size, A, b, v));
    }
  }

  /* Create term info */
  {
    FlucaFDTermLink term;

    PetscCall(FlucaFDTermLinkCreate_Internal(&term));
    term->deriv_order[deriv->dir] = deriv->deriv_order;
    term->accu_order[deriv->dir]  = deriv->accu_order;
    term->input_loc               = fd->input_loc;
    term->input_c                 = fd->input_c;
    PetscCall(FlucaFDTermLinkAppend_Internal(&fd->termlink, term));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_Derivative(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;
  PetscInt            c, idx;

  PetscFunctionBegin;
  switch (deriv->dir) {
  case FLUCAFD_X:
    idx = i;
    break;
  case FLUCAFD_Y:
    idx = j;
    break;
  case FLUCAFD_Z:
    idx = k;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported direction");
  }

  *ncols = deriv->ncols;
  PetscCall(PetscArraycpy(col, deriv->col, *ncols));
  if (idx < deriv->v_start) PetscCall(PetscArraycpy(v, deriv->v_prev, deriv->ncols));
  else if (idx >= deriv->v_end) PetscCall(PetscArraycpy(v, deriv->v_next, deriv->ncols));
  else PetscCall(PetscArraycpy(v, deriv->v[idx], deriv->ncols));

  /* Add the actual position to relative indices */
  for (c = 0; c < *ncols; ++c) {
    col[c].i += i;
    col[c].j += j;
    col[c].k += k;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_Derivative(FlucaFD fd)
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;

  PetscFunctionBegin;
  deriv->v += deriv->v_start;
  PetscCall(PetscFree(deriv->v));
  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_Derivative(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;
  PetscBool           isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Direction: %s\n", FlucaFDDirections[deriv->dir]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Order of derivative: %" PetscInt_FMT "\n", deriv->deriv_order));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Order of accuracy: %" PetscInt_FMT "\n", deriv->accu_order));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_Derivative(FlucaFD fd)
{
  FlucaFD_Derivative *deriv;

  PetscFunctionBegin;
  PetscCall(PetscNew(&deriv));
  deriv->dir         = FLUCAFD_X;
  deriv->deriv_order = 1;
  deriv->accu_order  = 1;
  deriv->ncols       = 0;
  deriv->v_start     = 0;
  deriv->v_end       = 0;
  deriv->v           = NULL;

  fd->data                = (void *)deriv;
  fd->ops->setfromoptions = FlucaFDSetFromOptions_Derivative;
  fd->ops->setup          = FlucaFDSetUp_Derivative;
  fd->ops->getstencilraw  = FlucaFDGetStencilRaw_Derivative;
  fd->ops->destroy        = FlucaFDDestroy_Derivative;
  fd->ops->view           = FlucaFDView_Derivative;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDDerivativeSetDerivativeOrder(FlucaFD fd, PetscInt deriv_order)
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDDERIVATIVE);
  PetscCheck(deriv_order >= 0, PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Order of derivative must be non-negative");
  deriv->deriv_order = deriv_order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDDerivativeSetAccuracyOrder(FlucaFD fd, PetscInt accu_order)
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDDERIVATIVE);
  PetscCheck(accu_order > 0, PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Order of accuracy must be positive");
  deriv->accu_order = accu_order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDDerivativeSetDirection(FlucaFD fd, FlucaFDDirection dir)
{
  FlucaFD_Derivative *deriv = (FlucaFD_Derivative *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDDERIVATIVE);
  deriv->dir = dir;
  PetscFunctionReturn(PETSC_SUCCESS);
}
