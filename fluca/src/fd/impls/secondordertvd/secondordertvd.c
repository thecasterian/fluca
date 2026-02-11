#include <fluca/private/flucafdimpl.h>

FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterSuperbee_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterMinmod_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterMC_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterVanLeer_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterVanAlbada_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterBarthJesperson_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterVenkatakrishnan_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterKoren_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterUpwind_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterSOU_Internal(PetscScalar);
FLUCA_INTERN PetscScalar FlucaFDSecondOrderTVDLimiterQUICK_Internal(PetscScalar);

PetscFunctionList FlucaFDLimiterList              = NULL;
PetscBool         FlucaFDLimiterRegisterAllCalled = PETSC_FALSE;

PetscErrorCode FlucaFDLimiterRegisterAll(void)
{
  PetscFunctionBegin;
  if (FlucaFDLimiterRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  FlucaFDLimiterRegisterAllCalled = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "superbee", FlucaFDSecondOrderTVDLimiterSuperbee_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "minmod", FlucaFDSecondOrderTVDLimiterMinmod_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "mc", FlucaFDSecondOrderTVDLimiterMC_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "vanleer", FlucaFDSecondOrderTVDLimiterVanLeer_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "vanalbada", FlucaFDSecondOrderTVDLimiterVanAlbada_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "barthjesperson", FlucaFDSecondOrderTVDLimiterBarthJesperson_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "venkatakrishnan", FlucaFDSecondOrderTVDLimiterVenkatakrishnan_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "koren", FlucaFDSecondOrderTVDLimiterKoren_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "upwind", FlucaFDSecondOrderTVDLimiterUpwind_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "sou", FlucaFDSecondOrderTVDLimiterSOU_Internal));
  PetscCall(PetscFunctionListAdd(&FlucaFDLimiterList, "quick", FlucaFDSecondOrderTVDLimiterQUICK_Internal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetFromOptions_SecondOrderTVD(FlucaFD fd, PetscOptionItems PetscOptionsObject)
{
  FlucaFD_SecondOrderTVD *tvd        = (FlucaFD_SecondOrderTVD *)fd->data;
  char                    lname[256] = "superbee";
  PetscBool               flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "FlucaFDSecondOrderTVD Options");
  PetscCall(PetscOptionsEnum("-flucafd_dir", "Direction", "FlucaFDSecondOrderTVDSetDirection", FlucaFDDirections, (PetscEnum)tvd->dir, (PetscEnum *)&tvd->dir, NULL));
  PetscCall(PetscOptionsFList("-flucafd_limiter", "Flux limiter type", "FlucaFDSecondOrderTVDSetLimiter", FlucaFDLimiterList, lname, lname, sizeof(lname), &flg));
  if (flg) PetscCall(PetscFunctionListFind(FlucaFDLimiterList, lname, &tvd->limiter));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetUp_SecondOrderTVD(FlucaFD fd)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;
  DMStagStencilLocation   expected_output_loc;

  PetscFunctionBegin;
  /* Validate input/output locations */
  PetscCheck(fd->input_loc == DMSTAG_ELEMENT, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Input location must be DMSTAG_ELEMENT for TVD interpolation");

  switch (tvd->dir) {
  case FLUCAFD_X:
    expected_output_loc = DMSTAG_LEFT;
    break;
  case FLUCAFD_Y:
    expected_output_loc = DMSTAG_DOWN;
    break;
  case FLUCAFD_Z:
    expected_output_loc = DMSTAG_BACK;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Invalid direction");
  }
  PetscCheck(fd->output_loc == expected_output_loc, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Output location must match direction (LEFT for X, DOWN for Y, BACK for Z)");

  /* Create internal gradient operator for dphi/dx (element -> face) */
  PetscCall(FlucaFDDerivativeCreate(fd->cdm, tvd->dir, 1, 1, fd->input_loc, fd->input_c, fd->output_loc, 0, &tvd->fd_grad));
  PetscCall(FlucaFDSetBoundaryConditions(tvd->fd_grad, fd->bcs));
  PetscCall(FlucaFDSetUp(tvd->fd_grad));

  /*
    Pre-compute alpha coefficients for non-uniform grids
      alpha_plus[i]  = (x_{i-1/2} - x_{i-1}) / (x_i - x_{i-1})
      alpha_minus[i] = (x_i - x_{i-1/2}) / (x_i - x_{i-1})
    for face i-1/2
  */
  {
    const PetscScalar **arr_coord;
    PetscBool           periodic;
    PetscInt            xg, ng, extrag, i;

    arr_coord = fd->arr_coord[tvd->dir];
    periodic  = fd->bcs[2 * tvd->dir].type == FLUCAFD_BC_PERIODIC;

    /* Local grid info */
    xg = fd->x[tvd->dir] - ((fd->is_first_rank[tvd->dir] && !periodic) ? 0 : fd->stencil_width);
    ng = fd->n[tvd->dir]                                                      //
       + ((fd->is_first_rank[tvd->dir] && !periodic) ? 0 : fd->stencil_width) //
       + ((fd->is_last_rank[tvd->dir] && !periodic) ? 0 : fd->stencil_width);
    extrag = (fd->is_last_rank[tvd->dir] && !periodic) ? 1 : 0;

    tvd->alpha_start = xg;
    tvd->alpha_end   = xg + ng + extrag;
    PetscCall(PetscCalloc1(ng + extrag, &tvd->alpha_plus));
    PetscCall(PetscCalloc1(ng + extrag, &tvd->alpha_minus));
    tvd->alpha_plus_base  = tvd->alpha_plus;
    tvd->alpha_minus_base = tvd->alpha_minus;
    tvd->alpha_plus -= xg;
    tvd->alpha_minus -= xg;

    for (i = xg; i < xg + ng + extrag; i++) {
      if ((i == 0 || i == fd->N[tvd->dir]) && !periodic) {
        tvd->alpha_plus[i]  = 0.5;
        tvd->alpha_minus[i] = 0.5;
      } else {
        PetscScalar x_face, x_left, x_right, dx;

        x_face  = arr_coord[i][fd->slot_coord_prev];
        x_left  = arr_coord[i - 1][fd->slot_coord_elem];
        x_right = arr_coord[i][fd->slot_coord_elem];
        dx      = x_right - x_left;
        if (PetscAbsScalar(dx) > 1e-14) {
          tvd->alpha_plus[i]  = (x_face - x_left) / dx;
          tvd->alpha_minus[i] = (x_right - x_face) / dx;
        } else {
          tvd->alpha_plus[i]  = 0.5;
          tvd->alpha_minus[i] = 0.5;
        }
      }
    }
  }

  /* Create term info: interpolation (deriv_order=0) with second-order accuracy */
  {
    FlucaFDTermLink term;

    PetscCall(FlucaFDTermLinkCreate_Internal(&term));
    term->deriv_order[tvd->dir] = 0;
    term->accu_order[tvd->dir]  = 2;
    term->input_loc             = fd->input_loc;
    term->input_c               = fd->input_c;
    PetscCall(FlucaFDTermLinkAppend_Internal(&fd->termlink, term));
  }

  PetscCheck(tvd->limiter, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Limiter not set. Use FlucaFDSecondOrderTVDSetLimiter() or -flucafd_limiter");
  if (tvd->limiter == FlucaFDSecondOrderTVDLimiterUpwind_Internal //
      || tvd->limiter == FlucaFDSecondOrderTVDLimiterSOU_Internal //
      || tvd->limiter == FlucaFDSecondOrderTVDLimiterQUICK_Internal)
    PetscCall(PetscInfo(fd, "Selected limiter is not second-order TVD\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFaceCenteredGradient(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscScalar *grad)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;
  PetscInt                ncols, bnd_idx, c;
  DMStagStencil           col[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar             v[FLUCAFD_MAX_STENCIL_SIZE];

  PetscFunctionBegin;
  PetscCall(FlucaFDGetStencil(tvd->fd_grad, i, j, k, &ncols, col, v));

  *grad = 0.0;
  for (c = 0; c < ncols; ++c)
    if (col[c].c >= 0) {
      /* Interior point */
      switch (fd->dim) {
      case 1:
        *grad += v[c] * tvd->arr_phi_1d[col[c].i][tvd->phi_slot];
        break;
      case 2:
        *grad += v[c] * tvd->arr_phi_2d[col[c].j][col[c].i][tvd->phi_slot];
        break;
      case 3:
        *grad += v[c] * tvd->arr_phi_3d[col[c].k][col[c].j][col[c].i][tvd->phi_slot];
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
      }
    } else if (FLUCAFD_BOUNDARY_FRONT <= col[c].c && col[c].c <= FLUCAFD_BOUNDARY_LEFT) {
      /* Boundary value */
      bnd_idx = -col[c].c - 1;
      *grad += v[c] * fd->bcs[bnd_idx].value;
    } else {
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported stencil point");
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_SecondOrderTVD(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;
  PetscInt                idx, i_u, j_u, k_u, i_d, j_d, k_d, i_fu, j_fu, k_fu, i_fc, j_fc, k_fc;
  PetscScalar             vel, alpha, grad_fu, grad_fc, r, psi, phi_u, phi_d;
  PetscBool               periodic, at_prev_boundary, at_next_boundary;

  PetscFunctionBegin;
  switch (tvd->dir) {
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
    SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Invalid direction");
  }
  PetscCheck(tvd->alpha_start <= idx && idx < tvd->alpha_end, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Face index out of range");

  switch (fd->dim) {
  case 1:
    vel = tvd->arr_vel_1d[i][tvd->vel_slot];
    break;
  case 2:
    vel = tvd->arr_vel_2d[j][i][tvd->vel_slot];
    break;
  case 3:
    vel = tvd->arr_vel_3d[k][j][i][tvd->vel_slot];
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
  }

  periodic         = fd->bcs[2 * tvd->dir].type == FLUCAFD_BC_PERIODIC;
  at_prev_boundary = (idx == 0 && !periodic);
  at_next_boundary = (idx == fd->N[tvd->dir] && !periodic);

  if (vel > 0) {
    i_u = (tvd->dir == FLUCAFD_X) ? i - 1 : i;
    j_u = (tvd->dir == FLUCAFD_Y) ? j - 1 : j;
    k_u = (tvd->dir == FLUCAFD_Z) ? k - 1 : k;
    i_d = i;
    j_d = j;
    k_d = k;
    if (at_prev_boundary) {
      *ncols     = 2;
      col[0].i   = i_u;
      col[0].j   = j_u;
      col[0].k   = k_u;
      col[0].loc = fd->input_loc;
      col[0].c   = fd->input_c;
      col[1].i   = i_d;
      col[1].j   = j_d;
      col[1].k   = k_d;
      col[1].loc = fd->input_loc;
      col[1].c   = fd->input_c;
      v[0]       = 0.5;
      v[1]       = 0.5;
    } else {
      alpha = tvd->alpha_plus[idx];
      i_fu  = (tvd->dir == FLUCAFD_X) ? i - 1 : i;
      j_fu  = (tvd->dir == FLUCAFD_Y) ? j - 1 : j;
      k_fu  = (tvd->dir == FLUCAFD_Z) ? k - 1 : k;
      i_fc  = i;
      j_fc  = j;
      k_fc  = k;
      PetscCall(ComputeFaceCenteredGradient(fd, i_fu, j_fu, k_fu, &grad_fu));
      PetscCall(ComputeFaceCenteredGradient(fd, i_fc, j_fc, k_fc, &grad_fc));
      r   = (PetscAbsScalar(grad_fc) > 1e-30) ? grad_fu / grad_fc : 1.;
      psi = tvd->limiter(r);
      switch (fd->dim) {
      case 1:
        phi_u = tvd->arr_phi_1d[i_u][tvd->phi_slot];
        phi_d = tvd->arr_phi_1d[i_d][tvd->phi_slot];
        break;
      case 2:
        phi_u = tvd->arr_phi_2d[j_u][i_u][tvd->phi_slot];
        phi_d = tvd->arr_phi_2d[j_d][i_d][tvd->phi_slot];
        break;
      case 3:
        phi_u = tvd->arr_phi_3d[k_u][j_u][i_u][tvd->phi_slot];
        phi_d = tvd->arr_phi_3d[k_d][j_d][i_d][tvd->phi_slot];
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
      }
      *ncols     = 2;
      col[0].i   = i_u;
      col[0].j   = j_u;
      col[0].k   = k_u;
      col[0].loc = fd->input_loc;
      col[0].c   = fd->input_c;
      col[1].i   = 0;
      col[1].j   = 0;
      col[1].k   = 0;
      col[1].loc = DMSTAG_ELEMENT;
      col[1].c   = FLUCAFD_CONSTANT;
      v[0]       = 1.;
      v[1]       = alpha * psi * (phi_d - phi_u);
    }
  } else {
    i_u = i;
    j_u = j;
    k_u = k;
    i_d = (tvd->dir == FLUCAFD_X) ? i - 1 : i;
    j_d = (tvd->dir == FLUCAFD_Y) ? j - 1 : j;
    k_d = (tvd->dir == FLUCAFD_Z) ? k - 1 : k;
    if (at_next_boundary) {
      *ncols     = 2;
      col[0].i   = i_u;
      col[0].j   = j_u;
      col[0].k   = k_u;
      col[0].loc = fd->input_loc;
      col[0].c   = fd->input_c;
      col[1].i   = i_d;
      col[1].j   = j_d;
      col[1].k   = k_d;
      col[1].loc = fd->input_loc;
      col[1].c   = fd->input_c;
      v[0]       = 0.5;
      v[1]       = 0.5;
    } else {
      alpha = tvd->alpha_minus[idx];
      i_fu  = (tvd->dir == FLUCAFD_X) ? i + 1 : i;
      j_fu  = (tvd->dir == FLUCAFD_Y) ? j + 1 : j;
      k_fu  = (tvd->dir == FLUCAFD_Z) ? k + 1 : k;
      i_fc  = i;
      j_fc  = j;
      k_fc  = k;
      PetscCall(ComputeFaceCenteredGradient(fd, i_fu, j_fu, k_fu, &grad_fu));
      PetscCall(ComputeFaceCenteredGradient(fd, i_fc, j_fc, k_fc, &grad_fc));
      r   = (PetscAbsScalar(grad_fc) > 1e-30) ? grad_fu / grad_fc : 1.;
      psi = tvd->limiter(r);
      switch (fd->dim) {
      case 1:
        phi_u = tvd->arr_phi_1d[i_u][tvd->phi_slot];
        phi_d = tvd->arr_phi_1d[i_d][tvd->phi_slot];
        break;
      case 2:
        phi_u = tvd->arr_phi_2d[j_u][i_u][tvd->phi_slot];
        phi_d = tvd->arr_phi_2d[j_d][i_d][tvd->phi_slot];
        break;
      case 3:
        phi_u = tvd->arr_phi_3d[k_u][j_u][i_u][tvd->phi_slot];
        phi_d = tvd->arr_phi_3d[k_d][j_d][i_d][tvd->phi_slot];
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
      }
      *ncols     = 2;
      col[0].i   = i;
      col[0].j   = j;
      col[0].k   = k;
      col[0].loc = fd->input_loc;
      col[0].c   = fd->input_c;
      col[1].i   = 0;
      col[1].j   = 0;
      col[1].k   = 0;
      col[1].loc = DMSTAG_ELEMENT;
      col[1].c   = FLUCAFD_CONSTANT;
      v[0]       = 1.;
      v[1]       = alpha * psi * (phi_d - phi_u);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_SecondOrderTVD(FlucaFD fd)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;

  PetscFunctionBegin;
  PetscCall(FlucaFDDestroy(&tvd->fd_grad));

  PetscCall(PetscFree(tvd->alpha_plus_base));
  PetscCall(PetscFree(tvd->alpha_minus_base));

  if (tvd->vel_dm) {
    switch (fd->dim) {
    case 1:
      PetscCall(DMStagVecRestoreArrayRead(tvd->vel_dm, tvd->vel_local, &tvd->arr_vel_1d));
      break;
    case 2:
      PetscCall(DMStagVecRestoreArrayRead(tvd->vel_dm, tvd->vel_local, &tvd->arr_vel_2d));
      break;
    case 3:
      PetscCall(DMStagVecRestoreArrayRead(tvd->vel_dm, tvd->vel_local, &tvd->arr_vel_3d));
      break;
    default:
      break;
    }
    PetscCall(VecDestroy(&tvd->vel_local));
    PetscCall(DMDestroy(&tvd->vel_dm));
  }

  if (tvd->phi_dm) {
    switch (fd->dim) {
    case 1:
      PetscCall(DMStagVecRestoreArrayRead(tvd->phi_dm, tvd->phi_local, &tvd->arr_phi_1d));
      break;
    case 2:
      PetscCall(DMStagVecRestoreArrayRead(tvd->phi_dm, tvd->phi_local, &tvd->arr_phi_2d));
      break;
    case 3:
      PetscCall(DMStagVecRestoreArrayRead(tvd->phi_dm, tvd->phi_local, &tvd->arr_phi_3d));
      break;
    default:
      break;
    }
    PetscCall(VecDestroy(&tvd->phi_local));
    PetscCall(DMDestroy(&tvd->phi_dm));
  }

  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_SecondOrderTVD(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;
  PetscBool               isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Direction: %s\n", FlucaFDDirections[tvd->dir]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Velocity component: %" PetscInt_FMT "\n", tvd->vel_c));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_SecondOrderTVD(FlucaFD fd)
{
  FlucaFD_SecondOrderTVD *tvd;

  PetscFunctionBegin;
  PetscCall(PetscNew(&tvd));
  tvd->dir              = FLUCAFD_X;
  tvd->limiter          = FlucaFDSecondOrderTVDLimiterSuperbee_Internal;
  tvd->alpha_start      = 0;
  tvd->alpha_end        = 0;
  tvd->alpha_plus       = NULL;
  tvd->alpha_minus      = NULL;
  tvd->alpha_plus_base  = NULL;
  tvd->alpha_minus_base = NULL;
  tvd->vel_c            = 0;
  tvd->vel_dm           = NULL;
  tvd->vel_local        = NULL;
  tvd->arr_vel_1d       = NULL;
  tvd->arr_vel_2d       = NULL;
  tvd->arr_vel_3d       = NULL;
  tvd->vel_slot         = 0;
  tvd->phi_dm           = NULL;
  tvd->phi_local        = NULL;
  tvd->arr_phi_1d       = NULL;
  tvd->arr_phi_2d       = NULL;
  tvd->arr_phi_3d       = NULL;
  tvd->phi_slot         = 0;
  tvd->fd_grad          = NULL;

  fd->data                = (void *)tvd;
  fd->ops->setfromoptions = FlucaFDSetFromOptions_SecondOrderTVD;
  fd->ops->setup          = FlucaFDSetUp_SecondOrderTVD;
  fd->ops->getstencilraw  = FlucaFDGetStencilRaw_SecondOrderTVD;
  fd->ops->destroy        = FlucaFDDestroy_SecondOrderTVD;
  fd->ops->view           = FlucaFDView_SecondOrderTVD;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDCreate(DM cdm, FlucaFDDirection dir, PetscInt input_c, PetscInt output_c, FlucaFD *fd)
{
  DMStagStencilLocation output_loc;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(cdm, DM_CLASSID, 1, DMPRODUCT);
  PetscAssertPointer(fd, 8);

  switch (dir) {
  case FLUCAFD_X:
    output_loc = DMSTAG_LEFT;
    break;
  case FLUCAFD_Y:
    output_loc = DMSTAG_DOWN;
    break;
  case FLUCAFD_Z:
    output_loc = DMSTAG_BACK;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)cdm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid direction");
  }

  PetscCall(FlucaFDCreate(PetscObjectComm((PetscObject)cdm), fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDSECONDORDERTVD));
  PetscCall(FlucaFDSetCoordinateDM(*fd, cdm));
  PetscCall(FlucaFDSetInputLocation(*fd, DMSTAG_ELEMENT, input_c));
  PetscCall(FlucaFDSetOutputLocation(*fd, output_loc, output_c));
  PetscCall(FlucaFDSecondOrderTVDSetDirection(*fd, dir));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDSetDirection(FlucaFD fd, FlucaFDDirection dir)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSECONDORDERTVD);
  tvd->dir = dir;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDSetLimiter(FlucaFD fd, const char *name)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;
  FlucaFDLimiterFn       *func;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSECONDORDERTVD);
  PetscCall(PetscFunctionListFind(FlucaFDLimiterList, name, &func));
  PetscCheck(func, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown limiter type: %s", name);
  tvd->limiter = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDSetVelocity(FlucaFD fd, Vec vel, PetscInt vel_c)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSECONDORDERTVD);
  PetscValidHeaderSpecific(vel, VEC_CLASSID, 2);
  tvd->vel_c = vel_c;
  if (!tvd->vel_dm) {
    PetscCall(VecGetDM(vel, &tvd->vel_dm));
    PetscCall(PetscObjectReference((PetscObject)tvd->vel_dm));
    PetscCall(DMCreateLocalVector(tvd->vel_dm, &tvd->vel_local));
    switch (fd->dim) {
    case 1:
      PetscCall(DMStagVecGetArrayRead(tvd->vel_dm, tvd->vel_local, &tvd->arr_vel_1d));
      break;
    case 2:
      PetscCall(DMStagVecGetArrayRead(tvd->vel_dm, tvd->vel_local, &tvd->arr_vel_2d));
      break;
    case 3:
      PetscCall(DMStagVecGetArrayRead(tvd->vel_dm, tvd->vel_local, &tvd->arr_vel_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
  }
  PetscCall(DMGlobalToLocal(tvd->vel_dm, vel, INSERT_VALUES, tvd->vel_local));
  PetscCall(DMStagGetLocationSlot(tvd->vel_dm, fd->output_loc, tvd->vel_c, &tvd->vel_slot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDSetCurrentSolution(FlucaFD fd, Vec phi)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSECONDORDERTVD);
  PetscValidHeaderSpecific(phi, VEC_CLASSID, 2);
  if (!tvd->phi_dm) {
    PetscCall(VecGetDM(phi, &tvd->phi_dm));
    PetscCall(PetscObjectReference((PetscObject)tvd->phi_dm));
    PetscCall(DMCreateLocalVector(tvd->phi_dm, &tvd->phi_local));
    switch (fd->dim) {
    case 1:
      PetscCall(DMStagVecGetArrayRead(tvd->phi_dm, tvd->phi_local, &tvd->arr_phi_1d));
      break;
    case 2:
      PetscCall(DMStagVecGetArrayRead(tvd->phi_dm, tvd->phi_local, &tvd->arr_phi_2d));
      break;
    case 3:
      PetscCall(DMStagVecGetArrayRead(tvd->phi_dm, tvd->phi_local, &tvd->arr_phi_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
  }
  PetscCall(DMGlobalToLocal(tvd->phi_dm, phi, INSERT_VALUES, tvd->phi_local));
  PetscCall(DMStagGetLocationSlot(tvd->phi_dm, fd->input_loc, fd->input_c, &tvd->phi_slot));
  PetscFunctionReturn(PETSC_SUCCESS);
}
