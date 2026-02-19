#include <fluca/private/flucafdimpl.h>
#include <petscdmda.h>

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
  if (flg) PetscCall(FlucaFDSecondOrderTVDSetLimiter(fd, lname));
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
  PetscCall(FlucaFDDerivativeCreate(fd->dm, tvd->dir, 1, 1, fd->input_loc, fd->input_c, fd->output_loc, 0, &tvd->fd_grad));
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
    PetscInt            gxs, gxm, gxe, i;

    arr_coord = fd->arr_coord[tvd->dir];
    periodic  = fd->periodic[tvd->dir];

    /* Local grid info */
    PetscCall(FlucaFDGetGhostCorners_Internal(fd, tvd->dir, PETSC_TRUE, &gxs, &gxm, &gxe));

    tvd->alpha_start = gxs;
    tvd->alpha_end   = gxs + gxm + gxe;
    PetscCall(PetscCalloc1(gxm + gxe, &tvd->alpha_plus));
    PetscCall(PetscCalloc1(gxm + gxe, &tvd->alpha_minus));
    tvd->alpha_plus_base  = tvd->alpha_plus;
    tvd->alpha_minus_base = tvd->alpha_minus;
    tvd->alpha_plus -= gxs;
    tvd->alpha_minus -= gxs;

    for (i = gxs; i < gxs + gxm + gxe; i++) {
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

  *grad = 0.;
  for (c = 0; c < ncols; ++c)
    if (col[c].c >= 0) {
      /* Interior point */
      switch (fd->dim) {
      case 1:
        *grad += v[c] * tvd->arr_phi_1d[col[c].i];
        break;
      case 2:
        *grad += v[c] * tvd->arr_phi_2d[col[c].j][col[c].i];
        break;
      case 3:
        *grad += v[c] * tvd->arr_phi_3d[col[c].k][col[c].j][col[c].i];
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
    vel = tvd->arr_vel_1d[i];
    break;
  case 2:
    vel = tvd->arr_vel_2d[j][i];
    break;
  case 3:
    vel = tvd->arr_vel_3d[k][j][i];
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
  }

  periodic         = fd->periodic[tvd->dir];
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
        phi_u = tvd->arr_phi_1d[i_u];
        phi_d = tvd->arr_phi_1d[i_d];
        break;
      case 2:
        phi_u = tvd->arr_phi_2d[j_u][i_u];
        phi_d = tvd->arr_phi_2d[j_d][i_d];
        break;
      case 3:
        phi_u = tvd->arr_phi_3d[k_u][j_u][i_u];
        phi_d = tvd->arr_phi_3d[k_d][j_d][i_d];
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
        phi_u = tvd->arr_phi_1d[i_u];
        phi_d = tvd->arr_phi_1d[i_d];
        break;
      case 2:
        phi_u = tvd->arr_phi_2d[j_u][i_u];
        phi_d = tvd->arr_phi_2d[j_d][i_d];
        break;
      case 3:
        phi_u = tvd->arr_phi_3d[k_u][j_u][i_u];
        phi_d = tvd->arr_phi_3d[k_d][j_d][i_d];
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
      PetscCall(DMDAVecRestoreArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_1d));
      break;
    case 2:
      PetscCall(DMDAVecRestoreArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_2d));
      break;
    case 3:
      PetscCall(DMDAVecRestoreArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_3d));
      break;
    default:
      break;
    }
    PetscCall(VecScatterDestroy(&tvd->vel_scatter));
    PetscCall(VecDestroy(&tvd->vel_local));
    PetscCall(DMDestroy(&tvd->vel_da));
    PetscCall(DMDestroy(&tvd->vel_dm));
  }

  if (tvd->phi_dm) {
    switch (fd->dim) {
    case 1:
      PetscCall(DMDAVecRestoreArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_1d));
      break;
    case 2:
      PetscCall(DMDAVecRestoreArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_2d));
      break;
    case 3:
      PetscCall(DMDAVecRestoreArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_3d));
      break;
    default:
      break;
    }
    PetscCall(VecScatterDestroy(&tvd->phi_scatter));
    PetscCall(VecDestroy(&tvd->phi_local));
    PetscCall(DMDestroy(&tvd->phi_da));
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
  tvd->vel_da           = NULL;
  tvd->vel_local        = NULL;
  tvd->vel_scatter      = NULL;
  tvd->arr_vel_1d       = NULL;
  tvd->arr_vel_2d       = NULL;
  tvd->arr_vel_3d       = NULL;
  tvd->phi_dm           = NULL;
  tvd->phi_da           = NULL;
  tvd->phi_local        = NULL;
  tvd->phi_scatter      = NULL;
  tvd->arr_phi_1d       = NULL;
  tvd->arr_phi_2d       = NULL;
  tvd->arr_phi_3d       = NULL;
  tvd->fd_grad          = NULL;

  fd->data                = (void *)tvd;
  fd->ops->setfromoptions = FlucaFDSetFromOptions_SecondOrderTVD;
  fd->ops->setup          = FlucaFDSetUp_SecondOrderTVD;
  fd->ops->getstencilraw  = FlucaFDGetStencilRaw_SecondOrderTVD;
  fd->ops->destroy        = FlucaFDDestroy_SecondOrderTVD;
  fd->ops->view           = FlucaFDView_SecondOrderTVD;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDCreate(DM dm, FlucaFDDirection dir, PetscInt input_c, PetscInt output_c, FlucaFD *fd)
{
  DMStagStencilLocation output_loc;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMSTAG);
  PetscAssertPointer(fd, 5);

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
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid direction");
  }

  PetscCall(FlucaFDCreate(PetscObjectComm((PetscObject)dm), fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDSECONDORDERTVD));
  PetscCall(FlucaFDSetDM(*fd, dm));
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

static PetscErrorCode CreateDMStagToDAScatter_Private(DM stag, PetscInt dim, DMStagStencilLocation loc, PetscInt c, Vec vec, DM *da, Vec *vec_local, VecScatter *scatter)
{
  MPI_Comm               comm;
  ISLocalToGlobalMapping ltog;
  IS                     is_from, is_to;
  DMBoundaryType         bt[3];
  PetscInt               N[3], num_ranks[3], sw, epe, slot;
  PetscInt               gx[3], gn[3];
  PetscInt               da_N[3], da_gx[3], da_gn[3];
  PetscInt               n_local, n_valid, d, i, j, k, cnt;
  PetscInt              *stag_local, *stag_global, *da_local;
  const PetscInt        *l[3];
  PetscInt              *l_da[3] = {NULL, NULL, NULL};

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)stag, &comm));

  /* Query DMStag properties */
  PetscCall(DMStagGetGlobalSizes(stag, &N[0], &N[1], &N[2]));
  PetscCall(DMStagGetNumRanks(stag, &num_ranks[0], &num_ranks[1], &num_ranks[2]));
  PetscCall(DMStagGetOwnershipRanges(stag, &l[0], &l[1], &l[2]));
  PetscCall(DMStagGetBoundaryTypes(stag, &bt[0], &bt[1], &bt[2]));
  PetscCall(DMStagGetGhostCorners(stag, &gx[0], &gx[1], &gx[2], &gn[0], &gn[1], &gn[2]));
  PetscCall(DMStagGetStencilWidth(stag, &sw));
  PetscCall(DMStagGetEntriesPerElement(stag, &epe));
  PetscCall(DMStagGetLocationSlot(stag, loc, c, &slot));
  PetscCall(DMGetLocalToGlobalMapping(stag, &ltog));

  /* Create DMDA with matching topology, 1 DOF per point */
  for (d = 0; d < dim; d++) {
    PetscBool face;
    PetscInt  extra;

    PetscCall(FlucaFDUseFaceCoordinate_Internal(loc, d, &face));
    extra   = (face && bt[d] != DM_BOUNDARY_PERIODIC) ? 1 : 0;
    da_N[d] = N[d] + extra;
    PetscCall(PetscMalloc1(num_ranks[d], &l_da[d]));
    PetscCall(PetscArraycpy(l_da[d], l[d], num_ranks[d]));
    l_da[d][num_ranks[d] - 1] += extra;
  }

  switch (dim) {
  case 1:
    PetscCall(DMDACreate1d(comm, bt[0], da_N[0], 1, sw, l_da[0], da));
    break;
  case 2:
    PetscCall(DMDACreate2d(comm, bt[0], bt[1], DMDA_STENCIL_BOX, da_N[0], da_N[1], num_ranks[0], num_ranks[1], 1, sw, l_da[0], l_da[1], da));
    break;
  case 3:
    PetscCall(DMDACreate3d(comm, bt[0], bt[1], bt[2], DMDA_STENCIL_BOX, da_N[0], da_N[1], da_N[2], num_ranks[0], num_ranks[1], num_ranks[2], 1, sw, l_da[0], l_da[1], l_da[2], da));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported dim %" PetscInt_FMT, dim);
  }
  PetscCall(DMSetUp(*da));
  for (d = 0; d < dim; d++) PetscCall(PetscFree(l_da[d]));

  /* Create local vector on DMDA */
  PetscCall(DMCreateLocalVector(*da, vec_local));

  /* Get DMDA ghost corners */
  PetscCall(DMDAGetGhostCorners(*da, &da_gx[0], &da_gx[1], &da_gx[2], &da_gn[0], &da_gn[1], &da_gn[2]));

  /* Build IS: for each DMDA ghost point, map DMStag local index of the target DOF to global index via l2g mapping */
  n_local = 1;
  for (d = 0; d < dim; d++) n_local *= da_gn[d];
  PetscCall(PetscMalloc1(n_local, &stag_local));
  PetscCall(PetscMalloc1(n_local, &stag_global));
  cnt = 0;
  switch (dim) {
  case 1:
    for (i = da_gx[0]; i < da_gx[0] + da_gn[0]; i++) stag_local[cnt++] = (i - gx[0]) * epe + slot;
    break;
  case 2:
    for (j = da_gx[1]; j < da_gx[1] + da_gn[1]; j++)
      for (i = da_gx[0]; i < da_gx[0] + da_gn[0]; i++) stag_local[cnt++] = ((j - gx[1]) * gn[0] + (i - gx[0])) * epe + slot;
    break;
  case 3:
    for (k = da_gx[2]; k < da_gx[2] + da_gn[2]; k++)
      for (j = da_gx[1]; j < da_gx[1] + da_gn[1]; j++)
        for (i = da_gx[0]; i < da_gx[0] + da_gn[0]; i++) stag_local[cnt++] = ((k - gx[2]) * gn[1] * gn[0] + (j - gx[1]) * gn[0] + (i - gx[0])) * epe + slot;
    break;
  default:
    break;
  }
  PetscCall(ISLocalToGlobalMappingApply(ltog, n_local, stag_local, stag_global));
  PetscCall(PetscFree(stag_local));

  /* Remove unmapped entries (-1) that arise when DMStag uses STAR stencil
     (whose L2G mapping excludes corner ghost points in 2D/3D). Build a
     matching DMDA local index array so the scatter targets correct positions
     in the full rectangular DMDA local vector. */
  PetscCall(PetscMalloc1(n_local, &da_local));
  n_valid = 0;
  for (i = 0; i < n_local; i++)
    if (stag_global[i] >= 0) {
      stag_global[n_valid] = stag_global[i];
      da_local[n_valid]    = i;
      n_valid++;
    }
  PetscCall(ISCreateGeneral(comm, n_valid, stag_global, PETSC_OWN_POINTER, &is_from));
  PetscCall(ISCreateGeneral(comm, n_valid, da_local, PETSC_OWN_POINTER, &is_to));

  /* Create VecScatter */
  PetscCall(VecScatterCreate(vec, is_from, *vec_local, is_to, scatter));
  PetscCall(ISDestroy(&is_from));
  PetscCall(ISDestroy(&is_to));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDSetVelocity(FlucaFD fd, Vec vel, PetscInt vel_c)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;
  DM                      vel_dm;
  PetscBool               isstag;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSECONDORDERTVD);
  PetscValidHeaderSpecific(vel, VEC_CLASSID, 2);
  PetscCall(VecGetDM(vel, &vel_dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)vel_dm, DMSTAG, &isstag));
  PetscCheck(isstag, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Vector is not on DMStag");

  /* Recreate scatter if DM or component changed */
  if (vel_dm != tvd->vel_dm || vel_c != tvd->vel_c) {
    if (tvd->vel_dm) {
      switch (fd->dim) {
      case 1:
        PetscCall(DMDAVecRestoreArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_1d));
        break;
      case 2:
        PetscCall(DMDAVecRestoreArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_2d));
        break;
      case 3:
        PetscCall(DMDAVecRestoreArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_3d));
        break;
      default:
        break;
      }
    }
    PetscCall(VecScatterDestroy(&tvd->vel_scatter));
    PetscCall(VecDestroy(&tvd->vel_local));
    PetscCall(DMDestroy(&tvd->vel_da));
    PetscCall(DMDestroy(&tvd->vel_dm));
  }

  tvd->vel_c = vel_c;
  if (!tvd->vel_dm) {
    tvd->vel_dm = vel_dm;
    PetscCall(PetscObjectReference((PetscObject)tvd->vel_dm));
    PetscCall(CreateDMStagToDAScatter_Private(tvd->vel_dm, fd->dim, fd->output_loc, vel_c, vel, &tvd->vel_da, &tvd->vel_local, &tvd->vel_scatter));

    /* Get array views on local DMDA vector (kept until destroy) */
    switch (fd->dim) {
    case 1:
      PetscCall(DMDAVecGetArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_1d));
      break;
    case 2:
      PetscCall(DMDAVecGetArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_2d));
      break;
    case 3:
      PetscCall(DMDAVecGetArrayRead(tvd->vel_da, tvd->vel_local, &tvd->arr_vel_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
  }
  /* Scatter only the needed face velocity component from global to local */
  PetscCall(VecScatterBegin(tvd->vel_scatter, vel, tvd->vel_local, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(tvd->vel_scatter, vel, tvd->vel_local, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSecondOrderTVDSetCurrentSolution(FlucaFD fd, Vec phi)
{
  FlucaFD_SecondOrderTVD *tvd = (FlucaFD_SecondOrderTVD *)fd->data;
  DM                      phi_dm;
  PetscBool               isstag;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDSECONDORDERTVD);
  PetscValidHeaderSpecific(phi, VEC_CLASSID, 2);
  PetscCall(VecGetDM(phi, &phi_dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)phi_dm, DMSTAG, &isstag));
  PetscCheck(isstag, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Vector is not on DMStag");

  /* Recreate scatter if DM changed */
  if (phi_dm != tvd->phi_dm) {
    if (tvd->phi_dm) {
      switch (fd->dim) {
      case 1:
        PetscCall(DMDAVecRestoreArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_1d));
        break;
      case 2:
        PetscCall(DMDAVecRestoreArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_2d));
        break;
      case 3:
        PetscCall(DMDAVecRestoreArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_3d));
        break;
      default:
        break;
      }
    }
    PetscCall(VecScatterDestroy(&tvd->phi_scatter));
    PetscCall(VecDestroy(&tvd->phi_local));
    PetscCall(DMDestroy(&tvd->phi_da));
    PetscCall(DMDestroy(&tvd->phi_dm));
  }

  if (!tvd->phi_dm) {
    tvd->phi_dm = phi_dm;
    PetscCall(PetscObjectReference((PetscObject)tvd->phi_dm));
    PetscCall(CreateDMStagToDAScatter_Private(tvd->phi_dm, fd->dim, fd->input_loc, fd->input_c, phi, &tvd->phi_da, &tvd->phi_local, &tvd->phi_scatter));

    /* Get array views on local DMDA vector (kept until destroy) */
    switch (fd->dim) {
    case 1:
      PetscCall(DMDAVecGetArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_1d));
      break;
    case 2:
      PetscCall(DMDAVecGetArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_2d));
      break;
    case 3:
      PetscCall(DMDAVecGetArrayRead(tvd->phi_da, tvd->phi_local, &tvd->arr_phi_3d));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
  }
  /* Scatter only the needed element component from global to local */
  PetscCall(VecScatterBegin(tvd->phi_scatter, phi, tvd->phi_local, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(tvd->phi_scatter, phi, tvd->phi_local, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}
