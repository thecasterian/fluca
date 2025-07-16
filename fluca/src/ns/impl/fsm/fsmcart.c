#include <fluca/private/flucaviewercgnsimpl.h>
#include <fluca/private/nsfsmimpl.h>
#include <flucameshcart.h>
#include <petscdmstag.h>

static PetscErrorCode ComputeRHSUStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSVStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSPprime2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeOperatorPprime2d_Private(KSP, Mat, Mat, void *);

static PetscErrorCode GetBoundaryConditions2d_Private(NS ns, NSBoundaryCondition *bcleft, NSBoundaryCondition *bcright, NSBoundaryCondition *bcdown, NSBoundaryCondition *bcup)
{
  PetscInt ileftb, irightb, idownb, iupb;

  PetscFunctionBegin;
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_LEFT, &ileftb));
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_RIGHT, &irightb));
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_DOWN, &idownb));
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_UP, &iupb));
  PetscCall(NSGetBoundaryCondition(ns, ileftb, bcleft));
  PetscCall(NSGetBoundaryCondition(ns, irightb, bcright));
  PetscCall(NSGetBoundaryCondition(ns, idownb, bcdown));
  PetscCall(NSGetBoundaryCondition(ns, iupb, bcup));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMInterpolateVelocity2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  Vec u = fsm->v[0], v = fsm->v[1];
  Vec fv = fsm->fv;

  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  PetscScalar       ***arrUV;
  const PetscScalar ***arru, ***arrv;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          wx_left, wx_right, wy_down, wy_up;
  PetscInt             i, j;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMStagVecGetArray(fdm, fv, &arrUV));
  PetscCall(DMStagVecGetArrayRead(dm, u, &arru));
  PetscCall(DMStagVecGetArrayRead(dm, v, &arrv));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      /* Left wall. */
      if (i == 0) {
        arrUV[j][i][ileft] = 0.0;
      }
      /* Right wall. */
      else if (i == M) {
        arrUV[j][i][ileft] = 0.0;
      } else {
        wx_left            = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
        wx_right           = arrcx[i][inextc] - arrcx[i][iprevc];
        arrUV[j][i][ileft] = (wx_right * arru[j][i - 1][ielem] + wx_left * arru[j][i][ielem]) / (wx_left + wx_right);
      }
    }
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      /* Bottom wall. */
      if (j == 0) {
        arrUV[j][i][idown] = 0.0;
      }
      /* Top wall. */
      else if (j == N) {
        arrUV[j][i][idown] = 0.0;
      } else {
        wy_down            = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
        wy_up              = arrcy[j][inextc] - arrcy[j][iprevc];
        arrUV[j][i][idown] = (wy_up * arrv[j - 1][i][ielem] + wy_down * arrv[j][i][ielem]) / (wy_down + wy_up);
      }
    }

  PetscCall(DMStagVecRestoreArray(fdm, fv, &arrUV));
  PetscCall(DMStagVecRestoreArrayRead(dm, u, &arru));
  PetscCall(DMStagVecRestoreArrayRead(dm, v, &arrv));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMLocalToLocalBegin(fdm, fv, INSERT_VALUES, fv));
  PetscCall(DMLocalToLocalEnd(fdm, fv, INSERT_VALUES, fv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateConvection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  const PetscReal t_next = ns->t + ns->dt;

  Vec u = fsm->v[0], v = fsm->v[1];
  Vec fv = fsm->fv;
  Vec Nu = fsm->N[0], Nv = fsm->N[1];
  Vec u_interp, v_interp;

  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  PetscScalar       ***arrNu, ***arrNv;
  PetscScalar       ***arru_interp, ***arrv_interp;
  const PetscScalar ***arru, ***arrv, ***arrUV;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, iright, idown, iup, ielem, iprevc, inextc, ielemc;
  PetscScalar          h1, h2, hx, hy;
  PetscInt             i, j;

  NSBoundaryCondition bcleft, bcright, bcdown, bcup;
  PetscReal           xb[2];
  PetscScalar         vb[2];

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMGetLocalVector(fdm, &u_interp));
  PetscCall(DMGetLocalVector(fdm, &v_interp));

  PetscCall(DMStagVecGetArray(dm, Nu, &arrNu));
  PetscCall(DMStagVecGetArray(dm, Nv, &arrNv));
  PetscCall(DMStagVecGetArray(fdm, u_interp, &arru_interp));
  PetscCall(DMStagVecGetArray(fdm, v_interp, &arrv_interp));
  PetscCall(DMStagVecGetArrayRead(dm, u, &arru));
  PetscCall(DMStagVecGetArrayRead(dm, v, &arrv));
  PetscCall(DMStagVecGetArrayRead(fdm, fv, &arrUV));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_RIGHT, 0, &iright));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_UP, 0, &iup));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(GetBoundaryConditions2d_Private(ns, &bcleft, &bcright, &bcdown, &bcup));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      if (i == 0) {
        /* Left boundary */
        switch (bcleft.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcleft.velocity(2, t_next, xb, vb, bcleft.ctx_velocity));
          arru_interp[j][i][ileft] = vb[0];
          arrv_interp[j][i][ileft] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                       = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
          h2                       = arrcx[i][inextc] - arrcx[i][iprevc];
          arru_interp[j][i][ileft] = (h2 * arru[j][i - 1][ielem] + h1 * arru[j][i][ielem]) / (h1 + h2);
          arrv_interp[j][i][ileft] = (h2 * arrv[j][i - 1][ielem] + h1 * arrv[j][i][ielem]) / (h1 + h2);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary: %s", NSBoundaryConditionTypes[bcleft.type]);
        }
      } else if (i == M) {
        /* Right boundary */
        switch (bcright.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcright.velocity(2, t_next, xb, vb, bcright.ctx_velocity));
          arru_interp[j][i][ileft] = vb[0];
          arrv_interp[j][i][ileft] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                       = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
          h2                       = arrcx[i][inextc] - arrcx[i][iprevc];
          arru_interp[j][i][ileft] = (h2 * arru[j][i - 1][ielem] + h1 * arru[j][i][ielem]) / (h1 + h2);
          arrv_interp[j][i][ileft] = (h2 * arrv[j][i - 1][ielem] + h1 * arrv[j][i][ielem]) / (h1 + h2);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary: %s", NSBoundaryConditionTypes[bcright.type]);
        }
      } else {
        h1                       = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
        h2                       = arrcx[i][inextc] - arrcx[i][iprevc];
        arru_interp[j][i][ileft] = (h2 * arru[j][i - 1][ielem] + h1 * arru[j][i][ielem]) / (h1 + h2);
        arrv_interp[j][i][ileft] = (h2 * arrv[j][i - 1][ielem] + h1 * arrv[j][i][ielem]) / (h1 + h2);
      }
    }
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      if (j == 0) {
        /* Down boundary */
        switch (bcdown.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][iprevc];
          PetscCall(bcdown.velocity(2, t_next, xb, vb, bcdown.ctx_velocity));
          arru_interp[j][i][idown] = vb[0];
          arrv_interp[j][i][idown] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                       = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
          h2                       = arrcy[j][inextc] - arrcy[j][iprevc];
          arru_interp[j][i][idown] = (h2 * arru[j - 1][i][ielem] + h1 * arru[j][i][ielem]) / (h1 + h2);
          arrv_interp[j][i][idown] = (h2 * arrv[j - 1][i][ielem] + h1 * arrv[j][i][ielem]) / (h1 + h2);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary: %s", NSBoundaryConditionTypes[bcdown.type]);
        }
      } else if (j == N) {
        /* Up boundary */
        switch (bcup.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][iprevc];
          PetscCall(bcup.velocity(2, t_next, xb, vb, bcup.ctx_velocity));
          arru_interp[j][i][idown] = vb[0];
          arrv_interp[j][i][idown] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                       = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
          h2                       = arrcy[j][inextc] - arrcy[j][iprevc];
          arru_interp[j][i][idown] = (h2 * arru[j - 1][i][ielem] + h1 * arru[j][i][ielem]) / (h1 + h2);
          arrv_interp[j][i][idown] = (h2 * arrv[j - 1][i][ielem] + h1 * arrv[j][i][ielem]) / (h1 + h2);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary: %s", NSBoundaryConditionTypes[bcup.type]);
        }
      } else {
        h1                       = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
        h2                       = arrcy[j][inextc] - arrcy[j][iprevc];
        arru_interp[j][i][idown] = (h2 * arru[j - 1][i][ielem] + h1 * arru[j][i][ielem]) / (h1 + h2);
        arrv_interp[j][i][idown] = (h2 * arrv[j - 1][i][ielem] + h1 * arrv[j][i][ielem]) / (h1 + h2);
      }
    }

  PetscCall(DMLocalToLocalBegin(fdm, u_interp, INSERT_VALUES, u_interp));
  PetscCall(DMLocalToLocalEnd(fdm, u_interp, INSERT_VALUES, u_interp));
  PetscCall(DMLocalToLocalBegin(fdm, v_interp, INSERT_VALUES, v_interp));
  PetscCall(DMLocalToLocalEnd(fdm, v_interp, INSERT_VALUES, v_interp));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      hx                 = arrcx[i][inextc] - arrcx[i][iprevc];
      hy                 = arrcy[j][inextc] - arrcy[j][iprevc];
      arrNu[j][i][ielem] = (arrUV[j][i][iright] * arru_interp[j][i][iright] - arrUV[j][i][ileft] * arru_interp[j][i][ileft]) / hx //
                         + (arrUV[j][i][iup] * arru_interp[j][i][iup] - arrUV[j][i][idown] * arru_interp[j][i][idown]) / hy;
      arrNv[j][i][ielem] = (arrUV[j][i][iright] * arrv_interp[j][i][iright] - arrUV[j][i][ileft] * arrv_interp[j][i][ileft]) / hx //
                         + (arrUV[j][i][iup] * arrv_interp[j][i][iup] - arrUV[j][i][idown] * arrv_interp[j][i][idown]) / hy;
    }

  PetscCall(DMStagVecRestoreArray(dm, Nu, &arrNu));
  PetscCall(DMStagVecRestoreArray(dm, Nv, &arrNv));
  PetscCall(DMStagVecRestoreArray(fdm, u_interp, &arru_interp));
  PetscCall(DMStagVecRestoreArray(fdm, v_interp, &arrv_interp));
  PetscCall(DMStagVecRestoreArrayRead(dm, u, &arru));
  PetscCall(DMStagVecRestoreArrayRead(dm, v, &arrv));
  PetscCall(DMStagVecRestoreArrayRead(fdm, fv, &arrUV));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMRestoreLocalVector(fdm, &u_interp));
  PetscCall(DMRestoreLocalVector(fdm, &v_interp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  const PetscReal rho    = ns->rho;
  const PetscReal dt     = ns->dt;
  const PetscReal t_next = ns->t + ns->dt;

  Vec u_star = fsm->v_star[0], v_star = fsm->v_star[1];
  Vec fv_star = fsm->fv_star;
  Vec p       = fsm->p_half;
  Vec u_tilde, v_tilde;

  PetscErrorCode (*rhs[2])(KSP, Vec, void *) = {ComputeRHSUStar2d_Private, ComputeRHSVStar2d_Private};
  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  PetscScalar       ***arru_tilde, ***arrv_tilde, ***arrUV_star;
  const PetscScalar ***arru_star, ***arrv_star, ***arrp;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          U_tilde, V_tilde, dpdx, dpdy;
  PetscScalar          h1, h2;
  PetscInt             d, i, j;

  Vec            p_global;
  Vec            dpdx_global, dpdy_global, dpdx_local, dpdy_local;
  PetscScalar ***arrdpdx, ***arrdpdy;

  NSBoundaryCondition bcleft, bcright, bcdown, bcup;
  PetscReal           xb[2];
  PetscScalar         vb[2];

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    Vec s;

    PetscCall(KSPSetComputeRHS(fsm->kspv[d], rhs[d], ns));
    PetscCall(KSPSetComputeOperators(fsm->kspv[d], ComputeOperatorsUVstar2d_Private, ns));
    PetscCall(KSPSolve(fsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(fsm->kspv[d], &s));
    PetscCall(DMGlobalToLocal(dm, s, INSERT_VALUES, fsm->v_star[d]));
  }

  /* Calculate face-centered intermediate velocity. */
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &dpdx_global));
  PetscCall(DMGetGlobalVector(dm, &dpdy_global));
  PetscCall(DMGetLocalVector(dm, &dpdx_local));
  PetscCall(DMGetLocalVector(dm, &dpdy_local));

  PetscCall(DMGetLocalVector(dm, &u_tilde));
  PetscCall(DMGetLocalVector(dm, &v_tilde));

  PetscCall(DMStagVecGetArray(dm, u_tilde, &arru_tilde));
  PetscCall(DMStagVecGetArray(dm, v_tilde, &arrv_tilde));
  PetscCall(DMStagVecGetArray(fdm, fv_star, &arrUV_star));
  PetscCall(DMStagVecGetArrayRead(dm, u_star, &arru_star));
  PetscCall(DMStagVecGetArrayRead(dm, v_star, &arrv_star));
  PetscCall(DMStagVecGetArrayRead(dm, p, &arrp));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(GetBoundaryConditions2d_Private(ns, &bcleft, &bcright, &bcdown, &bcup));

  PetscCall(DMLocalToGlobal(dm, p, INSERT_VALUES, p_global));
  PetscCall(MatMult(fsm->grad_p[0], p_global, dpdx_global));
  PetscCall(MatMult(fsm->grad_p[1], p_global, dpdy_global));
  PetscCall(DMGlobalToLocal(dm, dpdx_global, INSERT_VALUES, dpdx_local));
  PetscCall(DMGlobalToLocal(dm, dpdy_global, INSERT_VALUES, dpdy_local));
  PetscCall(DMStagVecGetArray(dm, dpdx_local, &arrdpdx));
  PetscCall(DMStagVecGetArray(dm, dpdy_local, &arrdpdy));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      arru_tilde[j][i][ielem] = arru_star[j][i][ielem] + dt / rho * arrdpdx[j][i][ielem];
      arrv_tilde[j][i][ielem] = arrv_star[j][i][ielem] + dt / rho * arrdpdy[j][i][ielem];
    }

  PetscCall(DMLocalToLocalBegin(dm, u_tilde, INSERT_VALUES, u_tilde));
  PetscCall(DMLocalToLocalEnd(dm, u_tilde, INSERT_VALUES, u_tilde));
  PetscCall(DMLocalToLocalBegin(dm, v_tilde, INSERT_VALUES, v_tilde));
  PetscCall(DMLocalToLocalEnd(dm, v_tilde, INSERT_VALUES, v_tilde));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      if (i == 0) {
        /* Left boundary */
        switch (bcleft.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcleft.velocity(2, t_next, xb, vb, bcleft.ctx_velocity));
          arrUV_star[j][i][ileft] = vb[0];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
          h2                      = arrcx[i][inextc] - arrcx[i][iprevc];
          U_tilde                 = (h2 * arru_tilde[j][i - 1][ielem] + h1 * arru_tilde[j][i][ielem]) / (h1 + h2);
          dpdx                    = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          arrUV_star[j][i][ileft] = U_tilde - dt / rho * dpdx;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary: %s", NSBoundaryConditionTypes[bcleft.type]);
        }
      } else if (i == M) {
        /* Right boundary */
        switch (bcright.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcright.velocity(2, t_next, xb, vb, bcright.ctx_velocity));
          arrUV_star[j][i][ileft] = vb[0];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
          h2                      = arrcx[i][inextc] - arrcx[i][iprevc];
          U_tilde                 = (h2 * arru_tilde[j][i - 1][ielem] + h1 * arru_tilde[j][i][ielem]) / (h1 + h2);
          dpdx                    = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          arrUV_star[j][i][ileft] = U_tilde - dt / rho * dpdx;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary: %s", NSBoundaryConditionTypes[bcright.type]);
        }
      } else {
        h1                      = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
        h2                      = arrcx[i][inextc] - arrcx[i][iprevc];
        U_tilde                 = (h2 * arru_tilde[j][i - 1][ielem] + h1 * arru_tilde[j][i][ielem]) / (h1 + h2);
        dpdx                    = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        arrUV_star[j][i][ileft] = U_tilde - dt / rho * dpdx;
      }
    }
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      if (j == 0) {
        /* Down boundary */
        switch (bcdown.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][iprevc];
          PetscCall(bcdown.velocity(2, t_next, xb, vb, bcdown.ctx_velocity));
          arrUV_star[j][i][idown] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
          h2                      = arrcy[j][inextc] - arrcy[j][iprevc];
          V_tilde                 = (h2 * arrv_tilde[j - 1][i][ielem] + h1 * arrv_tilde[j][i][ielem]) / (h1 + h2);
          dpdy                    = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          arrUV_star[j][i][idown] = V_tilde - dt / rho * dpdy;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary: %s", NSBoundaryConditionTypes[bcdown.type]);
        }
      } else if (j == N) {
        /* Up boundary */
        switch (bcup.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][iprevc];
          PetscCall(bcup.velocity(2, t_next, xb, vb, bcup.ctx_velocity));
          arrUV_star[j][i][idown] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
          h2                      = arrcy[j][inextc] - arrcy[j][iprevc];
          V_tilde                 = (h2 * arrv_tilde[j - 1][i][ielem] + h1 * arrv_tilde[j][i][ielem]) / (h1 + h2);
          dpdy                    = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          arrUV_star[j][i][idown] = V_tilde - dt / rho * dpdy;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary: %s", NSBoundaryConditionTypes[bcup.type]);
        }
      } else {
        h1                      = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
        h2                      = arrcy[j][inextc] - arrcy[j][iprevc];
        V_tilde                 = (h2 * arrv_tilde[j - 1][i][ielem] + h1 * arrv_tilde[j][i][ielem]) / (h1 + h2);
        dpdy                    = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
        arrUV_star[j][i][idown] = V_tilde - dt / rho * dpdy;
      }
    }

  PetscCall(DMStagVecRestoreArray(dm, u_tilde, &arru_tilde));
  PetscCall(DMStagVecRestoreArray(dm, v_tilde, &arrv_tilde));
  PetscCall(DMStagVecRestoreArray(fdm, fv_star, &arrUV_star));
  PetscCall(DMStagVecRestoreArrayRead(dm, u_star, &arru_star));
  PetscCall(DMStagVecRestoreArrayRead(dm, v_star, &arrv_star));
  PetscCall(DMStagVecRestoreArrayRead(dm, p, &arrp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagVecRestoreArray(dm, dpdx_local, &arrdpdx));
  PetscCall(DMStagVecRestoreArray(dm, dpdy_local, &arrdpdy));

  PetscCall(DMLocalToLocalBegin(fdm, fv_star, INSERT_VALUES, fv_star));
  PetscCall(DMLocalToLocalEnd(fdm, fv_star, INSERT_VALUES, fv_star));

  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &dpdx_global));
  PetscCall(DMRestoreGlobalVector(dm, &dpdy_global));
  PetscCall(DMRestoreLocalVector(dm, &dpdx_local));
  PetscCall(DMRestoreLocalVector(dm, &dpdy_local));

  PetscCall(DMRestoreLocalVector(dm, &u_tilde));
  PetscCall(DMRestoreLocalVector(dm, &v_tilde));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculatePressureCorrection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Vec     s;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(KSPSetComputeRHS(fsm->kspp, ComputeRHSPprime2d_Private, ns));
  PetscCall(KSPSetComputeOperators(fsm->kspp, ComputeOperatorPprime2d_Private, ns));
  PetscCall(KSPSolve(fsm->kspp, NULL, NULL));
  PetscCall(KSPGetSolution(fsm->kspp, &s));
  PetscCall(DMGlobalToLocal(dm, s, INSERT_VALUES, fsm->p_prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMUpdate2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  const PetscReal rho = ns->rho;
  const PetscReal mu  = ns->mu;
  const PetscReal dt  = ns->dt;

  Vec u = fsm->v[0], v = fsm->v[1];
  Vec u_star = fsm->v_star[0], v_star = fsm->v_star[1];
  Vec fv      = fsm->fv;
  Vec fv_star = fsm->fv_star;
  Vec p       = fsm->p_half;
  Vec p_prev  = fsm->p_half_prev;
  Vec p_full  = fsm->p;
  Vec p_prime = fsm->p_prime;

  PetscInt             M, N, x, y, m, n;
  PetscScalar       ***arru, ***arrv, ***arrp, ***arrp_prev, ***arrp_full, ***arrUV;
  const PetscScalar ***arru_star, ***arrv_star, ***arrp_prime, ***arrUV_star;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          dppdxl, dppdxr, dppdyd, dppdyu, dppdx, dppdy, d2ppdx2, d2ppdy2;
  PetscScalar          h1, h2, A;
  PetscInt             i, j;

  NSBoundaryCondition bcleft, bcright, bcdown, bcup;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecCopy(fsm->N[1], fsm->N_prev[1]));

  PetscCall(DMStagVecGetArray(dm, u, &arru));
  PetscCall(DMStagVecGetArray(dm, v, &arrv));
  PetscCall(DMStagVecGetArray(dm, p, &arrp));
  PetscCall(DMStagVecGetArray(dm, p_prev, &arrp_prev));
  PetscCall(DMStagVecGetArray(dm, p_full, &arrp_full));
  PetscCall(DMStagVecGetArray(fdm, fv, &arrUV));
  PetscCall(DMStagVecGetArrayRead(dm, u_star, &arru_star));
  PetscCall(DMStagVecGetArrayRead(dm, v_star, &arrv_star));
  PetscCall(DMStagVecGetArrayRead(dm, p_prime, &arrp_prime));
  PetscCall(DMStagVecGetArrayRead(fdm, fv_star, &arrUV_star));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(GetBoundaryConditions2d_Private(ns, &bcleft, &bcright, &bcdown, &bcup));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      if (i == 0) {
        /* Left boundary */
        switch (bcleft.type) {
        case NS_BC_VELOCITY:
          h1      = arrcx[i][ielemc] - arrcx[i][iprevc];
          h2      = arrcx[i + 1][ielemc] - arrcx[i][ielemc];
          A       = 2. * h1 / (h2 * (2. * h1 + h2));
          dppdx   = -A * arrp_prime[j][i][ielem] + A * arrp_prime[j][i + 1][ielem];
          dppdxl  = 0.;
          dppdxr  = (arrp_prime[j][i + 1][ielem] - arrp_prime[j][i][ielem]) / (arrcx[i + 1][ielemc] - arrcx[i][ielemc]);
          d2ppdx2 = (dppdxr - dppdxl) / (arrcx[i][inextc] - arrcx[i][iprevc]);
          break;
        case NS_BC_PERIODIC:
          dppdx   = (arrp_prime[j][i + 1][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
          dppdxl  = (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          dppdxr  = (arrp_prime[j][i + 1][ielem] - arrp_prime[j][i][ielem]) / (arrcx[i + 1][ielemc] - arrcx[i][ielemc]);
          d2ppdx2 = (dppdxr - dppdxl) / (arrcx[i][inextc] - arrcx[i][iprevc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary: %s", NSBoundaryConditionTypes[bcleft.type]);
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcright.type) {
        case NS_BC_VELOCITY:
          h1      = arrcx[i][inextc] - arrcx[i][ielemc];
          h2      = arrcx[i][ielemc] - arrcx[i - 1][ielemc];
          A       = 2. * h1 / (h2 * (2. * h1 + h2));
          dppdx   = -A * arrp_prime[j][i - 1][ielem] + A * arrp_prime[j][i][ielem];
          dppdxl  = (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          dppdxr  = 0.;
          d2ppdx2 = (dppdxr - dppdxl) / (arrcx[i][inextc] - arrcx[i][iprevc]);
          break;
        case NS_BC_PERIODIC:
          dppdx   = (arrp_prime[j][i + 1][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
          dppdxl  = (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          dppdxr  = (arrp_prime[j][i + 1][ielem] - arrp_prime[j][i][ielem]) / (arrcx[i + 1][ielemc] - arrcx[i][ielemc]);
          d2ppdx2 = (dppdxr - dppdxl) / (arrcx[i][inextc] - arrcx[i][iprevc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary: %s", NSBoundaryConditionTypes[bcright.type]);
        }
      } else {
        dppdx   = (arrp_prime[j][i + 1][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
        dppdxl  = (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        dppdxr  = (arrp_prime[j][i + 1][ielem] - arrp_prime[j][i][ielem]) / (arrcx[i + 1][ielemc] - arrcx[i][ielemc]);
        d2ppdx2 = (dppdxr - dppdxl) / (arrcx[i][inextc] - arrcx[i][iprevc]);
      }

      if (j == 0) {
        /* Down boundary */
        switch (bcdown.type) {
        case NS_BC_VELOCITY:
          h1      = arrcy[j][ielemc] - arrcy[j][iprevc];
          h2      = arrcy[j + 1][ielemc] - arrcy[j][ielemc];
          A       = 2. * h1 / (h2 * (2. * h1 + h2));
          dppdy   = -A * arrp_prime[j][i][ielem] + A * arrp_prime[j + 1][i][ielem];
          dppdyd  = 0.;
          dppdyu  = (arrp_prime[j + 1][i][ielem] - arrp_prime[j][i][ielem]) / (arrcy[j + 1][ielemc] - arrcy[j][ielemc]);
          d2ppdy2 = (dppdyu - dppdyd) / (arrcy[j][inextc] - arrcy[j][iprevc]);
          break;
        case NS_BC_PERIODIC:
          dppdy   = (arrp_prime[j + 1][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
          dppdyd  = (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          dppdyu  = (arrp_prime[j + 1][i][ielem] - arrp_prime[j][i][ielem]) / (arrcy[j + 1][ielemc] - arrcy[j][ielemc]);
          d2ppdy2 = (dppdyu - dppdyd) / (arrcy[j][inextc] - arrcy[j][iprevc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary: %s", NSBoundaryConditionTypes[bcdown.type]);
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcup.type) {
        case NS_BC_VELOCITY:
          h1      = arrcy[j][inextc] - arrcy[j][ielemc];
          h2      = arrcy[j][ielemc] - arrcy[j - 1][ielemc];
          A       = 2. * h1 / (h2 * (2. * h1 + h2));
          dppdy   = A * arrp_prime[j - 1][i][ielem] - A * arrp_prime[j][i][ielem];
          dppdyd  = (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          dppdyu  = 0.;
          d2ppdy2 = (dppdyu - dppdyd) / (arrcy[j][inextc] - arrcy[j][iprevc]);
          break;
        case NS_BC_PERIODIC:
          dppdy   = (arrp_prime[j + 1][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
          dppdyd  = (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          dppdyu  = (arrp_prime[j + 1][i][ielem] - arrp_prime[j][i][ielem]) / (arrcy[j + 1][ielemc] - arrcy[j][ielemc]);
          d2ppdy2 = (dppdyu - dppdyd) / (arrcy[j][inextc] - arrcy[j][iprevc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary: %s", NSBoundaryConditionTypes[bcup.type]);
        }
      } else {
        dppdy   = (arrp_prime[j + 1][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
        dppdyd  = (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
        dppdyu  = (arrp_prime[j + 1][i][ielem] - arrp_prime[j][i][ielem]) / (arrcy[j + 1][ielemc] - arrcy[j][ielemc]);
        d2ppdy2 = (dppdyu - dppdyd) / (arrcy[j][inextc] - arrcy[j][iprevc]);
      }

      arru[j][i][ielem]      = arru_star[j][i][ielem] - dt / rho * dppdx;
      arrv[j][i][ielem]      = arrv_star[j][i][ielem] - dt / rho * dppdy;
      arrp[j][i][ielem]      = arrp_prev[j][i][ielem] + arrp_prime[j][i][ielem] - 0.5 * mu * dt / rho * (d2ppdx2 + d2ppdy2);
      arrp_full[j][i][ielem] = 1.5 * arrp[j][i][ielem] - 0.5 * arrp_prev[j][i][ielem];
    }

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + 1; ++i) {
      if (i == 0) {
        /* Left boundary */
        switch (bcleft.type) {
        case NS_BC_VELOCITY:
          dppdx = 0.;
          break;
        case NS_BC_PERIODIC:
          dppdx = (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary: %s", NSBoundaryConditionTypes[bcleft.type]);
        }
      } else if (i == M) {
        /* Right boundary */
        switch (bcright.type) {
        case NS_BC_VELOCITY:
          dppdx = 0.;
          break;
        case NS_BC_PERIODIC:
          dppdx = (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary: %s", NSBoundaryConditionTypes[bcright.type]);
        }
      } else {
        dppdx = (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
      }
      arrUV[j][i][ileft] = arrUV_star[j][i][ileft] - dt / rho * dppdx;
    }

  for (j = y; j < y + n + 1; ++j)
    for (i = x; i < x + m; ++i) {
      if (j == 0) {
        /* Down boundary */
        switch (bcdown.type) {
        case NS_BC_VELOCITY:
          dppdy = 0.;
          break;
        case NS_BC_PERIODIC:
          dppdy = (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary: %s", NSBoundaryConditionTypes[bcdown.type]);
        }
      } else if (j == N) {
        /* Up boundary */
        switch (bcup.type) {
        case NS_BC_VELOCITY:
          dppdy = 0.;
          break;
        case NS_BC_PERIODIC:
          dppdy = (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary: %s", NSBoundaryConditionTypes[bcup.type]);
        }
      } else {
        dppdy = (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
      }
      arrUV[j][i][idown] = arrUV_star[j][i][idown] - dt / rho * dppdy;
    }

  PetscCall(DMStagVecRestoreArray(dm, u, &arru));
  PetscCall(DMStagVecRestoreArray(dm, v, &arrv));
  PetscCall(DMStagVecRestoreArray(dm, p, &arrp));
  PetscCall(DMStagVecRestoreArray(dm, p_prev, &arrp_prev));
  PetscCall(DMStagVecRestoreArray(dm, p_full, &arrp_full));
  PetscCall(DMStagVecRestoreArray(fdm, fv, &arrUV));
  PetscCall(DMStagVecRestoreArrayRead(dm, u_star, &arru_star));
  PetscCall(DMStagVecRestoreArrayRead(dm, v_star, &arrv_star));
  PetscCall(DMStagVecRestoreArrayRead(dm, p_prime, &arrp_prime));
  PetscCall(DMStagVecRestoreArrayRead(fdm, fv_star, &arrUV_star));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMLocalToLocalBegin(dm, u, INSERT_VALUES, u));
  PetscCall(DMLocalToLocalEnd(dm, u, INSERT_VALUES, u));
  PetscCall(DMLocalToLocalBegin(dm, v, INSERT_VALUES, v));
  PetscCall(DMLocalToLocalEnd(dm, v, INSERT_VALUES, v));
  PetscCall(DMLocalToLocalBegin(dm, p, INSERT_VALUES, p));
  PetscCall(DMLocalToLocalEnd(dm, p, INSERT_VALUES, p));

  PetscCall(NSFSMCalculateConvection2d_Cart_Internal(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSUStar2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  Vec u       = fsm->v[0];
  Vec Nu      = fsm->N[0];
  Vec Nu_prev = fsm->N_prev[0];
  Vec p       = fsm->p_half;

  PetscInt             M, N, x, y, m, n;
  DMStagStencil        row;
  PetscScalar          valb;
  const PetscScalar ***arrNu, ***arrNu_prev;
  PetscInt             ielem;
  PetscInt             i, j;

  Vec            p_global, dpdx_global, dpdx_local;
  PetscScalar ***arrdpdx;

  Vec            u_global, helm_u_global, helm_u_local, valbc_global, valbc_local;
  PetscScalar ***arrhelm_u, ***arrvalbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &dpdx_global));
  PetscCall(DMGetLocalVector(dm, &dpdx_local));
  PetscCall(DMGetGlobalVector(dm, &u_global));
  PetscCall(DMGetGlobalVector(dm, &helm_u_global));
  PetscCall(DMGetLocalVector(dm, &helm_u_local));
  PetscCall(DMGetGlobalVector(dm, &valbc_global));
  PetscCall(DMGetLocalVector(dm, &valbc_local));

  PetscCall(DMStagVecGetArrayRead(dm, Nu, &arrNu));
  PetscCall(DMStagVecGetArrayRead(dm, Nu_prev, &arrNu_prev));

  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));

  PetscCall(DMLocalToGlobal(dm, p, INSERT_VALUES, p_global));
  PetscCall(MatMult(fsm->grad_p[0], p_global, dpdx_global));
  PetscCall(DMGlobalToLocal(dm, dpdx_global, INSERT_VALUES, dpdx_local));
  PetscCall(DMStagVecGetArray(dm, dpdx_local, &arrdpdx));

  PetscCall(DMLocalToGlobal(dm, u, INSERT_VALUES, u_global));
  PetscCall(MatMult(fsm->helm_v, u_global, helm_u_global));
  PetscCall(NSFSMAddVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, helm_u_global));
  PetscCall(DMGlobalToLocal(dm, helm_u_global, INSERT_VALUES, helm_u_local));
  PetscCall(DMStagVecGetArray(dm, helm_u_local, &arrhelm_u));

  PetscCall(VecSet(valbc_global, 0.));
  PetscCall(NSFSMAddVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, valbc_global));
  PetscCall(DMGlobalToLocal(dm, valbc_global, INSERT_VALUES, valbc_local));
  PetscCall(DMStagVecGetArray(dm, valbc_local, &arrvalbc));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;
      valb  = arrhelm_u[j][i][ielem] - ns->dt * (1.5 * arrNu[j][i][ielem] - 0.5 * arrNu_prev[j][i][ielem]) - ns->dt / ns->rho * arrdpdx[j][i][ielem] + arrvalbc[j][i][ielem];
      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(dm, Nu, &arrNu));
  PetscCall(DMStagVecRestoreArrayRead(dm, Nu_prev, &arrNu_prev));

  PetscCall(DMStagVecRestoreArray(dm, dpdx_local, &arrdpdx));
  PetscCall(DMStagVecRestoreArray(dm, helm_u_local, &arrhelm_u));
  PetscCall(DMStagVecRestoreArray(dm, valbc_local, &arrvalbc));

  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &dpdx_global));
  PetscCall(DMRestoreLocalVector(dm, &dpdx_local));
  PetscCall(DMRestoreGlobalVector(dm, &u_global));
  PetscCall(DMRestoreGlobalVector(dm, &helm_u_global));
  PetscCall(DMRestoreLocalVector(dm, &helm_u_local));
  PetscCall(DMRestoreGlobalVector(dm, &valbc_global));
  PetscCall(DMRestoreLocalVector(dm, &valbc_local));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSVStar2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  Vec v       = fsm->v[1];
  Vec Nv      = fsm->N[1];
  Vec Nv_prev = fsm->N_prev[1];
  Vec p       = fsm->p_half;

  PetscInt             M, N, x, y, m, n;
  DMStagStencil        row;
  PetscScalar          valb;
  const PetscScalar ***arrNv, ***arrNv_prev;
  PetscInt             ielem;
  PetscInt             i, j;

  Vec            p_global, dpdy_global, dpdy_local;
  PetscScalar ***arrdpdy;

  Vec            v_global, helm_v_global, helm_v_local, valbc_global, valbc_local;
  PetscScalar ***arrhelm_v, ***arrvalbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &dpdy_global));
  PetscCall(DMGetLocalVector(dm, &dpdy_local));
  PetscCall(DMGetGlobalVector(dm, &v_global));
  PetscCall(DMGetGlobalVector(dm, &helm_v_global));
  PetscCall(DMGetLocalVector(dm, &helm_v_local));
  PetscCall(DMGetGlobalVector(dm, &valbc_global));
  PetscCall(DMGetLocalVector(dm, &valbc_local));

  PetscCall(DMStagVecGetArrayRead(dm, Nv, &arrNv));
  PetscCall(DMStagVecGetArrayRead(dm, Nv_prev, &arrNv_prev));

  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));

  PetscCall(DMLocalToGlobal(dm, p, INSERT_VALUES, p_global));
  PetscCall(MatMult(fsm->grad_p[1], p_global, dpdy_global));
  PetscCall(DMGlobalToLocal(dm, dpdy_global, INSERT_VALUES, dpdy_local));
  PetscCall(DMStagVecGetArray(dm, dpdy_local, &arrdpdy));

  PetscCall(DMLocalToGlobal(dm, v, INSERT_VALUES, v_global));
  PetscCall(MatMult(fsm->helm_v, v_global, helm_v_global));
  PetscCall(NSFSMAddVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, helm_v_global));
  PetscCall(DMGlobalToLocal(dm, helm_v_global, INSERT_VALUES, helm_v_local));
  PetscCall(DMStagVecGetArray(dm, helm_v_local, &arrhelm_v));

  PetscCall(VecSet(valbc_global, 0.));
  PetscCall(NSFSMAddVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, valbc_global));
  PetscCall(DMGlobalToLocal(dm, valbc_global, INSERT_VALUES, valbc_local));
  PetscCall(DMStagVecGetArray(dm, valbc_local, &arrvalbc));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;
      valb  = arrhelm_v[j][i][ielem] - ns->dt * (1.5 * arrNv[j][i][ielem] - 0.5 * arrNv_prev[j][i][ielem]) - ns->dt / ns->rho * arrdpdy[j][i][ielem] + arrvalbc[j][i][ielem];
      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(dm, Nv, &arrNv));
  PetscCall(DMStagVecRestoreArrayRead(dm, Nv_prev, &arrNv_prev));

  PetscCall(DMStagVecRestoreArray(dm, dpdy_local, &arrdpdy));
  PetscCall(DMStagVecRestoreArray(dm, helm_v_local, &arrhelm_v));
  PetscCall(DMStagVecRestoreArray(dm, valbc_local, &arrvalbc));

  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &dpdy_global));
  PetscCall(DMRestoreLocalVector(dm, &dpdy_local));
  PetscCall(DMRestoreGlobalVector(dm, &v_global));
  PetscCall(DMRestoreGlobalVector(dm, &helm_v_global));
  PetscCall(DMRestoreLocalVector(dm, &helm_v_local));
  PetscCall(DMRestoreGlobalVector(dm, &valbc_global));
  PetscCall(DMRestoreLocalVector(dm, &valbc_local));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSPprime2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  const PetscReal rho = ns->rho;
  const PetscReal dt  = ns->dt;

  Vec fv_star = fsm->fv_star;

  MPI_Comm             comm;
  PetscInt             M, N, x, y, m, n;
  DMStagStencil        row;
  PetscScalar          valb;
  const PetscScalar ***arrUV_star;
  const PetscScalar  **arrcx, **arrcy;
  PetscScalar          ap;
  PetscScalar          A, B;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          divUV_star;
  MatNullSpace         nullspace;
  PetscInt             i, j;

  NSBoundaryCondition bcleft, bcright, bcdown, bcup;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagVecGetArrayRead(fdm, fv_star, &arrUV_star));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(GetBoundaryConditions2d_Private(ns, &bcleft, &bcright, &bcdown, &bcup));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;
      ap    = 0.;

      if (i == 0) {
        /* Left boundary */
        switch (bcleft.type) {
        case NS_BC_VELOCITY:
          B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          ap += B;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          ap += A + B;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary: %s", NSBoundaryConditionTypes[bcleft.type]);
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcright.type) {
        case NS_BC_VELOCITY:
          A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          ap += A;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          ap += A + B;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary: %s", NSBoundaryConditionTypes[bcright.type]);
        }
      } else {
        A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
        B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
        ap += A + B;
      }

      if (j == 0) {
        /* Down boundary */
        switch (bcdown.type) {
        case NS_BC_VELOCITY:
          B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          ap += B;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          ap += A + B;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary: %s", NSBoundaryConditionTypes[bcdown.type]);
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcup.type) {
        case NS_BC_VELOCITY:
          A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          ap += A;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          ap += A + B;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary: %s", NSBoundaryConditionTypes[bcup.type]);
        }
      } else {
        A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
        B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
        ap += A + B;
      }

      divUV_star = (arrUV_star[j][i + 1][ileft] - arrUV_star[j][i][ileft]) / (arrcx[i][inextc] - arrcx[i][iprevc]) //
                 + (arrUV_star[j + 1][i][idown] - arrUV_star[j][i][idown]) / (arrcy[j][inextc] - arrcy[j][iprevc]);
      valb = -rho / dt * divUV_star / ap;

      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(fdm, fv_star, &arrUV_star));

  // TODO: below is only for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  (void)J;

  NS ns = (NS)ctx;
  DM dm;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(dm, ns->bcs, 1., -0.5 * ns->mu * ns->dt / ns->rho, Jpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorPprime2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS ns = (NS)ctx;
  (void)ns;

  MPI_Comm            comm;
  DM                  dm;
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[5];
  PetscScalar         v[5];
  PetscInt            ncols;
  MatNullSpace        nullspace;
  const PetscScalar **arrcx, **arrcy;
  PetscScalar         A, B;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j;

  NSBoundaryCondition bcleft, bcright, bcdown, bcup;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(GetBoundaryConditions2d_Private(ns, &bcleft, &bcright, &bcdown, &bcup));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (i = 0; i < 5; ++i) {
    col[i].loc = DMSTAG_ELEMENT;
    col[i].c   = 0;
  }

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i    = i;
      row.j    = j;
      col[0].i = i;
      col[0].j = j;
      v[0]     = 0.;
      v[1]     = 0.;
      v[2]     = 0.;
      v[3]     = 0.;
      v[4]     = 0.;
      ncols    = 1;

      if (i == 0) {
        /* Left boundary */
        switch (bcleft.type) {
        case NS_BC_VELOCITY:
          B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          v[0] += B;
          col[ncols].i = i + 1;
          col[ncols].j = j;
          v[ncols]     = -B;
          ++ncols;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          v[0] += A + B;
          col[ncols].i = i - 1;
          col[ncols].j = j;
          v[ncols]     = -A;
          ++ncols;
          col[ncols].i = i + 1;
          col[ncols].j = j;
          v[ncols]     = -B;
          ++ncols;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary: %s", NSBoundaryConditionTypes[bcleft.type]);
        }
      } else if (i == M - 1) {
        /* Right boundary */
        switch (bcright.type) {
        case NS_BC_VELOCITY:
          A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          v[0] += A;
          col[ncols].i = i - 1;
          col[ncols].j = j;
          v[ncols]     = -A;
          ++ncols;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
          v[0] += A + B;
          col[ncols].i = i - 1;
          col[ncols].j = j;
          v[ncols]     = -A;
          ++ncols;
          col[ncols].i = i + 1;
          col[ncols].j = j;
          v[ncols]     = -B;
          ++ncols;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary: %s", NSBoundaryConditionTypes[bcright.type]);
        }
      } else {
        A = 1. / ((arrcx[i][ielemc] - arrcx[i - 1][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
        B = 1. / ((arrcx[i + 1][ielemc] - arrcx[i][ielemc]) * (arrcx[i][inextc] - arrcx[i][iprevc]));
        v[0] += A + B;
        col[ncols].i = i - 1;
        col[ncols].j = j;
        v[ncols]     = -A;
        ++ncols;
        col[ncols].i = i + 1;
        col[ncols].j = j;
        v[ncols]     = -B;
        ++ncols;
      }

      if (j == 0) {
        /* Down boundary */
        switch (bcdown.type) {
        case NS_BC_VELOCITY:
          B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          v[0] += B;
          col[ncols].i = i;
          col[ncols].j = j + 1;
          v[ncols]     = -B;
          ++ncols;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          v[0] += A + B;
          col[ncols].i = i;
          col[ncols].j = j - 1;
          v[ncols]     = -A;
          ++ncols;
          col[ncols].i = i;
          col[ncols].j = j + 1;
          v[ncols]     = -B;
          ++ncols;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary: %s", NSBoundaryConditionTypes[bcdown.type]);
        }
      } else if (j == N - 1) {
        /* Up boundary */
        switch (bcup.type) {
        case NS_BC_VELOCITY:
          A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          v[0] += A;
          col[ncols].i = i;
          col[ncols].j = j - 1;
          v[ncols]     = -A;
          ++ncols;
          break;
        case NS_BC_PERIODIC:
          A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
          v[0] += A + B;
          col[ncols].i = i;
          col[ncols].j = j - 1;
          v[ncols]     = -A;
          ++ncols;
          col[ncols].i = i;
          col[ncols].j = j + 1;
          v[ncols]     = -B;
          ++ncols;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary: %s", NSBoundaryConditionTypes[bcup.type]);
        }
      } else {
        A = 1. / ((arrcy[j][ielemc] - arrcy[j - 1][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
        B = 1. / ((arrcy[j + 1][ielemc] - arrcy[j][ielemc]) * (arrcy[j][inextc] - arrcy[j][iprevc]));
        v[0] += A + B;
        col[ncols].i = i;
        col[ncols].j = j - 1;
        v[ncols]     = -A;
        ++ncols;
        col[ncols].i = i;
        col[ncols].j = j + 1;
        v[ncols]     = -B;
        ++ncols;
      }

      v[1] /= v[0];
      v[2] /= v[0];
      v[3] /= v[0];
      v[4] /= v[0];
      v[0] = 1.;

      PetscCall(DMStagMatSetValuesStencil(dm, Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

  // TODO: below is temporary for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(J, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS ns, PetscViewer v)
{
  PetscBool iscgns;
  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) PetscCall(NSViewSolution_FSM_Cart_CGNS_Internal(ns, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
