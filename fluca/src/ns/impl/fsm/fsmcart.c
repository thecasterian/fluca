#include <fluca/private/flucaviewercgnsimpl.h>
#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

static PetscErrorCode ComputeRHSUStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSVStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSPprime2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeOperatorPprime2d_Private(KSP, Mat, Mat, void *);

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
  PetscScalar          wx_left, wx_right, wy_down, wy_up, wx, wy;
  PetscInt             i, j;

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

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      // TODO: below is temporary for cavity flow
      if (i == 0) {
        /* Left wall. */
        arru_interp[j][i][ileft] = 0.0;
        arrv_interp[j][i][ileft] = 0.0;
      } else if (i == M) {
        /* Right wall. */
        arru_interp[j][i][ileft] = 0.0;
        arrv_interp[j][i][ileft] = 0.0;
      } else {
        wx_left                  = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
        wx_right                 = arrcx[i][inextc] - arrcx[i][iprevc];
        arru_interp[j][i][ileft] = (wx_right * arru[j][i - 1][ielem] + wx_left * arru[j][i][ielem]) / (wx_left + wx_right);
        arrv_interp[j][i][ileft] = (wx_right * arrv[j][i - 1][ielem] + wx_left * arrv[j][i][ielem]) / (wx_left + wx_right);
      }
    }
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      // TODO: below is temporary for cavity flow
      if (j == 0) {
        /* Bottom wall. */
        arru_interp[j][i][idown] = 0.0;
        arrv_interp[j][i][idown] = 0.0;
      } else if (j == N) {
        /* Top wall. */
        arru_interp[j][i][idown] = 1.0;
        arrv_interp[j][i][idown] = 0.0;
      } else {
        wy_down                  = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
        wy_up                    = arrcy[j][inextc] - arrcy[j][iprevc];
        arru_interp[j][i][idown] = (wy_up * arru[j - 1][i][ielem] + wy_down * arru[j][i][ielem]) / (wy_down + wy_up);
        arrv_interp[j][i][idown] = (wy_up * arrv[j - 1][i][ielem] + wy_down * arrv[j][i][ielem]) / (wy_down + wy_up);
      }
    }

  PetscCall(DMLocalToLocalBegin(fdm, u_interp, INSERT_VALUES, u_interp));
  PetscCall(DMLocalToLocalEnd(fdm, u_interp, INSERT_VALUES, u_interp));
  PetscCall(DMLocalToLocalBegin(fdm, v_interp, INSERT_VALUES, v_interp));
  PetscCall(DMLocalToLocalEnd(fdm, v_interp, INSERT_VALUES, v_interp));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx                 = arrcx[i][inextc] - arrcx[i][iprevc];
      wy                 = arrcy[j][inextc] - arrcy[j][iprevc];
      arrNu[j][i][ielem] = (arrUV[j][i][iright] * arru_interp[j][i][iright] - arrUV[j][i][ileft] * arru_interp[j][i][ileft]) / wx + (arrUV[j][i][iup] * arru_interp[j][i][iup] - arrUV[j][i][idown] * arru_interp[j][i][idown]) / wy;
      arrNv[j][i][ielem] = (arrUV[j][i][iright] * arrv_interp[j][i][iright] - arrUV[j][i][ileft] * arrv_interp[j][i][ileft]) / wx + (arrUV[j][i][iup] * arrv_interp[j][i][iup] - arrUV[j][i][idown] * arrv_interp[j][i][idown]) / wy;
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

  const PetscReal rho = ns->rho;
  const PetscReal dt  = ns->dt;

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
  PetscScalar          pw, pe, ps, pn, U_tilde, V_tilde, dpdx, dpdy;
  PetscScalar          wx_left, wx_right, wy_down, wy_up;
  PetscInt             d, i, j;

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

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) {
        pw = arrp[j][i][ielem];
      } else {
        pw = arrp[j][i - 1][ielem];
      }
      /* Right wall. */
      if (i == M - 1) {
        pe = arrp[j][i][ielem];
      } else {
        pe = arrp[j][i + 1][ielem];
      }
      /* Bottom wall. */
      if (j == 0) {
        ps = arrp[j][i][ielem];
      } else {
        ps = arrp[j - 1][i][ielem];
      }
      /* Top wall. */
      if (j == N - 1) {
        pn = arrp[j][i][ielem];
      } else {
        pn = arrp[j + 1][i][ielem];
      }

      arru_tilde[j][i][ielem] = arru_star[j][i][ielem] + dt / rho * (pe - pw) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
      arrv_tilde[j][i][ielem] = arrv_star[j][i][ielem] + dt / rho * (pn - ps) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
    }

  PetscCall(DMLocalToLocalBegin(dm, u_tilde, INSERT_VALUES, u_tilde));
  PetscCall(DMLocalToLocalEnd(dm, u_tilde, INSERT_VALUES, u_tilde));
  PetscCall(DMLocalToLocalBegin(dm, v_tilde, INSERT_VALUES, v_tilde));
  PetscCall(DMLocalToLocalEnd(dm, v_tilde, INSERT_VALUES, v_tilde));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) {
        arrUV_star[j][i][ileft] = 0.0;
      }
      /* Right wall. */
      else if (i == M) {
        arrUV_star[j][i][ileft] = 0.0;
      } else {
        wx_left                 = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
        wx_right                = arrcx[i][inextc] - arrcx[i][iprevc];
        U_tilde                 = (wx_right * arru_tilde[j][i - 1][ielem] + wx_left * arru_tilde[j][i][ielem]) / (wx_left + wx_right);
        dpdx                    = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        arrUV_star[j][i][ileft] = U_tilde - dt / rho * dpdx;
      }
    }
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      // TODO: below is temporary for cavity flow
      /* Bottom wall. */
      if (j == 0) {
        arrUV_star[j][i][idown] = 0.0;
      }
      /* Top wall. */
      else if (j == N) {
        arrUV_star[j][i][idown] = 0.0;
      } else {
        wy_down                 = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
        wy_up                   = arrcy[j][inextc] - arrcy[j][iprevc];
        V_tilde                 = (wy_up * arrv_tilde[j - 1][i][ielem] + wy_down * arrv_tilde[j][i][ielem]) / (wy_down + wy_up);
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

  PetscCall(DMLocalToLocalBegin(fdm, fv_star, INSERT_VALUES, fv_star));
  PetscCall(DMLocalToLocalEnd(fdm, fv_star, INSERT_VALUES, fv_star));

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
  Vec p_prime = fsm->p_prime;

  PetscInt             M, N, x, y, m, n;
  PetscScalar       ***arru, ***arrv, ***arrp, ***arrUV;
  const PetscScalar ***arru_star, ***arrv_star, ***arrp_prime, ***arrUV_star;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          ppp, ppw, ppe, pps, ppn, pplap;
  PetscScalar          wx, wy, aw, ae, as, an;
  PetscInt             i, j;

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

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx = arrcx[i][inextc] - arrcx[i][iprevc];
      wy = arrcy[j][inextc] - arrcy[j][iprevc];
      aw = 1.0 / (wx * (arrcx[i][ielemc] - arrcx[i - 1][ielemc]));
      ae = 1.0 / (wx * (arrcx[i + 1][ielemc] - arrcx[i][ielemc]));
      as = 1.0 / (wy * (arrcy[j][ielemc] - arrcy[j - 1][ielemc]));
      an = 1.0 / (wy * (arrcy[j + 1][ielemc] - arrcy[j][ielemc]));

      ppp = arrp_prime[j][i][ielem];

      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) ppw = arrp_prime[j][i][ielem];
      else ppw = arrp_prime[j][i - 1][ielem];
      /* Right wall. */
      if (i == M - 1) ppe = arrp_prime[j][i][ielem];
      else ppe = arrp_prime[j][i + 1][ielem];
      /* Bottom wall. */
      if (j == 0) pps = arrp_prime[j][i][ielem];
      else pps = arrp_prime[j - 1][i][ielem];
      /* Top wall. */
      if (j == N - 1) ppn = arrp_prime[j][i][ielem];
      else ppn = arrp_prime[j + 1][i][ielem];

      pplap = aw * ppw + ae * ppe + as * pps + an * ppn - (aw + ae + as + an) * ppp;

      arru[j][i][ielem] = arru_star[j][i][ielem] - dt / rho * (ppe - ppw) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
      arrv[j][i][ielem] = arrv_star[j][i][ielem] - dt / rho * (ppn - pps) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
      arrp[j][i][ielem] += ppp - 0.5 * mu * dt / rho * pplap;
    }

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + 1; ++i) {
      /* Left wall. */
      if (i == 0) arrUV[j][i][ileft] = 0.0;
      /* Right wall. */
      else if (i == M) arrUV[j][i][ileft] = 0.0;
      else arrUV[j][i][ileft] = arrUV_star[j][i][ileft] - dt / rho * (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
    }

  for (j = y; j < y + n + 1; ++j)
    for (i = x; i < x + m; ++i) {
      /* Bottom wall. */
      if (j == 0) arrUV[j][i][idown] = 0.0;
      /* Top wall. */
      else if (j == N) arrUV[j][i][idown] = 0.0;
      else arrUV[j][i][idown] = arrUV_star[j][i][idown] - dt / rho * (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
    }

  PetscCall(DMStagVecRestoreArray(dm, u, &arru));
  PetscCall(DMStagVecRestoreArray(dm, v, &arrv));
  PetscCall(DMStagVecRestoreArray(dm, p, &arrp));
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

  const PetscReal rho = ns->rho;
  const PetscReal mu  = ns->mu;
  const PetscReal dt  = ns->dt;

  Vec u = fsm->v[0], v = fsm->v[1];
  Vec Nu      = fsm->N[0];
  Vec Nu_prev = fsm->N_prev[0];
  Vec p       = fsm->p_half;

  PetscInt             M, N, x, y, m, n;
  DMStagStencil        row;
  PetscScalar          valb;
  const PetscScalar ***arrNu, ***arrNu_prev, ***arru, ***arrv, ***arrp;
  const PetscScalar  **arrcx, **arrcy;
  PetscScalar          up, uw, ue, us, un, pw, pe, dpdx, ulap;
  PetscScalar          wx, wy, aw, ae, as, an;
  PetscInt             ielem, iprevc, inextc, ielemc;
  PetscInt             i, j;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagVecGetArrayRead(dm, Nu, &arrNu));
  PetscCall(DMStagVecGetArrayRead(dm, Nu_prev, &arrNu_prev));
  PetscCall(DMStagVecGetArrayRead(dm, u, &arru));
  PetscCall(DMStagVecGetArrayRead(dm, v, &arrv));
  PetscCall(DMStagVecGetArrayRead(dm, p, &arrp));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx = arrcx[i][inextc] - arrcx[i][iprevc];
      wy = arrcy[j][inextc] - arrcy[j][iprevc];
      aw = 1.0 / (wx * (arrcx[i][ielemc] - arrcx[i - 1][ielemc]));
      ae = 1.0 / (wx * (arrcx[i + 1][ielemc] - arrcx[i][ielemc]));
      as = 1.0 / (wy * (arrcy[j][ielemc] - arrcy[j - 1][ielemc]));
      an = 1.0 / (wy * (arrcy[j + 1][ielemc] - arrcy[j][ielemc]));

      row.i = i;
      row.j = j;

      up = arru[j][i][ielem];
      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) {
        uw = -arru[j][i][ielem];
        pw = arrp[j][i][ielem];
      } else {
        uw = arru[j][i - 1][ielem];
        pw = arrp[j][i - 1][ielem];
      }
      /* Right wall. */
      if (i == M - 1) {
        ue = -arru[j][i][ielem];
        pe = arrp[j][i][ielem];
      } else {
        ue = arru[j][i + 1][ielem];
        pe = arrp[j][i + 1][ielem];
      }
      /* Bottom wall. */
      if (j == 0) us = -arru[j][i][ielem];
      else us = arru[j - 1][i][ielem];
      /* Top wall. */
      if (j == N - 1) un = 2.0 - arru[j][i][ielem];
      else un = arru[j + 1][i][ielem];

      /* Pressure gradient. */
      dpdx = (pe - pw) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
      /* Laplacian of u. */
      ulap = aw * uw + ae * ue + as * us + an * un - (aw + ae + as + an) * up;

      valb = up - 1.5 * dt * arrNu[j][i][ielem] + 0.5 * dt * arrNu_prev[j][i][ielem] - dt / rho * dpdx + 0.5 * mu * dt / rho * ulap;
      // TODO: below is temporary for cavity flow
      if (j == N - 1) valb += mu * dt / rho * an;

      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(dm, Nu, &arrNu));
  PetscCall(DMStagVecRestoreArrayRead(dm, Nu_prev, &arrNu_prev));
  PetscCall(DMStagVecRestoreArrayRead(dm, u, &arru));
  PetscCall(DMStagVecRestoreArrayRead(dm, v, &arrv));
  PetscCall(DMStagVecRestoreArrayRead(dm, p, &arrp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

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

  const PetscReal rho = ns->rho;
  const PetscReal mu  = ns->mu;
  const PetscReal dt  = ns->dt;

  Vec u = fsm->v[0], v = fsm->v[1];
  Vec Nv      = fsm->N[1];
  Vec Nv_prev = fsm->N_prev[1];
  Vec p       = fsm->p_half;

  PetscInt             M, N, x, y, m, n;
  DMStagStencil        row;
  PetscScalar          valb;
  const PetscScalar ***arrNv, ***arrNv_prev, ***arru, ***arrv, ***arrp;
  const PetscScalar  **arrcx, **arrcy;
  PetscScalar          vp, vw, ve, vs, vn, ps, pn, dpdy, vlap;
  PetscScalar          wx, wy, aw, ae, as, an;
  PetscInt             ielem, iprevc, inextc, ielemc;
  PetscInt             i, j;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagVecGetArrayRead(dm, Nv, &arrNv));
  PetscCall(DMStagVecGetArrayRead(dm, Nv_prev, &arrNv_prev));
  PetscCall(DMStagVecGetArrayRead(dm, u, &arru));
  PetscCall(DMStagVecGetArrayRead(dm, v, &arrv));
  PetscCall(DMStagVecGetArrayRead(dm, p, &arrp));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx = arrcx[i][inextc] - arrcx[i][iprevc];
      wy = arrcy[j][inextc] - arrcy[j][iprevc];
      aw = 1.0 / (wx * (arrcx[i][ielemc] - arrcx[i - 1][ielemc]));
      ae = 1.0 / (wx * (arrcx[i + 1][ielemc] - arrcx[i][ielemc]));
      as = 1.0 / (wy * (arrcy[j][ielemc] - arrcy[j - 1][ielemc]));
      an = 1.0 / (wy * (arrcy[j + 1][ielemc] - arrcy[j][ielemc]));

      row.i = i;
      row.j = j;

      vp = arrv[j][i][ielem];
      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) vw = -arrv[j][i][ielem];
      else vw = arrv[j][i - 1][ielem];
      /* Right wall. */
      if (i == M - 1) ve = -arrv[j][i][ielem];
      else ve = arrv[j][i + 1][ielem];
      /* Bottom wall. */
      if (j == 0) {
        vs = -arrv[j][i][ielem];
        ps = arrp[j][i][ielem];
      } else {
        vs = arrv[j - 1][i][ielem];
        ps = arrp[j - 1][i][ielem];
      }
      /* Top wall. */
      if (j == N - 1) {
        vn = -arrv[j][i][ielem];
        pn = arrp[j][i][ielem];
      } else {
        vn = arrv[j + 1][i][ielem];
        pn = arrp[j + 1][i][ielem];
      }

      /* Pressure gradient. */
      dpdy = (pn - ps) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
      /* Laplacian of v. */
      vlap = aw * vw + ae * ve + as * vs + an * vn - (aw + ae + as + an) * vp;

      valb = vp - 1.5 * dt * arrNv[j][i][ielem] + 0.5 * dt * arrNv_prev[j][i][ielem] - dt / rho * dpdy + 0.5 * mu * dt / rho * vlap;

      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(dm, Nv, &arrNv));
  PetscCall(DMStagVecRestoreArrayRead(dm, Nv_prev, &arrNv_prev));
  PetscCall(DMStagVecRestoreArrayRead(dm, u, &arru));
  PetscCall(DMStagVecRestoreArrayRead(dm, v, &arrv));
  PetscCall(DMStagVecRestoreArrayRead(dm, p, &arrp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

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
  PetscScalar          wx, wy, aw, ae, as, an;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          divUV_star;
  MatNullSpace         nullspace;
  PetscInt             i, j;

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

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx = arrcx[i][inextc] - arrcx[i][iprevc];
      wy = arrcy[j][inextc] - arrcy[j][iprevc];
      aw = 1.0 / (wx * (arrcx[i][ielemc] - arrcx[i - 1][ielemc]));
      ae = 1.0 / (wx * (arrcx[i + 1][ielemc] - arrcx[i][ielemc]));
      as = 1.0 / (wy * (arrcy[j][ielemc] - arrcy[j - 1][ielemc]));
      an = 1.0 / (wy * (arrcy[j + 1][ielemc] - arrcy[j][ielemc]));

      row.i = i;
      row.j = j;

      divUV_star = (arrUV_star[j][i + 1][ileft] - arrUV_star[j][i][ileft]) / wx + (arrUV_star[j + 1][i][idown] - arrUV_star[j][i][idown]) / wy;
      valb       = -rho / dt * divUV_star / (aw + ae + as + an);

      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(fdm, fv_star, &arrUV_star));

  // TODO: below is temporary for cavity flow
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

  const PetscReal rho = ns->rho;
  const PetscReal mu  = ns->mu;
  const PetscReal dt  = ns->dt;

  DM                  dm;
  PetscInt            M, N, x, y, m, n;
  DMStagStencil       row, col[5];
  PetscScalar         v[5];
  PetscInt            ncols;
  const PetscScalar **arrcx, **arrcy;
  PetscScalar         wx, wy, aw, ae, as, an;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (ncols = 0; ncols < 5; ++ncols) {
    col[ncols].loc = DMSTAG_ELEMENT;
    col[ncols].c   = 0;
  }

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx = arrcx[i][inextc] - arrcx[i][iprevc];
      wy = arrcy[j][inextc] - arrcy[j][iprevc];
      aw = 1.0 / (wx * (arrcx[i][ielemc] - arrcx[i - 1][ielemc]));
      ae = 1.0 / (wx * (arrcx[i + 1][ielemc] - arrcx[i][ielemc]));
      as = 1.0 / (wy * (arrcy[j][ielemc] - arrcy[j - 1][ielemc]));
      an = 1.0 / (wy * (arrcy[j + 1][ielemc] - arrcy[j][ielemc]));

      row.i = col[0].i = i;
      row.j = col[0].j = j;
      v[0]             = 1.0 + 0.5 * mu * dt / rho * (aw + ae + as + an);
      v[1] = v[2] = v[3] = v[4] = 0;
      ncols                     = 1;

      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) {
        v[0] += 0.5 * mu * dt / rho * aw;
      } else {
        col[ncols].i = i - 1;
        col[ncols].j = j;
        v[ncols]     = -0.5 * mu * dt / rho * aw;
        ++ncols;
      }
      /* Right wall. */
      if (i == M - 1) {
        v[0] += 0.5 * mu * dt / rho * ae;
      } else {
        col[ncols].i = i + 1;
        col[ncols].j = j;
        v[ncols]     = -0.5 * mu * dt / rho * ae;
        ++ncols;
      }
      /* Bottom wall. */
      if (j == 0) {
        v[0] += 0.5 * mu * dt / rho * as;
      } else {
        col[ncols].i = i;
        col[ncols].j = j - 1;
        v[ncols]     = -0.5 * mu * dt / rho * as;
        ++ncols;
      }
      /* Top wall. */
      if (j == N - 1) {
        v[0] += 0.5 * mu * dt / rho * an;
      } else {
        col[ncols].i = i;
        col[ncols].j = j + 1;
        v[ncols]     = -0.5 * mu * dt / rho * an;
        ++ncols;
      }

      PetscCall(DMStagMatSetValuesStencil(dm, Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
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
  PetscScalar         wx, wy, aw, ae, as, an, asum;
  PetscInt            iprevc, inextc, ielemc;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;
  for (ncols = 0; ncols < 5; ++ncols) {
    col[ncols].loc = DMSTAG_ELEMENT;
    col[ncols].c   = 0;
  }

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx   = arrcx[i][inextc] - arrcx[i][iprevc];
      wy   = arrcy[j][inextc] - arrcy[j][iprevc];
      aw   = 1.0 / (wx * (arrcx[i][ielemc] - arrcx[i - 1][ielemc]));
      ae   = 1.0 / (wx * (arrcx[i + 1][ielemc] - arrcx[i][ielemc]));
      as   = 1.0 / (wy * (arrcy[j][ielemc] - arrcy[j - 1][ielemc]));
      an   = 1.0 / (wy * (arrcy[j + 1][ielemc] - arrcy[j][ielemc]));
      asum = aw + ae + as + an;

      row.i = col[0].i = i;
      row.j = col[0].j = j;
      v[0]             = 1.0;
      v[1] = v[2] = v[3] = v[4] = 0;
      ncols                     = 1;

      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) {
        v[0] -= aw / asum;
      } else {
        col[ncols].i = i - 1;
        col[ncols].j = j;
        v[ncols]     = -aw / asum;
        ++ncols;
      }
      /* Right wall. */
      if (i == M - 1) {
        v[0] -= ae / asum;
      } else {
        col[ncols].i = i + 1;
        col[ncols].j = j;
        v[ncols]     = -ae / asum;
        ++ncols;
      }
      /* Bottom wall. */
      if (j == 0) {
        v[0] -= as / asum;
      } else {
        col[ncols].i = i;
        col[ncols].j = j - 1;
        v[ncols]     = -as / asum;
        ++ncols;
      }
      /* Top wall. */
      if (j == N - 1) {
        v[0] -= an / asum;
      } else {
        col[ncols].i = i;
        col[ncols].j = j + 1;
        v[ncols]     = -an / asum;
        ++ncols;
      }

      PetscCall(DMStagMatSetValuesStencil(dm, Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

  // TODO: below is temporary for cavity flow
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
