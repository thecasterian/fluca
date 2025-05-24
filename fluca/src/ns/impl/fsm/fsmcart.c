#include <fluca/private/meshcartimpl.h>
#include <fluca/private/ns_fsm.h>
#include <fluca/private/sol_fsm.h>
#include <petscdmstag.h>

extern PetscErrorCode ComputeRHSUStar2d(KSP, Vec, void *);
extern PetscErrorCode ComputeRHSVStar2d(KSP, Vec, void *);
extern PetscErrorCode ComputeOperatorsUVstar2d(KSP, Mat, Mat, void *);
extern PetscErrorCode ComputeRHSPprime2d(KSP, Vec, void *);
extern PetscErrorCode ComputeOperatorPprime2d(KSP, Mat, Mat, void *);

PetscErrorCode NSFSMInterpolateVelocity2d_MeshCart(NS ns)
{
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  PetscScalar       ***arrUV;
  const PetscScalar ***arru, ***arrv;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          wx_left, wx_right, wy_down, wy_up;
  PetscInt             i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(mesh->dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMStagVecGetArray(mesh->fdm, solfsm->fv, &arrUV));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(mesh->dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ielemc));

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

  PetscCall(DMStagVecRestoreArray(mesh->fdm, solfsm->fv, &arrUV));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMLocalToLocalBegin(mesh->fdm, solfsm->fv, INSERT_VALUES, solfsm->fv));
  PetscCall(DMLocalToLocalEnd(mesh->fdm, solfsm->fv, INSERT_VALUES, solfsm->fv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateConvection2d_MeshCart(NS ns)
{
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  Vec                  u_interp, v_interp;
  PetscScalar       ***arrNu, ***arrNv;
  PetscScalar       ***arru_interp, ***arrv_interp;
  const PetscScalar ***arru, ***arrv, ***arrUV;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, iright, idown, iup, ielem, iprevc, inextc, ielemc;
  PetscScalar          wx_left, wx_right, wy_down, wy_up, wx, wy;
  PetscInt             i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(mesh->dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMGetLocalVector(mesh->fdm, &u_interp));
  PetscCall(DMGetLocalVector(mesh->fdm, &v_interp));

  PetscCall(DMStagVecGetArray(mesh->dm, solfsm->N[0], &arrNu));
  PetscCall(DMStagVecGetArray(mesh->dm, solfsm->N[1], &arrNv));
  PetscCall(DMStagVecGetArray(mesh->fdm, u_interp, &arru_interp));
  PetscCall(DMStagVecGetArray(mesh->fdm, v_interp, &arrv_interp));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecGetArrayRead(mesh->fdm, solfsm->fv, &arrUV));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_RIGHT, 0, &iright));
  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_UP, 0, &iup));
  PetscCall(DMStagGetLocationSlot(mesh->dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ielemc));

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

  PetscCall(DMLocalToLocalBegin(mesh->fdm, u_interp, INSERT_VALUES, u_interp));
  PetscCall(DMLocalToLocalEnd(mesh->fdm, u_interp, INSERT_VALUES, u_interp));
  PetscCall(DMLocalToLocalBegin(mesh->fdm, v_interp, INSERT_VALUES, v_interp));
  PetscCall(DMLocalToLocalEnd(mesh->fdm, v_interp, INSERT_VALUES, v_interp));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      wx                 = arrcx[i][inextc] - arrcx[i][iprevc];
      wy                 = arrcy[j][inextc] - arrcy[j][iprevc];
      arrNu[j][i][ielem] = (arrUV[j][i][iright] * arru_interp[j][i][iright] - arrUV[j][i][ileft] * arru_interp[j][i][ileft]) / wx + (arrUV[j][i][iup] * arru_interp[j][i][iup] - arrUV[j][i][idown] * arru_interp[j][i][idown]) / wy;
      arrNv[j][i][ielem] = (arrUV[j][i][iright] * arrv_interp[j][i][iright] - arrUV[j][i][ileft] * arrv_interp[j][i][ileft]) / wx + (arrUV[j][i][iup] * arrv_interp[j][i][iup] - arrUV[j][i][idown] * arrv_interp[j][i][idown]) / wy;
    }

  PetscCall(DMStagVecRestoreArray(mesh->dm, solfsm->N[0], &arrNu));
  PetscCall(DMStagVecRestoreArray(mesh->dm, solfsm->N[1], &arrNv));
  PetscCall(DMStagVecRestoreArray(mesh->fdm, u_interp, &arru_interp));
  PetscCall(DMStagVecRestoreArray(mesh->fdm, v_interp, &arrv_interp));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecRestoreArrayRead(mesh->fdm, solfsm->fv, &arrUV));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMRestoreLocalVector(mesh->fdm, &u_interp));
  PetscCall(DMRestoreLocalVector(mesh->fdm, &v_interp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateIntermediateVelocity2d_MeshCart(NS ns)
{
  NS_FSM  *nsfsm  = (NS_FSM *)ns->data;
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

  PetscErrorCode (*rhs[2])(KSP, Vec, void *) = {ComputeRHSUStar2d, ComputeRHSVStar2d};
  Vec                  v;
  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  PetscScalar       ***arru_tilde, ***arrv_tilde, ***arrUV_star;
  const PetscScalar ***arru_star, ***arrv_star, ***arrp;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          pw, pe, ps, pn, U_tilde, V_tilde, dpdx, dpdy;
  PetscScalar          wx_left, wx_right, wy_down, wy_up;
  PetscInt             d, i, j;

  PetscFunctionBegin;
  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    PetscCall(KSPSetComputeRHS(nsfsm->kspv[d], rhs[d], ns));
    PetscCall(KSPSetComputeOperators(nsfsm->kspv[d], ComputeOperatorsUVstar2d, ns));
    PetscCall(KSPSolve(nsfsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(nsfsm->kspv[d], &v));
    PetscCall(DMGlobalToLocal(mesh->dm, v, INSERT_VALUES, solfsm->v_star[d]));
  }

  /* Calculate face-centered intermediate velocity. */
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(mesh->dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMStagVecGetArray(mesh->dm, solfsm->v_tilde[0], &arru_tilde));
  PetscCall(DMStagVecGetArray(mesh->dm, solfsm->v_tilde[1], &arrv_tilde));
  PetscCall(DMStagVecGetArray(mesh->fdm, solfsm->fv_star, &arrUV_star));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->v_star[0], &arru_star));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->v_star[1], &arrv_star));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(mesh->dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ielemc));

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

      arru_tilde[j][i][ielem] = arru_star[j][i][ielem] + ns->dt / ns->rho * (pe - pw) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
      arrv_tilde[j][i][ielem] = arrv_star[j][i][ielem] + ns->dt / ns->rho * (pn - ps) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
    }

  PetscCall(DMLocalToLocalBegin(mesh->dm, solfsm->v_tilde[0], INSERT_VALUES, solfsm->v_tilde[0]));
  PetscCall(DMLocalToLocalEnd(mesh->dm, solfsm->v_tilde[0], INSERT_VALUES, solfsm->v_tilde[0]));
  PetscCall(DMLocalToLocalBegin(mesh->dm, solfsm->v_tilde[1], INSERT_VALUES, solfsm->v_tilde[1]));
  PetscCall(DMLocalToLocalEnd(mesh->dm, solfsm->v_tilde[1], INSERT_VALUES, solfsm->v_tilde[1]));

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
        arrUV_star[j][i][ileft] = U_tilde - ns->dt / ns->rho * dpdx;
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
        arrUV_star[j][i][idown] = V_tilde - ns->dt / ns->rho * dpdy;
      }
    }

  PetscCall(DMStagVecRestoreArray(mesh->dm, solfsm->v_tilde[0], &arru_tilde));
  PetscCall(DMStagVecRestoreArray(mesh->dm, solfsm->v_tilde[1], &arrv_tilde));
  PetscCall(DMStagVecRestoreArray(mesh->fdm, solfsm->fv_star, &arrUV_star));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->v_star[0], &arru_star));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->v_star[1], &arrv_star));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMLocalToLocalBegin(mesh->fdm, solfsm->fv_star, INSERT_VALUES, solfsm->fv_star));
  PetscCall(DMLocalToLocalEnd(mesh->fdm, solfsm->fv_star, INSERT_VALUES, solfsm->fv_star));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculatePressureCorrection2d_MeshCart(NS ns)
{
  NS_FSM  *nsfsm  = (NS_FSM *)ns->data;
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

  Vec v;

  PetscFunctionBegin;
  PetscCall(KSPSetComputeRHS(nsfsm->kspp, ComputeRHSPprime2d, ns));
  PetscCall(KSPSetComputeOperators(nsfsm->kspp, ComputeOperatorPprime2d, ns));
  PetscCall(KSPSolve(nsfsm->kspp, NULL, NULL));
  PetscCall(KSPGetSolution(nsfsm->kspp, &v));
  PetscCall(DMGlobalToLocal(mesh->dm, v, INSERT_VALUES, solfsm->p_prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMUpdate2d_MeshCart(NS ns)
{
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

  PetscInt             M, N, x, y, m, n;
  PetscScalar       ***arru, ***arrv, ***arrp, ***arrUV;
  const PetscScalar ***arru_star, ***arrv_star, ***arrp_prime, ***arrUV_star;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          ppp, ppw, ppe, pps, ppn, pplap;
  PetscScalar          wx, wy, aw, ae, as, an;
  PetscInt             i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(mesh->dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(VecCopy(solfsm->p_half, solfsm->p_half_prev));
  PetscCall(VecCopy(solfsm->N[0], solfsm->N_prev[0]));
  PetscCall(VecCopy(solfsm->N[1], solfsm->N_prev[1]));

  PetscCall(DMStagVecGetArray(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecGetArray(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecGetArray(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagVecGetArray(mesh->fdm, solfsm->fv, &arrUV));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->v_star[0], &arru_star));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->v_star[1], &arrv_star));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->p_prime, &arrp_prime));
  PetscCall(DMStagVecGetArrayRead(mesh->fdm, solfsm->fv_star, &arrUV_star));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(mesh->dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ielemc));

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

      arru[j][i][ielem] = arru_star[j][i][ielem] - ns->dt / ns->rho * (ppe - ppw) / (arrcx[i + 1][ielemc] - arrcx[i - 1][ielemc]);
      arrv[j][i][ielem] = arrv_star[j][i][ielem] - ns->dt / ns->rho * (ppn - pps) / (arrcy[j + 1][ielemc] - arrcy[j - 1][ielemc]);
      arrp[j][i][ielem] += ppp - 0.5 * ns->mu * ns->dt / ns->rho * pplap;
    }

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + 1; ++i) {
      /* Left wall. */
      if (i == 0) arrUV[j][i][ileft] = 0.0;
      /* Right wall. */
      else if (i == M) arrUV[j][i][ileft] = 0.0;
      else arrUV[j][i][ileft] = arrUV_star[j][i][ileft] - ns->dt / ns->rho * (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
    }

  for (j = y; j < y + n + 1; ++j)
    for (i = x; i < x + m; ++i) {
      /* Bottom wall. */
      if (j == 0) arrUV[j][i][idown] = 0.0;
      /* Top wall. */
      else if (j == N) arrUV[j][i][idown] = 0.0;
      else arrUV[j][i][idown] = arrUV_star[j][i][idown] - ns->dt / ns->rho * (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
    }

  PetscCall(DMStagVecRestoreArray(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecRestoreArray(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecRestoreArray(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagVecRestoreArray(mesh->fdm, solfsm->fv, &arrUV));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->v_star[0], &arru_star));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->v_star[1], &arrv_star));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->p_prime, &arrp_prime));
  PetscCall(DMStagVecRestoreArrayRead(mesh->fdm, solfsm->fv_star, &arrUV_star));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMLocalToLocalBegin(mesh->dm, sol->v[0], INSERT_VALUES, sol->v[0]));
  PetscCall(DMLocalToLocalEnd(mesh->dm, sol->v[0], INSERT_VALUES, sol->v[0]));
  PetscCall(DMLocalToLocalBegin(mesh->dm, sol->v[1], INSERT_VALUES, sol->v[1]));
  PetscCall(DMLocalToLocalEnd(mesh->dm, sol->v[1], INSERT_VALUES, sol->v[1]));
  PetscCall(DMLocalToLocalBegin(mesh->dm, solfsm->p_half, INSERT_VALUES, solfsm->p_half));
  PetscCall(DMLocalToLocalEnd(mesh->dm, solfsm->p_half, INSERT_VALUES, solfsm->p_half));

  PetscCall(NSFSMCalculateConvection2d_MeshCart(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSUStar2d(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS       ns     = (NS)ctx;
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

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
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(mesh->dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->N[0], &arrNu));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->N_prev[0], &arrNu_prev));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(mesh->dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ielemc));

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

      valb = up - 1.5 * ns->dt * arrNu[j][i][ielem] + 0.5 * ns->dt * arrNu_prev[j][i][ielem] - ns->dt / ns->rho * dpdx + 0.5 * ns->mu * ns->dt / ns->rho * ulap;
      // TODO: below is temporary for cavity flow
      if (j == N - 1) valb += ns->mu * ns->dt / ns->rho * an;

      PetscCall(DMStagVecSetValuesStencil(mesh->dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->N[0], &arrNu));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->N_prev[0], &arrNu_prev));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSVStar2d(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS       ns     = (NS)ctx;
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

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
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(mesh->dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->N[1], &arrNv));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->N_prev[1], &arrNv_prev));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecGetArrayRead(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(mesh->dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ielemc));

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

      valb = vp - 1.5 * ns->dt * arrNv[j][i][ielem] + 0.5 * ns->dt * arrNv_prev[j][i][ielem] - ns->dt / ns->rho * dpdy + 0.5 * ns->mu * ns->dt / ns->rho * vlap;

      PetscCall(DMStagVecSetValuesStencil(mesh->dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->N[1], &arrNv));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->N_prev[1], &arrNv_prev));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[0], &arru));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, sol->v[1], &arrv));
  PetscCall(DMStagVecRestoreArrayRead(mesh->dm, solfsm->p_half, &arrp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSPprime2d(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS       ns     = (NS)ctx;
  Mesh     mesh   = ns->mesh;
  Sol      sol    = ns->sol;
  Sol_FSM *solfsm = (Sol_FSM *)sol->data;

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
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(mesh->dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagVecGetArrayRead(mesh->fdm, solfsm->fv_star, &arrUV_star));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(mesh->fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(mesh->dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ielemc));

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
      valb       = -ns->rho / ns->dt * divUV_star / (aw + ae + as + an);

      PetscCall(DMStagVecSetValuesStencil(mesh->dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(mesh->fdm, solfsm->fv_star, &arrUV_star));

  // TODO: below is temporary for cavity flow
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorsUVstar2d(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  (void)J;

  NS ns = (NS)ctx;

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
      v[0]             = 1.0 + 0.5 * ns->mu * ns->dt / ns->rho * (aw + ae + as + an);
      v[1] = v[2] = v[3] = v[4] = 0;
      ncols                     = 1;

      // TODO: below is temporary for cavity flow
      /* Left wall. */
      if (i == 0) {
        v[0] += 0.5 * ns->mu * ns->dt / ns->rho * aw;
      } else {
        col[ncols].i = i - 1;
        col[ncols].j = j;
        v[ncols]     = -0.5 * ns->mu * ns->dt / ns->rho * aw;
        ++ncols;
      }
      /* Right wall. */
      if (i == M - 1) {
        v[0] += 0.5 * ns->mu * ns->dt / ns->rho * ae;
      } else {
        col[ncols].i = i + 1;
        col[ncols].j = j;
        v[ncols]     = -0.5 * ns->mu * ns->dt / ns->rho * ae;
        ++ncols;
      }
      /* Bottom wall. */
      if (j == 0) {
        v[0] += 0.5 * ns->mu * ns->dt / ns->rho * as;
      } else {
        col[ncols].i = i;
        col[ncols].j = j - 1;
        v[ncols]     = -0.5 * ns->mu * ns->dt / ns->rho * as;
        ++ncols;
      }
      /* Top wall. */
      if (j == N - 1) {
        v[0] += 0.5 * ns->mu * ns->dt / ns->rho * an;
      } else {
        col[ncols].i = i;
        col[ncols].j = j + 1;
        v[ncols]     = -0.5 * ns->mu * ns->dt / ns->rho * an;
        ++ncols;
      }

      PetscCall(DMStagMatSetValuesStencil(dm, Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
    }

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorPprime2d(KSP ksp, Mat J, Mat Jpre, void *ctx)
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
