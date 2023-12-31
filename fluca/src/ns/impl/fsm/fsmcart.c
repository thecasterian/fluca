#include <fluca/private/mesh_cartesian.h>
#include <fluca/private/ns_fsm.h>
#include <fluca/private/sol_fsm.h>
#include <petscdmstag.h>

#define DERIV2W (cart->a[0][i][0])
#define DERIV2E (cart->a[0][i][1])
#define DERIV2S (cart->a[1][j][0])
#define DERIV2N (cart->a[1][j][1])

extern PetscErrorCode ComputeRHSUStar2d(KSP, Vec, void *);
extern PetscErrorCode ComputeRHSVStar2d(KSP, Vec, void *);
extern PetscErrorCode ComputeOperatorsUVstar2d(KSP, Mat, Mat, void *);
extern PetscErrorCode ComputeRHSPprime2d(KSP, Vec, void *);
extern PetscErrorCode ComputeOperatorPprime2d(KSP, Mat, Mat, void *);

PetscErrorCode NSFSMInterpolateVelocity2d_MeshCartesian(NS ns) {
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    PetscInt M, N, xs, ys, xm, ym;
    PetscReal ***arrUV;
    const PetscReal ***arru, ***arrv;
    const PetscReal **arrwx, **arrwy;
    PetscInt iU, iV, ielem, iw;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm + 1; i++) {
            /* Left wall. */
            if (i == 0)
                arrUV[j][i][iU] = 0.0;
            /* Right wall. */
            else if (i == M)
                arrUV[j][i][iU] = 0.0;
            else
                arrUV[j][i][iU] = (arrwx[i][iw] * arru[j][i - 1][ielem] + arrwx[i - 1][iw] * arru[j][i][ielem]) /
                                  (arrwx[i][iw] + arrwx[i - 1][iw]);
        }
    for (j = ys; j < ys + ym + 1; j++)
        for (i = xs; i < xs + xm; i++) {
            /* Bottom wall. */
            if (j == 0)
                arrUV[j][i][iV] = 0.0;
            /* Top wall. */
            else if (j == N)
                arrUV[j][i][iV] = 0.0;
            else
                arrUV[j][i][iV] = (arrwy[j][iw] * arrv[j - 1][i][ielem] + arrwy[j - 1][iw] * arrv[j][i][ielem]) /
                                  (arrwy[j][iw] + arrwy[j - 1][iw]);
        }

    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateConvection2d_MeshCartesian(NS ns) {
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    PetscInt M, N, xs, ys, xm, ym;
    PetscReal ***arrNu, ***arrNv;
    const PetscReal ***arru, ***arrv, ***arrUV;
    const PetscReal **arrwx, **arrwy;
    PetscInt iU, iV, ielem, iw;
    PetscReal uw, ue, us, un, vw, ve, vs, vn;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArray(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMStagVecGetArray(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArrayRead(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                uw = 0.0;
                vw = 0.0;
            } else {
                uw = (arrwx[i][iw] * arru[j][i - 1][ielem] + arrwx[i - 1][iw] * arru[j][i][ielem]) /
                     (arrwx[i][iw] + arrwx[i - 1][iw]);
                vw = (arrwx[i][iw] * arrv[j][i - 1][ielem] + arrwx[i - 1][iw] * arrv[j][i][ielem]) /
                     (arrwx[i][iw] + arrwx[i - 1][iw]);
            }
            /* Right wall. */
            if (i == M - 1) {
                ue = 0.0;
                ve = 0.0;
            } else {
                ue = (arrwx[i][iw] * arru[j][i + 1][ielem] + arrwx[i + 1][iw] * arru[j][i][ielem]) /
                     (arrwx[i][iw] + arrwx[i + 1][iw]);
                ve = (arrwx[i][iw] * arrv[j][i + 1][ielem] + arrwx[i + 1][iw] * arrv[j][i][ielem]) /
                     (arrwx[i][iw] + arrwx[i + 1][iw]);
            }
            /* Bottom wall. */
            if (j == 0) {
                us = 0.0;
                vs = 0.0;
            } else {
                us = (arrwy[j][iw] * arru[j - 1][i][ielem] + arrwy[j - 1][iw] * arru[j][i][ielem]) /
                     (arrwy[j][iw] + arrwy[j - 1][iw]);
                vs = (arrwy[j][iw] * arrv[j - 1][i][ielem] + arrwy[j - 1][iw] * arrv[j][i][ielem]) /
                     (arrwy[j][iw] + arrwy[j - 1][iw]);
            }
            /* Top wall. */
            if (j == N - 1) {
                un = 0.0;
                vn = 0.0;
            } else {
                un = (arrwy[j][iw] * arru[j + 1][i][ielem] + arrwy[j + 1][iw] * arru[j][i][ielem]) /
                     (arrwy[j][iw] + arrwy[j + 1][iw]);
                vn = (arrwy[j][iw] * arrv[j + 1][i][ielem] + arrwy[j + 1][iw] * arrv[j][i][ielem]) /
                     (arrwy[j][iw] + arrwy[j + 1][iw]);
            }

            arrNu[j][i][ielem] = (arrUV[j][i + 1][iU] * ue - arrUV[j][i][iU] * uw) / arrwx[i][iw] +
                                 (arrUV[j + 1][i][iV] * un - arrUV[j][i][iV] * us) / arrwy[j][iw];
            arrNv[j][i][ielem] = (arrUV[j][i + 1][iU] * ve - arrUV[j][i][iU] * vw) / arrwx[i][iw] +
                                 (arrUV[j + 1][i][iV] * vn - arrUV[j][i][iV] * vs) / arrwy[j][iw];
        }

    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[1], cart->width[1], &arrwy));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateIntermediateVelocity2d_MeshCartesian(NS ns) {
    NS_FSM *nsfsm = (NS_FSM *)ns->data;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    PetscErrorCode (*rhs[2])(KSP, Vec, void *) = {ComputeRHSUStar2d, ComputeRHSVStar2d};
    Vec x;
    PetscInt M, N, xs, ys, xm, ym;
    PetscReal ***arru_tilde, ***arrv_tilde, ***arrUV_star;
    const PetscReal ***arru_star, ***arrv_star, ***arrp;
    const PetscReal **arrwx, **arrwy, **arrcx, **arrcy;
    PetscInt iU, iV, ielem, iw, ic;
    PetscReal pw, pe, ps, pn, U_tilde, V_tilde, dpdx, dpdy;
    PetscInt d, i, j;

    PetscFunctionBegin;

    /* Solve for cell-centered intermediate velocity. */
    for (d = 0; d < 2; d++) {
        PetscCall(KSPSetComputeRHS(nsfsm->ksp, rhs[d], ns));
        PetscCall(KSPSetComputeOperators(nsfsm->ksp, ComputeOperatorsUVstar2d, ns));
        PetscCall(KSPSolve(nsfsm->ksp, NULL, NULL));
        PetscCall(KSPGetSolution(nsfsm->ksp, &x));
        PetscCall(DMGlobalToLocal(cart->dm, x, INSERT_VALUES, solfsm->v_star[d]));
    }

    /* Calculate face-centered intermediate velocity. */
    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArray(cart->dm, solfsm->v_tilde[0], &arru_tilde));
    PetscCall(DMStagVecGetArray(cart->dm, solfsm->v_tilde[1], &arrv_tilde));
    PetscCall(DMStagVecGetArray(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &ic));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
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

            arru_tilde[j][i][ielem] =
                arru_star[j][i][ielem] + ns->dt / ns->rho * (pe - pw) / (arrcx[i + 1][ic] - arrcx[i - 1][ic]);
            arrv_tilde[j][i][ielem] =
                arrv_star[j][i][ielem] + ns->dt / ns->rho * (pn - ps) / (arrcy[j + 1][ic] - arrcy[j - 1][ic]);
        }

    PetscCall(DMLocalToLocalBegin(cart->dm, solfsm->v_tilde[0], INSERT_VALUES, solfsm->v_tilde[0]));
    PetscCall(DMLocalToLocalEnd(cart->dm, solfsm->v_tilde[0], INSERT_VALUES, solfsm->v_tilde[0]));
    PetscCall(DMLocalToLocalBegin(cart->dm, solfsm->v_tilde[1], INSERT_VALUES, solfsm->v_tilde[1]));
    PetscCall(DMLocalToLocalEnd(cart->dm, solfsm->v_tilde[1], INSERT_VALUES, solfsm->v_tilde[1]));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm + 1; i++) {
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                arrUV_star[j][i][iU] = 0.0;
            }
            /* Right wall. */
            else if (i == M) {
                arrUV_star[j][i][iU] = 0.0;
            } else {
                U_tilde = (arrwx[i][iw] * arru_tilde[j][i - 1][ielem] + arrwx[i - 1][iw] * arru_tilde[j][i][ielem]) /
                          (arrwx[i][iw] + arrwx[i - 1][iw]);
                dpdx = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ic] - arrcx[i - 1][ic]);
                arrUV_star[j][i][iU] = U_tilde - ns->dt / ns->rho * dpdx;
            }
        }
    for (j = ys; j < ys + ym + 1; j++)
        for (i = xs; i < xs + xm; i++) {
            // TODO: below is temporary for cavity flow
            /* Bottom wall. */
            if (j == 0) {
                arrUV_star[j][i][iV] = 0.0;
            }
            /* Top wall. */
            else if (j == N) {
                arrUV_star[j][i][iV] = 0.0;
            } else {
                V_tilde = (arrwy[j][iw] * arrv_tilde[j - 1][i][ielem] + arrwy[j - 1][iw] * arrv_tilde[j][i][ielem]) /
                          (arrwy[j][iw] + arrwy[j - 1][iw]);
                dpdy = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (arrcy[j][ic] - arrcy[j - 1][ic]);
                arrUV_star[j][i][iV] = V_tilde - ns->dt / ns->rho * dpdy;
            }
        }

    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->v_tilde[0], &arru_tilde));
    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->v_tilde[1], &arrv_tilde));
    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculatePressureCorrection2d_MeshCartesian(NS ns) {
    NS_FSM *nsfsm = (NS_FSM *)ns->data;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    Vec x;

    PetscFunctionBegin;

    PetscCall(KSPSetComputeRHS(nsfsm->ksp, ComputeRHSPprime2d, ns));
    PetscCall(KSPSetComputeOperators(nsfsm->ksp, ComputeOperatorPprime2d, ns));
    PetscCall(KSPSolve(nsfsm->ksp, NULL, NULL));
    PetscCall(KSPGetSolution(nsfsm->ksp, &x));
    PetscCall(DMGlobalToLocal(cart->dm, x, INSERT_VALUES, solfsm->p_prime));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMUpdate2d_MeshCartesian(NS ns) {
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    PetscInt M, N, xs, ys, xm, ym;
    PetscReal ***arru, ***arrv, ***arrp, ***arrUV;
    const PetscReal ***arru_star, ***arrv_star, ***arrp_prime, ***arrUV_star;
    const PetscReal **arrwx, **arrwy, **arrcx, **arrcy;
    PetscInt iU, iV, ielem, iw, ic;
    PetscReal ppp, ppw, ppe, pps, ppn, pplap;
    PetscReal aw, ae, as, an;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(VecCopy(solfsm->p_half, solfsm->p_half_prev));
    PetscCall(VecCopy(solfsm->N[0], solfsm->N_prev[0]));
    PetscCall(VecCopy(solfsm->N[1], solfsm->N_prev[1]));

    PetscCall(DMStagVecGetArray(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArray(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArray(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecGetArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->p_prime, &arrp_prime));
    PetscCall(DMStagVecGetArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &ic));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            aw = 1.0 / (arrwx[i][iw] * (arrcx[i][ic] - arrcx[i - 1][ic]));
            ae = 1.0 / (arrwx[i][iw] * (arrcx[i + 1][ic] - arrcx[i][ic]));
            as = 1.0 / (arrwy[j][iw] * (arrcy[j][ic] - arrcy[j - 1][ic]));
            an = 1.0 / (arrwy[j][iw] * (arrcy[j + 1][ic] - arrcy[j][ic]));

            ppp = arrp_prime[j][i][ielem];

            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0)
                ppw = arrp_prime[j][i][ielem];
            else
                ppw = arrp_prime[j][i - 1][ielem];
            /* Right wall. */
            if (i == M)
                ppe = arrp_prime[j][i][ielem];
            else
                ppe = arrp_prime[j][i + 1][ielem];
            /* Bottom wall. */
            if (j == 0)
                pps = arrp_prime[j][i][ielem];
            else
                pps = arrp_prime[j - 1][i][ielem];
            /* Top wall. */
            if (j == N)
                ppn = arrp_prime[j][i][ielem];
            else
                ppn = arrp_prime[j + 1][i][ielem];

            pplap = aw * ppw + ae * ppe + as * pps + an * ppn - (aw + ae + as + an) * ppp;

            arru[j][i][ielem] =
                arru_star[j][i][ielem] - ns->dt / ns->rho * (ppe - ppw) / (arrcx[i + 1][ic] - arrcx[i - 1][ic]);
            arrv[j][i][ielem] =
                arrv_star[j][i][ielem] - ns->dt / ns->rho * (ppn - pps) / (arrcy[j + 1][ic] - arrcy[j - 1][ic]);
            arrp[j][i][ielem] += ppp - 0.5 * ns->mu * ns->dt / ns->rho * pplap;
        }

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm + 1; i++) {
            /* Left wall. */
            if (i == 0)
                arrUV[j][i][iU] = 0.0;
            /* Right wall. */
            else if (i == M)
                arrUV[j][i][iU] = 0.0;
            else
                arrUV[j][i][iU] = arrUV_star[j][i][iU] - ns->dt / ns->rho *
                                                             (arrp_prime[j][i][ielem] - arrp_prime[j][i - 1][ielem]) /
                                                             (arrcx[i][ic] - arrcx[i - 1][ic]);
        }

    for (j = ys; j < ys + ym + 1; j++)
        for (i = xs; i < xs + xm; i++) {
            /* Bottom wall. */
            if (j == 0)
                arrUV[j][i][iV] = 0.0;
            /* Top wall. */
            else if (j == N)
                arrUV[j][i][iV] = 0.0;
            else
                arrUV[j][i][iV] = arrUV_star[j][i][iV] - ns->dt / ns->rho *
                                                             (arrp_prime[j][i][ielem] - arrp_prime[j - 1][i][ielem]) /
                                                             (arrcy[j][ic] - arrcy[j - 1][ic]);
        }

    PetscCall(DMStagVecRestoreArray(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArray(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_prime, &arrp_prime));
    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMLocalToLocalBegin(cart->dm, sol->v[0], INSERT_VALUES, sol->v[0]));
    PetscCall(DMLocalToLocalEnd(cart->dm, sol->v[0], INSERT_VALUES, sol->v[0]));
    PetscCall(DMLocalToLocalBegin(cart->dm, sol->v[1], INSERT_VALUES, sol->v[1]));
    PetscCall(DMLocalToLocalEnd(cart->dm, sol->v[1], INSERT_VALUES, sol->v[1]));
    PetscCall(DMLocalToLocalBegin(cart->dm, solfsm->p_half, INSERT_VALUES, solfsm->p_half));
    PetscCall(DMLocalToLocalEnd(cart->dm, solfsm->p_half, INSERT_VALUES, solfsm->p_half));

    PetscCall(NSFSMCalculateConvection2d_MeshCartesian(ns));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSUStar2d(KSP ksp, Vec b, void *ctx) {
    (void)ksp;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    PetscInt M, N, xs, ys, xm, ym;
    DMStagStencil row;
    PetscReal valb;
    const PetscReal ***arrNu, ***arrNu_prev, ***arru, ***arrv, ***arrp;
    const PetscReal **arrwx, **arrwy, **arrcx, **arrcy;
    PetscReal up, uw, ue, us, un, pw, pe, dpdx, ulap;
    PetscReal aw, ae, as, an;
    PetscInt ielem, iw, ic;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N_prev[0], &arrNu_prev));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &ic));

    row.loc = DMSTAG_ELEMENT;
    row.c = 0;

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            aw = 1.0 / (arrwx[i][iw] * (arrcx[i][ic] - arrcx[i - 1][ic]));
            ae = 1.0 / (arrwx[i][iw] * (arrcx[i + 1][ic] - arrcx[i][ic]));
            as = 1.0 / (arrwy[j][iw] * (arrcy[j][ic] - arrcy[j - 1][ic]));
            an = 1.0 / (arrwy[j][iw] * (arrcy[j + 1][ic] - arrcy[j][ic]));

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
            if (j == 0)
                us = -arru[j][i][ielem];
            else
                us = arru[j - 1][i][ielem];
            /* Top wall. */
            if (j == N - 1)
                un = 2.0 - arru[j][i][ielem];
            else
                un = arru[j + 1][i][ielem];

            /* Pressure gradient. */
            dpdx = (pe - pw) / (arrcx[i + 1][ic] - arrcx[i - 1][ic]);
            /* Laplacian of u. */
            ulap = aw * uw + ae * ue + as * us + an * un - (aw + ae + as + an) * up;

            valb = up - 1.5 * ns->dt * arrNu[j][i][ielem] + 0.5 * ns->dt * arrNu_prev[j][i][ielem] -
                   ns->dt / ns->rho * dpdx + 0.5 * ns->mu * ns->dt / ns->rho * ulap;
            // TODO: below is temporary for cavity flow
            if (j == N - 1)
                valb += ns->mu * ns->dt / ns->rho * an;

            PetscCall(DMStagVecSetValuesStencil(cart->dm, b, 1, &row, &valb, INSERT_VALUES));
        }

    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N_prev[0], &arrNu_prev));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSVStar2d(KSP ksp, Vec b, void *ctx) {
    (void)ksp;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    PetscInt M, N, xs, ys, xm, ym;
    DMStagStencil row;
    PetscReal valb;
    const PetscReal ***arrNv, ***arrNv_prev, ***arru, ***arrv, ***arrp;
    const PetscReal **arrwx, **arrwy, **arrcx, **arrcy;
    PetscReal vp, vw, ve, vs, vn, ps, pn, dpdy, vlap;
    PetscReal aw, ae, as, an;
    PetscInt ielem, iw, ic;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N_prev[1], &arrNv_prev));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->subdm[1], DMSTAG_ELEMENT, 0, &iw));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &ic));

    row.loc = DMSTAG_ELEMENT;
    row.c = 0;

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            aw = 1.0 / (arrwx[i][iw] * (arrcx[i][ic] - arrcx[i - 1][ic]));
            ae = 1.0 / (arrwx[i][iw] * (arrcx[i + 1][ic] - arrcx[i][ic]));
            as = 1.0 / (arrwy[j][iw] * (arrcy[j][ic] - arrcy[j - 1][ic]));
            an = 1.0 / (arrwy[j][iw] * (arrcy[j + 1][ic] - arrcy[j][ic]));

            row.i = i;
            row.j = j;

            vp = arrv[j][i][ielem];
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0)
                vw = -arrv[j][i][ielem];
            else
                vw = arrv[j][i - 1][ielem];
            /* Right wall. */
            if (i == M - 1)
                ve = -arrv[j][i][ielem];
            else
                ve = arrv[j][i + 1][ielem];
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
            dpdy = (pn - ps) / (arrcy[j + 1][ic] - arrcy[j - 1][ic]);
            /* Laplacian of v. */
            vlap = aw * vw + ae * ve + as * vs + an * vn - (aw + ae + as + an) * vp;

            valb = vp - 1.5 * ns->dt * arrNv[j][i][ielem] + 0.5 * ns->dt * arrNv_prev[j][i][ielem] -
                   ns->dt / ns->rho * dpdy + 0.5 * ns->mu * ns->dt / ns->rho * vlap;

            PetscCall(DMStagVecSetValuesStencil(cart->dm, b, 1, &row, &valb, INSERT_VALUES));
        }

    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N_prev[1], &arrNv_prev));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSPprime2d(KSP ksp, Vec b, void *ctx) {
    (void)ksp;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    MPI_Comm comm;
    PetscInt M, N, xs, ys, xm, ym;
    DMStagStencil row;
    PetscReal valb;
    const PetscReal ***arrUV_star;
    const PetscReal **arrwx, **arrwy, **arrcx, **arrcy;
    PetscReal aw, ae, as, an;
    PetscInt iU, iV, ielem, iw, ic;
    PetscReal divUV_star;
    MatNullSpace nullspace;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &ic));

    row.loc = DMSTAG_ELEMENT;
    row.c = 0;

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            aw = 1.0 / (arrwx[i][iw] * (arrcx[i][ic] - arrcx[i - 1][ic]));
            ae = 1.0 / (arrwx[i][iw] * (arrcx[i + 1][ic] - arrcx[i][ic]));
            as = 1.0 / (arrwy[j][iw] * (arrcy[j][ic] - arrcy[j - 1][ic]));
            an = 1.0 / (arrwy[j][iw] * (arrcy[j + 1][ic] - arrcy[j][ic]));

            row.i = i;
            row.j = j;

            divUV_star = (arrUV_star[j][i + 1][iU] - arrUV_star[j][i][iU]) / arrwx[i][iw] +
                         (arrUV_star[j + 1][i][iV] - arrUV_star[j][i][iV]) / arrwy[j][iw];
            valb = -ns->rho / ns->dt * divUV_star / (aw + ae + as + an);

            PetscCall(DMStagVecSetValuesStencil(cart->dm, b, 1, &row, &valb, INSERT_VALUES));
        }

    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));

    // TODO: below is temporary for cavity flow
    /* Remove null space. */
    PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
    PetscCall(MatNullSpaceRemove(nullspace, b));
    PetscCall(MatNullSpaceDestroy(&nullspace));

    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorsUVstar2d(KSP ksp, Mat J, Mat Jpre, void *ctx) {
    (void)ksp;
    (void)J;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscInt M, N, xs, ys, xm, ym;
    DMStagStencil row, col[5];
    PetscReal v[5];
    PetscInt ncols;
    const PetscReal **arrwx, **arrwy, **arrcx, **arrcy;
    PetscReal aw, ae, as, an;
    PetscInt iw, ic;
    PetscInt i, j;

    PetscFunctionBegin;

    // TODO: support multigrid method (dm of KSP != cart->dm)

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &ic));

    row.loc = DMSTAG_ELEMENT;
    row.c = 0;
    for (ncols = 0; ncols < 5; ncols++) {
        col[ncols].loc = DMSTAG_ELEMENT;
        col[ncols].c = 0;
    }

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            aw = 1.0 / (arrwx[i][iw] * (arrcx[i][ic] - arrcx[i - 1][ic]));
            ae = 1.0 / (arrwx[i][iw] * (arrcx[i + 1][ic] - arrcx[i][ic]));
            as = 1.0 / (arrwy[j][iw] * (arrcy[j][ic] - arrcy[j - 1][ic]));
            an = 1.0 / (arrwy[j][iw] * (arrcy[j + 1][ic] - arrcy[j][ic]));

            row.i = col[0].i = i;
            row.j = col[0].j = j;
            v[0] = 1.0 + 0.5 * ns->mu * ns->dt / ns->rho * (aw + ae + as + an);
            v[1] = v[2] = v[3] = v[4] = 0;
            ncols = 1;

            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * aw;
            } else {
                col[ncols].i = i - 1;
                col[ncols].j = j;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * aw;
                ncols++;
            }
            /* Right wall. */
            if (i == M - 1) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * ae;
            } else {
                col[ncols].i = i + 1;
                col[ncols].j = j;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * ae;
                ncols++;
            }
            /* Bottom wall. */
            if (j == 0) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * as;
            } else {
                col[ncols].i = i;
                col[ncols].j = j - 1;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * as;
                ncols++;
            }
            /* Top wall. */
            if (j == N - 1) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * an;
            } else {
                col[ncols].i = i;
                col[ncols].j = j + 1;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * an;
                ncols++;
            }

            PetscCall(DMStagMatSetValuesStencil(cart->dm, Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
        }

    PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorPprime2d(KSP ksp, Mat J, Mat Jpre, void *ctx) {
    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    MPI_Comm comm;
    PetscInt M, N, xs, ys, xm, ym;
    DMStagStencil row, col[5];
    PetscReal v[5];
    PetscInt ncols;
    MatNullSpace nullspace;
    const PetscReal **arrwx, **arrwy, **arrcx, **arrcy;
    PetscReal aw, ae, as, an, asum;
    PetscInt iw, ic;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

    // TODO: support multigrid method (dm of KSP != cart->dm)

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecGetArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetLocationSlot(cart->subdm[0], DMSTAG_ELEMENT, 0, &iw));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &ic));

    row.loc = DMSTAG_ELEMENT;
    row.c = 0;
    for (ncols = 0; ncols < 5; ncols++) {
        col[ncols].loc = DMSTAG_ELEMENT;
        col[ncols].c = 0;
    }

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            aw = 1.0 / (arrwx[i][iw] * (arrcx[i][ic] - arrcx[i - 1][ic]));
            ae = 1.0 / (arrwx[i][iw] * (arrcx[i + 1][ic] - arrcx[i][ic]));
            as = 1.0 / (arrwy[j][iw] * (arrcy[j][ic] - arrcy[j - 1][ic]));
            an = 1.0 / (arrwy[j][iw] * (arrcy[j + 1][ic] - arrcy[j][ic]));
            asum = aw + ae + as + an;

            row.i = col[0].i = i;
            row.j = col[0].j = j;
            v[0] = 1.0;
            v[1] = v[2] = v[3] = v[4] = 0;
            ncols = 1;

            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                v[0] -= aw / asum;
            } else {
                col[ncols].i = i - 1;
                col[ncols].j = j;
                v[ncols] = -aw / asum;
                ncols++;
            }
            /* Right wall. */
            if (i == M - 1) {
                v[0] -= ae / asum;
            } else {
                col[ncols].i = i + 1;
                col[ncols].j = j;
                v[ncols] = -ae / asum;
                ncols++;
            }
            /* Bottom wall. */
            if (j == 0) {
                v[0] -= as / asum;
            } else {
                col[ncols].i = i;
                col[ncols].j = j - 1;
                v[ncols] = -as / asum;
                ncols++;
            }
            /* Top wall. */
            if (j == N - 1) {
                v[0] -= an / asum;
            } else {
                col[ncols].i = i;
                col[ncols].j = j + 1;
                v[ncols] = -an / asum;
                ncols++;
            }

            PetscCall(DMStagMatSetValuesStencil(cart->dm, Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
        }

    PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

    // TODO: below is temporary for cavity flow
    /* Remove null space. */
    PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
    PetscCall(MatSetNullSpace(J, nullspace));
    PetscCall(MatNullSpaceDestroy(&nullspace));

    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[0], cart->width[0], &arrwx));
    PetscCall(DMStagVecRestoreArrayRead(cart->subdm[1], cart->width[1], &arrwy));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, &arrcx, &arrcy, NULL));

    PetscFunctionReturn(PETSC_SUCCESS);
}
