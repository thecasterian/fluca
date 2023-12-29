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
    PetscInt iU, iV, ielem;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[1], &arrv));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm + 1; i++) {
            /* Left wall. */
            if (i == 0)
                arrUV[j][i][iU] = 0.0;
            /* Right wall. */
            else if (i == M)
                arrUV[j][i][iU] = 0.0;
            else
                arrUV[j][i][iU] = (cart->w[0][i] * arru[j][i - 1][ielem] + cart->w[0][i - 1] * arru[j][i][ielem]) /
                                  (cart->w[0][i] + cart->w[0][i - 1]);
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
                arrUV[j][i][iV] = (cart->w[1][j] * arrv[j - 1][i][ielem] + cart->w[1][j - 1] * arrv[j][i][ielem]) /
                                  (cart->w[1][j] + cart->w[1][j - 1]);
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
    PetscInt iU, iV, ielem;
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

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                uw = 0.0;
                vw = 0.0;
            } else {
                uw = (cart->w[0][i] * arru[j][i - 1][ielem] + cart->w[0][i - 1] * arru[j][i][ielem]) /
                     (cart->w[0][i] + cart->w[0][i - 1]);
                vw = (cart->w[0][i] * arrv[j][i - 1][ielem] + cart->w[0][i - 1] * arrv[j][i][ielem]) /
                     (cart->w[0][i] + cart->w[0][i - 1]);
            }
            /* Right wall. */
            if (i == M - 1) {
                ue = 0.0;
                ve = 0.0;
            } else {
                ue = (cart->w[0][i] * arru[j][i + 1][ielem] + cart->w[0][i + 1] * arru[j][i][ielem]) /
                     (cart->w[0][i] + cart->w[0][i + 1]);
                ve = (cart->w[0][i] * arrv[j][i + 1][ielem] + cart->w[0][i + 1] * arrv[j][i][ielem]) /
                     (cart->w[0][i] + cart->w[0][i + 1]);
            }
            /* Bottom wall. */
            if (j == 0) {
                us = 0.0;
                vs = 0.0;
            } else {
                us = (cart->w[1][j] * arru[j - 1][i][ielem] + cart->w[1][j - 1] * arru[j][i][ielem]) /
                     (cart->w[1][j] + cart->w[1][j - 1]);
                vs = (cart->w[1][j] * arrv[j - 1][i][ielem] + cart->w[1][j - 1] * arrv[j][i][ielem]) /
                     (cart->w[1][j] + cart->w[1][j - 1]);
            }
            /* Top wall. */
            if (j == N - 1) {
                un = 0.0;
                vn = 0.0;
            } else {
                un = (cart->w[1][j] * arru[j + 1][i][ielem] + cart->w[1][j + 1] * arru[j][i][ielem]) /
                     (cart->w[1][j] + cart->w[1][j + 1]);
                vn = (cart->w[1][j] * arrv[j + 1][i][ielem] + cart->w[1][j + 1] * arrv[j][i][ielem]) /
                     (cart->w[1][j] + cart->w[1][j + 1]);
            }

            arrNu[j][i][ielem] = (arrUV[j][i + 1][iU] * ue - arrUV[j][i][iU] * uw) / cart->w[0][i] +
                                 (arrUV[j + 1][i][iV] * un - arrUV[j][i][iV] * us) / cart->w[1][j];
            arrNv[j][i][ielem] = (arrUV[j][i + 1][iU] * ve - arrUV[j][i][iU] * vw) / cart->w[0][i] +
                                 (arrUV[j + 1][i][iV] * vn - arrUV[j][i][iV] * vs) / cart->w[1][j];
        }

    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv, &arrUV));

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
    PetscInt iU, iV, ielem;
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

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));

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
                arru_star[j][i][ielem] + ns->dt / ns->rho * (pe - pw) / (cart->c[0][i + 1] - cart->c[0][i - 1]);
            arrv_tilde[j][i][ielem] =
                arrv_star[j][i][ielem] + ns->dt / ns->rho * (pn - ps) / (cart->c[1][j + 1] - cart->c[1][j - 1]);
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
                U_tilde = (cart->w[0][i] * arru_tilde[j][i - 1][ielem] + cart->w[0][i - 1] * arru_tilde[j][i][ielem]) /
                          (cart->w[0][i] + cart->w[0][i - 1]);
                dpdx = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (cart->c[0][i] - cart->c[0][i - 1]);
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
                V_tilde = (cart->w[1][j] * arrv_tilde[j - 1][i][ielem] + cart->w[1][j - 1] * arrv_tilde[j][i][ielem]) /
                          (cart->w[1][j] + cart->w[1][j - 1]);
                dpdy = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (cart->c[1][j] - cart->c[1][j - 1]);
                arrUV_star[j][i][iV] = V_tilde - ns->dt / ns->rho * dpdy;
            }
        }

    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->v_tilde[0], &arru_tilde));
    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->v_tilde[1], &arrv_tilde));
    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));

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
    PetscInt iU, iV, ielem;
    PetscReal ppp, ppw, ppe, pps, ppn, pplap;
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

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
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

            pplap = DERIV2W * ppw + DERIV2E * ppe + DERIV2S * pps + DERIV2N * ppn -
                    (DERIV2W + DERIV2E + DERIV2S + DERIV2N) * ppp;

            arru[j][i][ielem] =
                arru_star[j][i][ielem] - ns->dt / ns->rho * (ppe - ppw) / (cart->c[0][i + 1] - cart->c[0][i - 1]);
            arrv[j][i][ielem] =
                arrv_star[j][i][ielem] - ns->dt / ns->rho * (ppn - pps) / (cart->c[1][j + 1] - cart->c[1][j - 1]);
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
                                                             (cart->c[0][i] - cart->c[0][i - 1]);
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
                                                             (cart->c[1][j] - cart->c[1][j - 1]);
        }

    PetscCall(DMStagVecRestoreArray(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArray(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArray(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_prime, &arrp_prime));
    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));

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
    PetscReal up, uw, ue, us, un, pw, pe, dpdx, ulap;
    PetscInt ielem;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N_prev[0], &arrNu_prev));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));

    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    row.loc = DMSTAG_ELEMENT;
    row.c = 0;

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
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
            dpdx = (pe - pw) / (cart->c[0][i + 1] - cart->c[0][i - 1]);
            /* Laplacian of u. */
            ulap = DERIV2W * uw + DERIV2E * ue + DERIV2S * us + DERIV2N * un -
                   (DERIV2W + DERIV2E + DERIV2S + DERIV2N) * up;

            valb = up - 1.5 * ns->dt * arrNu[j][i][ielem] + 0.5 * ns->dt * arrNu_prev[j][i][ielem] -
                   ns->dt / ns->rho * dpdx + 0.5 * ns->mu * ns->dt / ns->rho * ulap;
            // TODO: below is temporary for cavity flow
            if (j == N - 1)
                valb += ns->mu * ns->dt / ns->rho * DERIV2N;

            PetscCall(DMStagVecSetValuesStencil(cart->dm, b, 1, &row, &valb, INSERT_VALUES));
        }

    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N_prev[0], &arrNu_prev));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));

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
    PetscReal vp, vw, ve, vs, vn, ps, pn, dpdy, vlap;
    PetscInt ielem;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->N_prev[1], &arrNv_prev));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));

    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    row.loc = DMSTAG_ELEMENT;
    row.c = 0;

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
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
            dpdy = (pn - ps) / (cart->c[1][j + 1] - cart->c[1][j - 1]);
            /* Laplacian of v. */
            vlap = DERIV2W * vw + DERIV2E * ve + DERIV2S * vs + DERIV2N * vn -
                   (DERIV2W + DERIV2E + DERIV2S + DERIV2N) * vp;

            valb = vp - 1.5 * ns->dt * arrNv[j][i][ielem] + 0.5 * ns->dt * arrNv_prev[j][i][ielem] -
                   ns->dt / ns->rho * dpdy + 0.5 * ns->mu * ns->dt / ns->rho * vlap;

            PetscCall(DMStagVecSetValuesStencil(cart->dm, b, 1, &row, &valb, INSERT_VALUES));
        }

    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->N_prev[1], &arrNv_prev));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));

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
    PetscInt iU, iV, ielem;
    PetscReal divUV_star;
    MatNullSpace nullspace;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    PetscCall(DMStagVecGetArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));
    PetscCall(DMStagGetLocationSlot(cart->dm, DMSTAG_ELEMENT, 0, &ielem));
    row.loc = DMSTAG_ELEMENT;
    row.c = 0;

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            row.i = i;
            row.j = j;

            divUV_star = (arrUV_star[j][i + 1][iU] - arrUV_star[j][i][iU]) / cart->w[0][i] +
                         (arrUV_star[j + 1][i][iV] - arrUV_star[j][i][iV]) / cart->w[1][j];
            valb = -ns->rho / ns->dt * divUV_star / (DERIV2W + DERIV2E + DERIV2S + DERIV2N);

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
    PetscInt i, j;

    PetscFunctionBegin;

    // TODO: support multigrid method (dm of KSP != cart->dm)

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    row.loc = DMSTAG_ELEMENT;
    row.c = 0;
    for (ncols = 0; ncols < 5; ncols++) {
        col[ncols].loc = DMSTAG_ELEMENT;
        col[ncols].c = 0;
    }

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            row.i = col[0].i = i;
            row.j = col[0].j = j;
            v[0] = 1.0 + 0.5 * ns->mu * ns->dt / ns->rho * (DERIV2W + DERIV2E + DERIV2S + DERIV2N);
            v[1] = v[2] = v[3] = v[4] = 0;
            ncols = 1;

            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * DERIV2W;
            } else {
                col[ncols].i = i - 1;
                col[ncols].j = j;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * DERIV2W;
                ncols++;
            }
            /* Right wall. */
            if (i == M - 1) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * DERIV2E;
            } else {
                col[ncols].i = i + 1;
                col[ncols].j = j;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * DERIV2E;
                ncols++;
            }
            /* Bottom wall. */
            if (j == 0) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * DERIV2S;
            } else {
                col[ncols].i = i;
                col[ncols].j = j - 1;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * DERIV2S;
                ncols++;
            }
            /* Top wall. */
            if (j == N - 1) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * DERIV2N;
            } else {
                col[ncols].i = i;
                col[ncols].j = j + 1;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * DERIV2N;
                ncols++;
            }

            PetscCall(DMStagMatSetValuesStencil(cart->dm, Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
        }

    PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

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
    PetscReal asum;
    MatNullSpace nullspace;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

    // TODO: support multigrid method (dm of KSP != cart->dm)

    PetscCall(DMStagGetGlobalSizes(cart->dm, &M, &N, NULL));
    PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

    row.loc = DMSTAG_ELEMENT;
    row.c = 0;
    for (ncols = 0; ncols < 5; ncols++) {
        col[ncols].loc = DMSTAG_ELEMENT;
        col[ncols].c = 0;
    }

    for (j = ys; j < ys + ym; j++)
        for (i = xs; i < xs + xm; i++) {
            row.i = col[0].i = i;
            row.j = col[0].j = j;
            v[0] = 1.0;
            v[1] = v[2] = v[3] = v[4] = 0;
            ncols = 1;

            asum = DERIV2W + DERIV2E + DERIV2S + DERIV2N;

            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                v[0] -= DERIV2W / asum;
            } else {
                col[ncols].i = i - 1;
                col[ncols].j = j;
                v[ncols] = -DERIV2W / asum;
                ncols++;
            }
            /* Right wall. */
            if (i == M - 1) {
                v[0] -= DERIV2E / asum;
            } else {
                col[ncols].i = i + 1;
                col[ncols].j = j;
                v[ncols] = -DERIV2E / asum;
                ncols++;
            }
            /* Bottom wall. */
            if (j == 0) {
                v[0] -= DERIV2S / asum;
            } else {
                col[ncols].i = i;
                col[ncols].j = j - 1;
                v[ncols] = -DERIV2S / asum;
                ncols++;
            }
            /* Top wall. */
            if (j == N - 1) {
                v[0] -= DERIV2N / asum;
            } else {
                col[ncols].i = i;
                col[ncols].j = j + 1;
                v[ncols] = -DERIV2N / asum;
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

    PetscFunctionReturn(PETSC_SUCCESS);
}
