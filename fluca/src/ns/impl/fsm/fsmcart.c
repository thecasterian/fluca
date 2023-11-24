#include <fluca/private/mesh_cartesian.h>
#include <fluca/private/ns_fsm.h>
#include <fluca/private/sol_fsm.h>
#include <petscdmda.h>
#include <petscdmstag.h>

#define DERIV2W (cart->a[0][i][0])
#define DERIV2E (cart->a[0][i][1])
#define DERIV2S (cart->a[1][j][0])
#define DERIV2N (cart->a[1][j][1])

extern PetscErrorCode NSFSMComputeRHSUStar2d(KSP ksp, Vec b, void *ctx);
extern PetscErrorCode NSFSMComputeRHSVStar2d(KSP ksp, Vec b, void *ctx);
extern PetscErrorCode NSFSMComputeOperatorsUVstar2d(KSP ksp, Mat J, Mat Jpre, void *ctx);
extern PetscErrorCode NSFSMComputeRHSPprime2d(KSP ksp, Vec b, void *ctx);
extern PetscErrorCode NSFSMComputeRHSPprime2d(KSP, Vec, void *);
extern PetscErrorCode NSFSMComputeOperatorPprime2d(KSP, Mat, Mat, void *);

PetscErrorCode NSFSMInterpolateVelocity2d(NS ns) {
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    DMDALocalInfo info;
    PetscReal ***arrUV;
    const PetscReal **arru, **arrv;
    PetscInt iU, iV;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMDAGetLocalInfo(cart->dm, &info));

    PetscCall(DMStagVecGetArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[1], &arrv));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm + 1; i++) {
            /* Left wall. */
            if (i == 0)
                arrUV[j][i][iU] = 0.0;
            /* Right wall. */
            else if (i == info.mx)
                arrUV[j][i][iU] = 0.0;
            else
                arrUV[j][i][iU] = (cart->w[0][i] * arru[j][i - 1] + cart->w[0][i - 1] * arru[j][i]) /
                                  (cart->w[0][i] + cart->w[0][i - 1]);
        }

    for (j = info.ys; j < info.ys + info.ym + 1; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            /* Bottom wall. */
            if (j == 0)
                arrUV[j][i][iV] = 0.0;
            /* Top wall. */
            else if (j == info.my)
                arrUV[j][i][iV] = 0.0;
            else
                arrUV[j][i][iV] = (cart->w[1][j] * arrv[j - 1][i] + cart->w[1][j - 1] * arrv[j][i]) /
                                  (cart->w[1][j] + cart->w[1][j - 1]);
        }

    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateConvection2d(NS ns) {
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    DMDALocalInfo info;
    PetscReal **arrNu, **arrNv;
    const PetscReal **arru, **arrv, ***arrUV;
    PetscInt iU, iV;
    PetscReal uw, ue, us, un, vw, ve, vs, vn;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMDAGetLocalInfo(cart->dm, &info));

    PetscCall(DMDAVecGetArray(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMDAVecGetArray(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecGetArrayRead(cart->fdm, solfsm->fv, &arrUV));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                uw = 0.0;
                vw = 0.0;
            } else {
                uw = (cart->w[0][i] * arru[j][i - 1] + cart->w[0][i - 1] * arru[j][i]) /
                     (cart->w[0][i] + cart->w[0][i - 1]);
                vw = (cart->w[0][i] * arrv[j][i - 1] + cart->w[0][i - 1] * arrv[j][i]) /
                     (cart->w[0][i] + cart->w[0][i - 1]);
            }
            /* Right wall. */
            if (i == info.mx - 1) {
                ue = 0.0;
                ve = 0.0;
            } else {
                ue = (cart->w[0][i] * arru[j][i + 1] + cart->w[0][i + 1] * arru[j][i]) /
                     (cart->w[0][i] + cart->w[0][i + 1]);
                ve = (cart->w[0][i] * arrv[j][i + 1] + cart->w[0][i + 1] * arrv[j][i]) /
                     (cart->w[0][i] + cart->w[0][i + 1]);
            }
            /* Bottom wall. */
            if (j == 0) {
                us = 0.0;
                vs = 0.0;
            } else {
                us = (cart->w[1][j] * arru[j - 1][i] + cart->w[1][j - 1] * arru[j][i]) /
                     (cart->w[1][j] + cart->w[1][j - 1]);
                vs = (cart->w[1][j] * arrv[j - 1][i] + cart->w[1][j - 1] * arrv[j][i]) /
                     (cart->w[1][j] + cart->w[1][j - 1]);
            }
            /* Top wall. */
            if (j == info.my - 1) {
                un = 0.0;
                vn = 0.0;
            } else {
                un = (cart->w[1][j] * arru[j + 1][i] + cart->w[1][j + 1] * arru[j][i]) /
                     (cart->w[1][j] + cart->w[1][j + 1]);
                vn = (cart->w[1][j] * arrv[j + 1][i] + cart->w[1][j + 1] * arrv[j][i]) /
                     (cart->w[1][j] + cart->w[1][j + 1]);
            }

            arrNu[j][i] = (arrUV[j][i + 1][iU] * ue - arrUV[j][i][iU] * uw) / cart->w[0][i] +
                          (arrUV[j + 1][i][iV] * un - arrUV[j][i][iV] * us) / cart->w[1][j];
            arrNv[j][i] = (arrUV[j][i + 1][iU] * ve - arrUV[j][i][iU] * vw) / cart->w[0][i] +
                          (arrUV[j + 1][i][iV] * vn - arrUV[j][i][iV] * vs) / cart->w[1][j];
        }

    PetscCall(DMDAVecRestoreArray(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMDAVecRestoreArray(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv, &arrUV));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateIntermediateVelocity2d(NS ns) {
    NS_FSM *nsfsm = (NS_FSM *)ns->data;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    PetscErrorCode (*rhs[2])(KSP, Vec, void *) = {NSFSMComputeRHSUStar2d, NSFSMComputeRHSVStar2d};
    Vec x;
    DMDALocalInfo info;
    PetscReal **arru_tilde, **arrv_tilde, ***arrUV_star;
    const PetscReal **arru_star, **arrv_star, **arrp;
    PetscInt iU, iV;
    PetscReal pw, pe, ps, pn, U_tilde, V_tilde, dpdx, dpdy;
    PetscInt d, i, j;

    PetscFunctionBegin;

    /* Solve for cell-centered intermediate velocity. */
    for (d = 0; d < 2; d++) {
        PetscCall(KSPSetComputeRHS(nsfsm->ksp, rhs[d], ns));
        PetscCall(KSPSetComputeOperators(nsfsm->ksp, NSFSMComputeOperatorsUVstar2d, ns));
        PetscCall(KSPSolve(nsfsm->ksp, NULL, NULL));
        PetscCall(KSPGetSolution(nsfsm->ksp, &x));
        PetscCall(DMGlobalToLocal(cart->dm, x, INSERT_VALUES, solfsm->v_star[d]));
    }

    /* Calculate face-centered intermediate velocity. */
    PetscCall(DMDAGetLocalInfo(cart->dm, &info));

    PetscCall(DMDAVecGetArray(cart->dm, solfsm->v_tilde[0], &arru_tilde));
    PetscCall(DMDAVecGetArray(cart->dm, solfsm->v_tilde[1], &arrv_tilde));
    PetscCall(DMStagVecGetArray(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                pw = arrp[j][i];
            } else {
                pw = arrp[j][i - 1];
            }
            /* Right wall. */
            if (i == info.mx - 1) {
                pe = arrp[j][i];
            } else {
                pe = arrp[j][i + 1];
            }
            /* Bottom wall. */
            if (j == 0) {
                ps = arrp[j][i];
            } else {
                ps = arrp[j - 1][i];
            }
            /* Top wall. */
            if (j == info.my - 1) {
                pn = arrp[j][i];
            } else {
                pn = arrp[j + 1][i];
            }

            arru_tilde[j][i] = arru_star[j][i] + ns->dt / ns->rho * (pe - pw) / (cart->c[0][i + 1] - cart->c[0][i - 1]);
            arrv_tilde[j][i] = arrv_star[j][i] + ns->dt / ns->rho * (pn - ps) / (cart->c[1][j + 1] - cart->c[1][j - 1]);
        }

    // PetscCall(DMDAVecRestoreArray(cart->dm, solfsm->v_tilde[0], &arru_tilde));
    // PetscCall(DMDAVecRestoreArray(cart->dm, solfsm->v_tilde[1], &arrv_tilde));

    PetscCall(DMLocalToLocalBegin(cart->dm, solfsm->v_tilde[0], INSERT_VALUES, solfsm->v_tilde[0]));
    PetscCall(DMLocalToLocalEnd(cart->dm, solfsm->v_tilde[0], INSERT_VALUES, solfsm->v_tilde[0]));
    PetscCall(DMLocalToLocalBegin(cart->dm, solfsm->v_tilde[1], INSERT_VALUES, solfsm->v_tilde[1]));
    PetscCall(DMLocalToLocalEnd(cart->dm, solfsm->v_tilde[1], INSERT_VALUES, solfsm->v_tilde[1]));

    // PetscCall(DMDAVecGetArray(cart->dm, solfsm->v_tilde[0], &arru_tilde));
    // PetscCall(DMDAVecGetArray(cart->dm, solfsm->v_tilde[1], &arrv_tilde));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm + 1; i++) {
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                arrUV_star[j][i][iU] = 0.0;
            }
            /* Right wall. */
            else if (i == info.mx) {
                arrUV_star[j][i][iU] = 0.0;
            } else {
                U_tilde = (cart->w[0][i] * arru_tilde[j][i - 1] + cart->w[0][i - 1] * arru_tilde[j][i]) /
                          (cart->w[0][i] + cart->w[0][i - 1]);
                dpdx = (arrp[j][i] - arrp[j][i - 1]) / (cart->c[0][i] - cart->c[0][i - 1]);
                arrUV_star[j][i][iU] = U_tilde - ns->dt / ns->rho * dpdx;
            }
        }
    for (j = info.ys; j < info.ys + info.ym + 1; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            // TODO: below is temporary for cavity flow
            /* Bottom wall. */
            if (j == 0) {
                arrUV_star[j][i][iV] = 0.0;
            }
            /* Top wall. */
            else if (j == info.my) {
                arrUV_star[j][i][iV] = 0.0;
            } else {
                V_tilde = (cart->w[1][j] * arrv_tilde[j - 1][i] + cart->w[1][j - 1] * arrv_tilde[j][i]) /
                          (cart->w[1][j] + cart->w[1][j - 1]);
                dpdy = (arrp[j][i] - arrp[j - 1][i]) / (cart->c[1][j] - cart->c[1][j - 1]);
                arrUV_star[j][i][iV] = V_tilde - ns->dt / ns->rho * dpdy;
            }
        }

    PetscCall(DMDAVecRestoreArray(cart->dm, solfsm->v_tilde[0], &arru_tilde));
    PetscCall(DMDAVecRestoreArray(cart->dm, solfsm->v_tilde[1], &arrv_tilde));
    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv_star, &arrUV_star));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculatePressureCorrection2d(NS ns) {
    NS_FSM *nsfsm = (NS_FSM *)ns->data;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    Vec x;

    PetscFunctionBegin;

    PetscCall(KSPSetComputeRHS(nsfsm->ksp, NSFSMComputeRHSPprime2d, ns));
    PetscCall(KSPSetComputeOperators(nsfsm->ksp, NSFSMComputeOperatorPprime2d, ns));
    PetscCall(KSPSolve(nsfsm->ksp, NULL, NULL));
    PetscCall(KSPGetSolution(nsfsm->ksp, &x));
    PetscCall(DMGlobalToLocal(cart->dm, x, INSERT_VALUES, solfsm->p_prime));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMUpdate2d(NS ns) {
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    DMDALocalInfo info;
    PetscReal **arru, **arrv, **arrp, ***arrUV;
    const PetscReal **arru_star, **arrv_star, **arrp_prime, ***arrUV_star;
    PetscInt iU, iV;
    PetscReal ppp, ppw, ppe, pps, ppn, pplap;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMDAGetLocalInfo(cart->dm, &info));

    PetscCall(VecCopy(solfsm->p_half, solfsm->p_half_prev));
    PetscCall(VecCopy(solfsm->N[0], solfsm->N_prev[0]));
    PetscCall(VecCopy(solfsm->N[1], solfsm->N_prev[1]));

    PetscCall(DMDAVecGetArray(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecGetArray(cart->dm, sol->v[1], &arrv));
    PetscCall(DMDAVecGetArray(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecGetArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->p_prime, &arrp_prime));
    PetscCall(DMStagVecGetArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            ppp = arrp_prime[j][i];

            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0)
                ppw = arrp_prime[j][i];
            else
                ppw = arrp_prime[j][i - 1];
            /* Right wall. */
            if (i == info.mx)
                ppe = arrp_prime[j][i];
            else
                ppe = arrp_prime[j][i + 1];
            /* Bottom wall. */
            if (j == 0)
                pps = arrp_prime[j][i];
            else
                pps = arrp_prime[j - 1][i];
            /* Top wall. */
            if (j == info.my)
                ppn = arrp_prime[j][i];
            else
                ppn = arrp_prime[j + 1][i];

            pplap = DERIV2W * ppw + DERIV2E * ppe + DERIV2S * pps + DERIV2N * ppn -
                    (DERIV2W + DERIV2E + DERIV2S + DERIV2N) * ppp;

            arru[j][i] = arru_star[j][i] - ns->dt / ns->rho * (ppe - ppw) / (cart->c[0][i + 1] - cart->c[0][i - 1]);
            arrv[j][i] = arrv_star[j][i] - ns->dt / ns->rho * (ppn - pps) / (cart->c[1][j + 1] - cart->c[1][j - 1]);
            arrp[j][i] += ppp - 0.5 * ns->mu * ns->dt / ns->rho * pplap;
        }

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm + 1; i++) {
            /* Left wall. */
            if (i == 0)
                arrUV[j][i][iU] = 0.0;
            /* Right wall. */
            else if (i == info.mx)
                arrUV[j][i][iU] = 0.0;
            else
                arrUV[j][i][iU] = arrUV_star[j][i][iU] - ns->dt / ns->rho * (arrp_prime[j][i] - arrp_prime[j][i - 1]) /
                                                             (cart->c[0][i] - cart->c[0][i - 1]);
        }

    for (j = info.ys; j < info.ys + info.ym + 1; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            /* Bottom wall. */
            if (j == 0)
                arrUV[j][i][iV] = 0.0;
            /* Top wall. */
            else if (j == info.my)
                arrUV[j][i][iV] = 0.0;
            else
                arrUV[j][i][iV] = arrUV_star[j][i][iV] - ns->dt / ns->rho * (arrp_prime[j][i] - arrp_prime[j - 1][i]) /
                                                             (cart->c[1][j] - cart->c[1][j - 1]);
        }

    PetscCall(DMDAVecRestoreArray(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecRestoreArray(cart->dm, sol->v[1], &arrv));
    PetscCall(DMDAVecRestoreArray(cart->dm, solfsm->p_half, &arrp));
    PetscCall(DMStagVecRestoreArray(cart->fdm, solfsm->fv, &arrUV));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->v_star[0], &arru_star));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->v_star[1], &arrv_star));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->p_prime, &arrp_prime));
    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));

    PetscCall(DMLocalToLocalBegin(cart->dm, sol->v[0], INSERT_VALUES, sol->v[0]));
    PetscCall(DMLocalToLocalEnd(cart->dm, sol->v[0], INSERT_VALUES, sol->v[0]));
    PetscCall(DMLocalToLocalBegin(cart->dm, sol->v[1], INSERT_VALUES, sol->v[1]));
    PetscCall(DMLocalToLocalEnd(cart->dm, sol->v[1], INSERT_VALUES, sol->v[1]));
    PetscCall(DMLocalToLocalBegin(cart->dm, solfsm->p_half, INSERT_VALUES, solfsm->p_half));
    PetscCall(DMLocalToLocalEnd(cart->dm, solfsm->p_half, INSERT_VALUES, solfsm->p_half));

    PetscCall(NSFSMCalculateConvection2d(ns));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputeRHSUStar2d(KSP ksp, Vec b, void *ctx) {
    (void)ksp;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    DMDALocalInfo info;
    PetscReal **arrb;
    const PetscReal **arrNu, **arrNu_prev, **arru, **arrv, **arrp;
    PetscReal up, uw, ue, us, un, pw, pe, dpdx, ulap;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMDAGetLocalInfo(cart->dm, &info));

    PetscCall(DMDAVecGetArray(cart->dm, b, &arrb));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->N_prev[0], &arrNu_prev));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            up = arru[j][i];
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0) {
                uw = -arru[j][i];
                pw = arrp[j][i];
            } else {
                uw = arru[j][i - 1];
                pw = arrp[j][i - 1];
            }
            /* Right wall. */
            if (i == info.mx - 1) {
                ue = -arru[j][i];
                pe = arrp[j][i];
            } else {
                ue = arru[j][i + 1];
                pe = arrp[j][i + 1];
            }
            /* Bottom wall. */
            if (j == 0)
                us = -arru[j][i];
            else
                us = arru[j - 1][i];
            /* Top wall. */
            if (j == info.my - 1)
                un = 2.0 - arru[j][i];
            else
                un = arru[j + 1][i];

            /* Pressure gradient. */
            dpdx = (pe - pw) / (cart->c[0][i + 1] - cart->c[0][i - 1]);
            /* Laplacian of u. */
            ulap = DERIV2W * uw + DERIV2E * ue + DERIV2S * us + DERIV2N * un -
                   (DERIV2W + DERIV2E + DERIV2S + DERIV2N) * up;

            arrb[j][i] = up - 1.5 * ns->dt * arrNu[j][i] + 0.5 * ns->dt * arrNu_prev[j][i] - ns->dt / ns->rho * dpdx +
                         0.5 * ns->mu * ns->dt / ns->rho * ulap;
            // TODO: below is temporary for cavity flow
            if (j == info.my - 1)
                arrb[j][i] += ns->mu * ns->dt / ns->rho * DERIV2N;
        }

    PetscCall(DMDAVecRestoreArray(cart->dm, b, &arrb));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->N[0], &arrNu));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->N_prev[0], &arrNu_prev));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));

    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputeRHSVStar2d(KSP ksp, Vec b, void *ctx) {
    (void)ksp;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    DMDALocalInfo info;
    PetscReal **arrb;
    const PetscReal **arrNv, **arrNv_prev, **arru, **arrv, **arrp;
    PetscReal vp, vw, ve, vs, vn, ps, pn, dpdy, vlap;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(DMDAGetLocalInfo(cart->dm, &info));

    PetscCall(DMDAVecGetArray(cart->dm, b, &arrb));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->N_prev[1], &arrNv_prev));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecGetArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMDAVecGetArrayRead(cart->dm, solfsm->p_half, &arrp));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            vp = arrv[j][i];
            // TODO: below is temporary for cavity flow
            /* Left wall. */
            if (i == 0)
                vw = -arrv[j][i];
            else
                vw = arrv[j][i - 1];
            /* Right wall. */
            if (i == info.mx - 1)
                ve = -arrv[j][i];
            else
                ve = arrv[j][i + 1];
            /* Bottom wall. */
            if (j == 0) {
                vs = -arrv[j][i];
                ps = arrp[j][i];
            } else {
                vs = arrv[j - 1][i];
                ps = arrp[j - 1][i];
            }
            /* Top wall. */
            if (j == info.my - 1) {
                vn = -arrv[j][i];
                pn = arrp[j][i];
            } else {
                vn = arrv[j + 1][i];
                pn = arrp[j + 1][i];
            }

            /* Pressure gradient. */
            dpdy = (pn - ps) / (cart->c[1][j + 1] - cart->c[1][j - 1]);
            /* Laplacian of v. */
            vlap = DERIV2W * vw + DERIV2E * ve + DERIV2S * vs + DERIV2N * vn -
                   (DERIV2W + DERIV2E + DERIV2S + DERIV2N) * vp;

            arrb[j][i] = vp - 1.5 * ns->dt * arrNv[j][i] + 0.5 * ns->dt * arrNv_prev[j][i] - ns->dt / ns->rho * dpdy +
                         0.5 * ns->mu * ns->dt / ns->rho * vlap;
        }

    PetscCall(DMDAVecRestoreArray(cart->dm, b, &arrb));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->N[1], &arrNv));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->N_prev[1], &arrNv_prev));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[0], &arru));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, sol->v[1], &arrv));
    PetscCall(DMDAVecRestoreArrayRead(cart->dm, solfsm->p_half, &arrp));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputeRHSPprime2d(KSP ksp, Vec b, void *ctx) {
    (void)ksp;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    Sol sol = ns->sol;
    Sol_FSM *solfsm = (Sol_FSM *)sol->data;

    MPI_Comm comm;
    DMDALocalInfo info;
    PetscReal **arrb;
    const PetscReal ***arrUV_star;
    PetscInt iU, iV;
    PetscReal divUV_star;
    MatNullSpace nullspace;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

    PetscCall(DMDAGetLocalInfo(cart->dm, &info));

    PetscCall(DMDAVecGetArray(cart->dm, b, &arrb));
    PetscCall(DMStagVecGetArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));

    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_LEFT, 0, &iU));
    PetscCall(DMStagGetLocationSlot(cart->fdm, DMSTAG_DOWN, 0, &iV));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            divUV_star = (arrUV_star[j][i + 1][iU] - arrUV_star[j][i][iU]) / cart->w[0][i] +
                         (arrUV_star[j + 1][i][iV] - arrUV_star[j][i][iV]) / cart->w[1][j];
            arrb[j][i] = -ns->rho / ns->dt * divUV_star / (DERIV2W + DERIV2E + DERIV2S + DERIV2N);
        }

    PetscCall(DMDAVecRestoreArray(cart->dm, b, &arrb));
    PetscCall(DMStagVecRestoreArrayRead(cart->fdm, solfsm->fv_star, &arrUV_star));

    // TODO: below is temporary for cavity flow
    /* Remove null space. */
    PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
    PetscCall(MatNullSpaceRemove(nullspace, b));
    PetscCall(MatNullSpaceDestroy(&nullspace));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputeOperatorsUVstar2d(KSP ksp, Mat J, Mat Jpre, void *ctx) {
    (void)J;

    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    DM da;
    DMDALocalInfo info;
    MatStencil row, col[5];
    PetscReal v[5];
    PetscInt ncols;
    PetscInt i, j;

    PetscFunctionBegin;

    // TODO: support multigrid method (da != cart->dm)
    PetscCall(KSPGetDM(ksp, &da));
    PetscCall(DMDAGetLocalInfo(da, &info));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
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
            if (i == info.mx - 1) {
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
            if (j == info.my - 1) {
                v[0] += 0.5 * ns->mu * ns->dt / ns->rho * DERIV2N;
            } else {
                col[ncols].i = i;
                col[ncols].j = j + 1;
                v[ncols] = -0.5 * ns->mu * ns->dt / ns->rho * DERIV2N;
                ncols++;
            }

            PetscCall(MatSetValuesStencil(Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
        }

    PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMComputeOperatorPprime2d(KSP ksp, Mat J, Mat Jpre, void *ctx) {
    NS ns = (NS)ctx;
    Mesh mesh = ns->mesh;
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    MPI_Comm comm;
    DM da;
    DMDALocalInfo info;
    MatStencil row, col[5];
    PetscReal v[5];
    PetscInt ncols;
    PetscReal asum;
    MatNullSpace nullspace;
    PetscInt i, j;

    PetscFunctionBegin;

    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

    // TODO: support multigrid method (da != cart->dm)
    PetscCall(KSPGetDM(ksp, &da));
    PetscCall(DMDAGetLocalInfo(da, &info));

    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
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
            if (i == info.mx - 1) {
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
            if (j == info.my - 1) {
                v[0] -= DERIV2N / asum;
            } else {
                col[ncols].i = i;
                col[ncols].j = j + 1;
                v[ncols] = -DERIV2N / asum;
                ncols++;
            }

            PetscCall(MatSetValuesStencil(Jpre, 1, &row, ncols, col, v, INSERT_VALUES));
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
