#include <fluca/private/segsrkimpl.h>
#include "srktab.h"

/* Apply pressure gradient G(p) to a full-DM vector: accumulates into dst.
   Reads pressure from src, writes gradient to velocity DOFs of dst. */
static PetscErrorCode ApplyGradP_Private(Seg seg, Vec src, Vec dst)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;
  PetscInt d;

  PetscFunctionBegin;
  for (d = 0; d < srk->dim; d++) PetscCall(FlucaFDApply(srk->fd_grad_p[d], 0, seg->dm, seg->dm, src, dst));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Copy the pressure block out of a full-DM vector: dst = src[is_p]. */
static PetscErrorCode GetPressure_Private(Vec src, IS is_p, Vec dst)
{
  Vec sub;

  PetscFunctionBegin;
  PetscCall(VecGetSubVector(src, is_p, &sub));
  PetscCall(VecCopy(sub, dst));
  PetscCall(VecRestoreSubVector(src, is_p, &sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Copy a pressure-sized vector into the pressure block of a full-DM vector: dst[is_p] = src. */
static PetscErrorCode SetPressure_Private(Vec dst, IS is_p, Vec src)
{
  Vec sub;

  PetscFunctionBegin;
  PetscCall(VecGetSubVector(dst, is_p, &sub));
  PetscCall(VecCopy(src, sub));
  PetscCall(VecRestoreSubVector(dst, is_p, &sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegStep_SRK(Seg seg)
{
  Seg_SRK   *srk = (Seg_SRK *)seg->data;
  SRKTableau tab = srk->tableau;
  PetscReal  tau = seg->dt;
  PetscInt   s = tab->s, dim = srk->dim;
  PetscInt   i, j, d;
  PetscReal  gamma, tau_hat, inv_tau_hat, alpha, alpha_tau_hat;
  PetscReal  stage_time, a_ij, at_ij;
  Vec        rhs_comp, sol_comp, rhs_p, sol_p;
  Vec        Z_d, sub, w_p, up_p, w_vel, up_vel;
  Vec        p_tilde; /* pressure prediction, aliases srk->mu_work */

  PetscFunctionBegin;
  PetscCheck(tau > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Time step size must be positive");

  gamma         = tab->A[(s - 1) * s + (s - 1)];  /* a_ss */
  tau_hat       = gamma * tau;                    /* implicit per-stage timescale gamma*tau */
  inv_tau_hat   = 1. / tau_hat;                   /* Helmholtz shift and slope-recovery factor */
  alpha         = 0.5 * tab->alpha_tau_max / tau; /* Baumgarte: alpha*tau = 0.5*(alpha*tau)_max */
  alpha_tau_hat = alpha * tau_hat;                /* alpha * gamma * tau */

  /* Reassemble Helmholtz matrices if dt changed */
  if (tau != srk->dt_assembled) PetscCall(SegSRKAssembleHelmholtz(seg));

  /* Save U_prev = solution at start of step */
  PetscCall(VecCopy(seg->sol, srk->U_prev));

  /* Create temporary per-component and pressure vectors */
  rhs_comp = NULL;
  sol_comp = NULL;
  rhs_p    = NULL;
  sol_p    = NULL;
  PetscCall(MatCreateVecs(srk->A_helm[0], NULL, &rhs_comp));
  PetscCall(VecDuplicate(rhs_comp, &sol_comp));
  PetscCall(MatCreateVecs(srk->A_pres, NULL, &rhs_p));
  PetscCall(VecDuplicate(rhs_p, &sol_p));

  /* Allow KSPs to read options on first call */
  if (srk->first_step) {
    for (d = 0; d < dim; d++) PetscCall(KSPSetFromOptions(srk->ksp_helm[d]));
    PetscCall(KSPSetFromOptions(srk->ksp_pres));
  }

  /* === Stage loop === */
  for (i = 0; i < s; i++) {
    stage_time = seg->t + tab->c[i] * tau;

    if (tab->A[i * s + i] == 0.) {
      /* --- Explicit stage (a_ii = 0, must be stage 0 only) --- */
      PetscCheck(i == 0, PetscObjectComm((PetscObject)seg), PETSC_ERR_SUP, "Only the first stage may be explicit; stage %" PetscInt_FMT " has a_ii = 0", i);
      PetscCall(VecCopy(srk->U_prev, srk->Y[i]));
      PetscCall(VecZeroEntries(srk->K_u[i]));

      if (tab->fsal && !srk->first_step) {
        PetscCall(VecCopy(srk->K_hat_prev, srk->K_hat_u[i]));
      } else {
        PetscCall(seg->rhsfn(stage_time, srk->Y[i], srk->K_hat_u[i], seg->rhsfn_ctx));
        PetscCall(VecZeroEntries(srk->work1));
        PetscCall(ApplyGradP_Private(seg, srk->Y[i], srk->work1));
        PetscCall(VecAXPY(srk->K_hat_u[i], -1., srk->work1));
      }

      /* mu_tilde[0] = -alpha_tau_hat * p_prev (Section 5.3, explicit first stage) */
      PetscCall(GetPressure_Private(srk->U_prev, srk->is_p, srk->mu_tilde[i]));
      PetscCall(VecScale(srk->mu_tilde[i], -alpha_tau_hat));

      /* CK non-ARS: compute K_u[0] = F_diff(u_1) and nu_tilde_1 */
      if (!tab->ars_type) {
        /* K_u[0] = F_diff(u_1) */
        PetscCall(VecZeroEntries(srk->K_u[i]));
        for (d = 0; d < dim; d++) PetscCall(FlucaFDApply(srk->fd_diff[d], 0, seg->dm, seg->dm, srk->Y[i], srk->K_u[i]));

        /* nu_tilde_1 = D(K_1^u + K_hat_1^u + alpha * u^{n-1}) */
        PetscCall(VecCopy(srk->K_u[i], srk->work2));
        PetscCall(VecAXPY(srk->work2, 1., srk->K_hat_u[i]));
        PetscCall(VecGetSubVector(srk->work2, srk->is_vel, &w_vel));
        PetscCall(VecGetSubVector(srk->U_prev, srk->is_vel, &up_vel));
        PetscCall(VecAXPY(w_vel, alpha, up_vel));
        PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_vel, &up_vel));
        PetscCall(VecRestoreSubVector(srk->work2, srk->is_vel, &w_vel));
        PetscCall(VecZeroEntries(srk->work1));
        PetscCall(FlucaFDApply(srk->fd_div, 0, seg->dm, seg->dm, srk->work2, srk->work1));
        PetscCall(GetPressure_Private(srk->work1, srk->is_p, srk->nu_tilde_1));
      }
    } else {
      /* --- Implicit stage (a_ii = gamma) --- */

      /* Step 1: Accumulate u_{i,*} in Z */
      PetscCall(VecCopy(srk->U_prev, srk->Z));
      for (j = 0; j < i; j++) {
        a_ij  = tab->A[i * s + j];
        at_ij = tab->At[i * s + j];
        if (a_ij != 0.) PetscCall(VecAXPY(srk->Z, tau * a_ij, srk->K_u[j]));
        if (at_ij != 0.) PetscCall(VecAXPY(srk->Z, tau * at_ij, srk->K_hat_u[j]));
      }

      /* Step 2: Helmholtz solve -- purely viscous, NO pressure gradient */
      for (d = 0; d < dim; d++) {
        PetscCall(VecGetSubVector(srk->Z, srk->is_comp[d], &Z_d));
        PetscCall(VecCopy(Z_d, rhs_comp));
        PetscCall(VecScale(rhs_comp, inv_tau_hat));
        PetscCall(VecRestoreSubVector(srk->Z, srk->is_comp[d], &Z_d));

        PetscCall(KSPSolve(srk->ksp_helm[d], rhs_comp, sol_comp));

        PetscCall(VecGetSubVector(srk->Y[i], srk->is_comp[d], &sub));
        PetscCall(VecCopy(sol_comp, sub));
        PetscCall(VecRestoreSubVector(srk->Y[i], srk->is_comp[d], &sub));
      }

      /* Step 3: K_u[i] = shift * (Y[i] - Z), zero pressure DOFs */
      PetscCall(VecWAXPY(srk->K_u[i], -1., srk->Z, srk->Y[i]));
      PetscCall(VecScale(srk->K_u[i], inv_tau_hat));
      PetscCall(VecGetSubVector(srk->K_u[i], srk->is_p, &sub));
      PetscCall(VecZeroEntries(sub));
      PetscCall(VecRestoreSubVector(srk->K_u[i], srk->is_p, &sub));

      /* Step 4: Evaluate C^u via RHS callback (stored temporarily in K_hat_u[i]) */
      PetscCall(seg->rhsfn(stage_time, srk->Y[i], srk->K_hat_u[i], seg->rhsfn_ctx));

      /* Step 5: General mu/mu_tilde pressure prediction (Section 5.3)
         mu_j = p_prev + (1/gamma) * sum_{k<j} A[j][k] * mu_tilde[k]
         p_tilde_j = mu_j - alpha * tau_hat * p_prev */
      p_tilde = srk->mu_work; /* reuse as p_tilde storage */

      /* mu_j = p_prev */
      PetscCall(VecGetSubVector(srk->U_prev, srk->is_p, &up_p));
      PetscCall(VecCopy(up_p, srk->mu_work));
      /* mu_j += (1/gamma) * sum_{k<j} A[j][k] * mu_tilde[k] */
      for (j = 0; j < i; j++) {
        a_ij = tab->A[i * s + j];
        if (a_ij != 0.) PetscCall(VecAXPY(srk->mu_work, a_ij / gamma, srk->mu_tilde[j]));
      }
      /* p_tilde = mu_j - alpha_tau_hat * p_prev */
      PetscCall(VecAXPY(p_tilde, -alpha_tau_hat, up_p));
      PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_p, &up_p));

      /* Step 6: Pressure Poisson solve (equation 10)
         A_pres * delta_p = -D(K_u + C^u - G(p_tilde) + alpha * u^{n-1})
         where D = fd_div (includes rho), A_pres = -L */

      /* work2 = K_u[i] + C^u[i] */
      PetscCall(VecCopy(srk->K_u[i], srk->work2));
      PetscCall(VecAXPY(srk->work2, 1., srk->K_hat_u[i]));

      /* Subtract G(p_tilde): place p_tilde into full-size pressure DOFs */
      PetscCall(VecZeroEntries(srk->work3));
      PetscCall(SetPressure_Private(srk->work3, srk->is_p, p_tilde));

      PetscCall(VecZeroEntries(srk->work1));
      PetscCall(ApplyGradP_Private(seg, srk->work3, srk->work1));
      PetscCall(VecAXPY(srk->work2, -1., srk->work1));

      /* Baumgarte: add alpha * u_prev to R_vel so fd_div captures alpha * D * u^{n-1} */
      PetscCall(VecGetSubVector(srk->work2, srk->is_vel, &w_vel));
      PetscCall(VecGetSubVector(srk->U_prev, srk->is_vel, &up_vel));
      PetscCall(VecAXPY(w_vel, alpha, up_vel));
      PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_vel, &up_vel));
      PetscCall(VecRestoreSubVector(srk->work2, srk->is_vel, &w_vel));

      /* Apply divergence and extract pressure RHS */
      PetscCall(VecZeroEntries(srk->work1));
      PetscCall(FlucaFDApply(srk->fd_div, 0, seg->dm, seg->dm, srk->work2, srk->work1));
      PetscCall(GetPressure_Private(srk->work1, srk->is_p, rhs_p));
      PetscCall(VecScale(rhs_p, -1.));

      /* CK non-ARS: add nu_j = d_j * nu_tilde_1 (equation 10, -nu_j on LHS becomes +nu_j on RHS
         since A_pres = -L; see theory guide Section "Stage Algorithm", stages j >= 2) */
      if (!tab->ars_type) PetscCall(VecAXPY(rhs_p, srk->d_j[i], srk->nu_tilde_1));

      PetscCall(KSPSolve(srk->ksp_pres, rhs_p, sol_p));

      /* Step 7: Set pressure in Y[i]: p[i] = p_tilde + delta_p */
      PetscCall(VecGetSubVector(srk->Y[i], srk->is_p, &w_p));
      PetscCall(VecCopy(p_tilde, w_p));
      PetscCall(VecAXPY(w_p, 1., sol_p));
      PetscCall(VecRestoreSubVector(srk->Y[i], srk->is_p, &w_p));

      /* mu_tilde[i] = p[i] - mu_j
         mu_j = p_tilde + alpha_tau_hat*p_prev, so mu_tilde[i] = p[i] - p_tilde - alpha_tau_hat*p_prev */
      PetscCall(GetPressure_Private(srk->Y[i], srk->is_p, srk->mu_tilde[i]));
      PetscCall(VecAXPY(srk->mu_tilde[i], -1., p_tilde));
      PetscCall(VecGetSubVector(srk->U_prev, srk->is_p, &up_p));
      PetscCall(VecAXPY(srk->mu_tilde[i], -alpha_tau_hat, up_p));
      PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_p, &up_p));

      /* Step 8: K_hat_u[i] = C^u[i] - G(p[i]) */
      PetscCall(VecZeroEntries(srk->work1));
      PetscCall(ApplyGradP_Private(seg, srk->Y[i], srk->work1));
      PetscCall(VecAXPY(srk->K_hat_u[i], -1., srk->work1));
    }
  }

  /* Solution update:
     y^n = y^{n-1} + tau * sum_k b[k]*K_u[k] + tau * sum_k bt[k]*K_hat_u[k]
     For combined stiffly accurate (A last row = b, At last row = bt), this equals Y[s-1]. */
  if (tab->stiffly_accurate && tab->explicit_stiffly_accurate) {
    PetscCall(VecCopy(srk->Y[s - 1], seg->sol));
  } else {
    /* General weighted update */
    PetscCall(VecCopy(srk->U_prev, seg->sol));
    for (i = 0; i < s; i++) {
      if (tab->b[i] != 0.) PetscCall(VecAXPY(seg->sol, tau * tab->b[i], srk->K_u[i]));
      if (tab->bt[i] != 0.) PetscCall(VecAXPY(seg->sol, tau * tab->bt[i], srk->K_hat_u[i]));
    }
    /* Pressure: the weighted velocity update above only affects velocity DOFs
       (K_u and K_hat_u have zero pressure DOFs). Reconstruct pressure from the
       last stage solve, which is the best available pressure approximation. */
    PetscCall(VecGetSubVector(srk->Y[s - 1], srk->is_p, &w_p));
    PetscCall(VecGetSubVector(seg->sol, srk->is_p, &sub));
    PetscCall(VecCopy(w_p, sub));
    PetscCall(VecRestoreSubVector(seg->sol, srk->is_p, &sub));
    PetscCall(VecRestoreSubVector(srk->Y[s - 1], srk->is_p, &w_p));
  }

  /* FSAL: save K_hat_u for next step */
  if (tab->fsal) PetscCall(VecCopy(srk->K_hat_u[s - 1], srk->K_hat_prev));

  srk->first_step = PETSC_FALSE;

  PetscCall(VecDestroy(&rhs_comp));
  PetscCall(VecDestroy(&sol_comp));
  PetscCall(VecDestroy(&rhs_p));
  PetscCall(VecDestroy(&sol_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}
