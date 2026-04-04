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

PetscErrorCode SegStep_SRK(Seg seg)
{
  Seg_SRK   *srk = (Seg_SRK *)seg->data;
  SRKTableau tab = srk->tableau;
  PetscReal  h   = seg->dt;
  PetscInt   s = tab->s, dim = srk->dim;
  PetscInt   i, j, d;
  PetscReal  gamma, tau_check, helm_shift, shift, alpha;
  PetscReal  stage_time, at_ij, a_ij;
  Vec        rhs_comp, sol_comp, rhs_p, sol_p;
  Vec        Z_d, sub, w_p, up_p, w_vel, up_vel;
  Vec        p_tilde; /* pressure prediction, aliases srk->mu_work */

  PetscFunctionBegin;
  PetscCheck(h > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Time step size must be positive");

  gamma      = tab->At[(s - 1) * s + (s - 1)]; /* a_ss */
  tau_check  = gamma * h;
  helm_shift = srk->rho / tau_check;
  shift      = 1. / tau_check;
  alpha      = 1. / h; /* Baumgarte parameter alpha = 1/tau (Section 5.1) */

  /* Reassemble Helmholtz matrices if dt changed */
  if (h != srk->dt_assembled) PetscCall(SegSRKAssembleHelmholtz(seg));

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
    stage_time = seg->t + tab->ct[i] * h;

    if (tab->At[i * s + i] == 0.) {
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
        PetscCall(VecAXPY(srk->K_hat_u[i], -1. / srk->rho, srk->work1));
      }

      /* mu_tilde[0] = -gamma * p_prev (Section 5.3, explicit first stage) */
      PetscCall(VecGetSubVector(srk->U_prev, srk->is_p, &up_p));
      PetscCall(VecCopy(up_p, srk->mu_tilde[i]));
      PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_p, &up_p));
      PetscCall(VecScale(srk->mu_tilde[i], -gamma));
    } else {
      /* --- Implicit stage (a_ii = gamma) --- */

      /* Step 1: Accumulate u_{i,*} in Z */
      PetscCall(VecCopy(srk->U_prev, srk->Z));
      for (j = 0; j < i; j++) {
        at_ij = tab->At[i * s + j];
        a_ij  = tab->A[i * s + j];
        if (at_ij != 0.) PetscCall(VecAXPY(srk->Z, h * at_ij, srk->K_u[j]));
        if (a_ij != 0.) PetscCall(VecAXPY(srk->Z, h * a_ij, srk->K_hat_u[j]));
      }

      /* Step 2: Helmholtz solve -- purely viscous, NO pressure gradient */
      for (d = 0; d < dim; d++) {
        PetscCall(VecGetSubVector(srk->Z, srk->is_comp[d], &Z_d));
        PetscCall(VecCopy(Z_d, rhs_comp));
        PetscCall(VecScale(rhs_comp, helm_shift));
        PetscCall(VecRestoreSubVector(srk->Z, srk->is_comp[d], &Z_d));

        PetscCall(KSPSolve(srk->ksp_helm[d], rhs_comp, sol_comp));

        PetscCall(VecGetSubVector(srk->Y[i], srk->is_comp[d], &sub));
        PetscCall(VecCopy(sol_comp, sub));
        PetscCall(VecRestoreSubVector(srk->Y[i], srk->is_comp[d], &sub));
      }

      /* Step 3: K_u[i] = shift * (Y[i] - Z), zero pressure DOFs */
      PetscCall(VecWAXPY(srk->K_u[i], -1., srk->Z, srk->Y[i]));
      PetscCall(VecScale(srk->K_u[i], shift));
      PetscCall(VecGetSubVector(srk->K_u[i], srk->is_p, &sub));
      PetscCall(VecZeroEntries(sub));
      PetscCall(VecRestoreSubVector(srk->K_u[i], srk->is_p, &sub));

      /* Step 4: Evaluate C^u via RHS callback (stored temporarily in K_hat_u[i]) */
      PetscCall(seg->rhsfn(stage_time, srk->Y[i], srk->K_hat_u[i], seg->rhsfn_ctx));

      /* Step 5: General mu/mu_tilde pressure prediction (Section 5.3)
         mu_j = p_prev + (1/gamma) * sum_{k<j} At[j][k] * mu_tilde[k]
         p_tilde_j = mu_j - alpha * tau_check * p_prev = mu_j - gamma * p_prev */
      p_tilde = srk->mu_work; /* reuse as p_tilde storage */

      /* mu_j = p_prev */
      PetscCall(VecGetSubVector(srk->U_prev, srk->is_p, &up_p));
      PetscCall(VecCopy(up_p, srk->mu_work));
      /* mu_j += (1/gamma) * sum_{k<j} At[j][k] * mu_tilde[k] */
      for (j = 0; j < i; j++) {
        at_ij = tab->At[i * s + j];
        if (at_ij != 0.) PetscCall(VecAXPY(srk->mu_work, at_ij / gamma, srk->mu_tilde[j]));
      }
      /* p_tilde = mu_j - gamma * p_prev */
      PetscCall(VecAXPY(p_tilde, -gamma, up_p));
      PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_p, &up_p));

      /* Step 6: Pressure Poisson solve
         A_pres * delta_p = -(1/rho) * fd_div(R_vel + alpha * U_prev_vel)
         where R_vel = K_u[i] + C^u[i] - G(p_tilde)/rho */

      /* work2 = K_u[i] + C^u[i] */
      PetscCall(VecCopy(srk->K_u[i], srk->work2));
      PetscCall(VecAXPY(srk->work2, 1., srk->K_hat_u[i]));

      /* Subtract G(p_tilde)/rho: place p_tilde into full-size pressure DOFs */
      PetscCall(VecZeroEntries(srk->work3));
      PetscCall(VecGetSubVector(srk->work3, srk->is_p, &w_p));
      PetscCall(VecCopy(p_tilde, w_p));
      PetscCall(VecRestoreSubVector(srk->work3, srk->is_p, &w_p));

      PetscCall(VecZeroEntries(srk->work1));
      PetscCall(ApplyGradP_Private(seg, srk->work3, srk->work1));
      PetscCall(VecAXPY(srk->work2, -1. / srk->rho, srk->work1));

      /* Baumgarte: add alpha * u_prev to R_vel so fd_div captures alpha * D * u^{n-1} */
      PetscCall(VecGetSubVector(srk->work2, srk->is_vel, &w_vel));
      PetscCall(VecGetSubVector(srk->U_prev, srk->is_vel, &up_vel));
      PetscCall(VecAXPY(w_vel, alpha, up_vel));
      PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_vel, &up_vel));
      PetscCall(VecRestoreSubVector(srk->work2, srk->is_vel, &w_vel));

      /* Apply divergence and extract pressure RHS */
      PetscCall(VecZeroEntries(srk->work1));
      PetscCall(FlucaFDApply(srk->fd_div, 0, seg->dm, seg->dm, srk->work2, srk->work1));
      PetscCall(VecGetSubVector(srk->work1, srk->is_p, &w_p));
      PetscCall(VecCopy(w_p, rhs_p));
      PetscCall(VecRestoreSubVector(srk->work1, srk->is_p, &w_p));
      PetscCall(VecScale(rhs_p, -1. / srk->rho));

      PetscCall(KSPSolve(srk->ksp_pres, rhs_p, sol_p));

      /* Step 7: Set pressure in Y[i]: p[i] = p_tilde + delta_p */
      PetscCall(VecGetSubVector(srk->Y[i], srk->is_p, &w_p));
      PetscCall(VecCopy(p_tilde, w_p));
      PetscCall(VecAXPY(w_p, 1., sol_p));
      PetscCall(VecRestoreSubVector(srk->Y[i], srk->is_p, &w_p));

      /* mu_tilde[i] = p[i] - mu_j
         mu_j = p_tilde + gamma*p_prev, so mu_tilde[i] = p[i] - p_tilde - gamma*p_prev */
      PetscCall(VecGetSubVector(srk->Y[i], srk->is_p, &w_p));
      PetscCall(VecCopy(w_p, srk->mu_tilde[i]));
      PetscCall(VecRestoreSubVector(srk->Y[i], srk->is_p, &w_p));
      PetscCall(VecAXPY(srk->mu_tilde[i], -1., p_tilde));
      PetscCall(VecGetSubVector(srk->U_prev, srk->is_p, &up_p));
      PetscCall(VecAXPY(srk->mu_tilde[i], -gamma, up_p));
      PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_p, &up_p));

      /* Step 8: K_hat_u[i] = C^u[i] - G(p[i])/rho */
      PetscCall(VecZeroEntries(srk->work1));
      PetscCall(ApplyGradP_Private(seg, srk->Y[i], srk->work1));
      PetscCall(VecAXPY(srk->K_hat_u[i], -1. / srk->rho, srk->work1));
    }
  }

  /* Solution update:
     y^n = y^{n-1} + h * sum_k bt[k]*K_u[k] + h * sum_k b[k]*K_hat_u[k]
     For combined stiffly accurate (At last row = bt, A last row = b), this equals Y[s-1]. */
  if (tab->stiffly_accurate && tab->explicit_stiffly_accurate) {
    PetscCall(VecCopy(srk->Y[s - 1], seg->sol));
  } else {
    /* General weighted update */
    PetscCall(VecCopy(srk->U_prev, seg->sol));
    for (i = 0; i < s; i++) {
      if (tab->bt[i] != 0.) PetscCall(VecAXPY(seg->sol, h * tab->bt[i], srk->K_u[i]));
      if (tab->b[i] != 0.) PetscCall(VecAXPY(seg->sol, h * tab->b[i], srk->K_hat_u[i]));
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
