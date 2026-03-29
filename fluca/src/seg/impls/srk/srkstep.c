#include <fluca/private/segsrkimpl.h>

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

/* Compute the pressure prediction p_tilde for stage j.
   For ARKIMEX L2 with sigma_1=0, r_sigma=0, alpha=1/tau (Baumgarte):
     Stage 0: p_tilde = not used (no pressure solve, p_0 = p^{n-1})
     Stage 1: p_tilde = 0 (the Baumgarte correction cancels the accumulated pressure)
   Returns p_tilde as a scalar coefficient on p_prev (for periodic BCs with H=0). */
static PetscReal ComputePTildeCoeff_Private(Seg_SRK *srk, PetscInt stage, PetscReal h)
{
  PetscReal gamma, alpha, tau_check, mu_coeff, p_tilde_coeff;
  PetscInt  s;

  s         = srk->s;
  gamma     = srk->At[(s - 1) * s + (s - 1)]; /* a_ss */
  alpha     = 1. / h;
  tau_check = gamma * h;

  /* mu_tilde_coeff[0] = -gamma (Section 5.3: mu_tilde_1 = -alpha*tau_check*p_prev) */
  mu_coeff = 1. + (1. / gamma) * srk->At[stage * s + 0] * (-gamma);
  /* For L2 stage 1: mu_coeff = 1 + (1/gamma)*(1-gamma)*(-gamma) = gamma */

  p_tilde_coeff = mu_coeff - alpha * tau_check;
  /* For L2 stage 1: p_tilde_coeff = gamma - gamma = 0 */

  return p_tilde_coeff;
}

PetscErrorCode SegStep_SRK(Seg seg)
{
  Seg_SRK  *srk = (Seg_SRK *)seg->data;
  PetscReal h   = seg->dt;
  PetscInt  s = srk->s, dim = srk->dim;
  PetscInt  i, j, d;
  PetscReal gamma, tau_check, helm_shift, shift, alpha;
  PetscReal stage_time, p_tilde_coeff, at_ij, a_ij;
  Vec       rhs_comp, sol_comp, rhs_p, sol_p;
  Vec       Z_d, sub, w_p, up_p, w_vel, up_vel;

  PetscFunctionBegin;
  PetscCheck(h > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Time step size must be positive");

  gamma      = srk->At[(s - 1) * s + (s - 1)]; /* a_ss */
  tau_check  = gamma * h;
  helm_shift = srk->rho / tau_check;
  shift      = 1. / tau_check;
  alpha      = 1. / h; /* Baumgarte parameter α = 1/τ (Section 5.1) */

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
    stage_time = seg->t + srk->ct[i] * h;

    if (srk->At[i * s + i] == 0.) {
      /* --- Explicit stage (i=0 for L2) --- */
      PetscCall(VecCopy(srk->U_prev, srk->Y[i]));
      PetscCall(VecZeroEntries(srk->K_u[i]));

      if (srk->fsal && !srk->first_step) {
        PetscCall(VecCopy(srk->K_hat_prev, srk->K_hat_u[i]));
      } else {
        PetscCall(seg->rhsfn(stage_time, srk->Y[i], srk->K_hat_u[i], seg->rhsfn_ctx));
        PetscCall(VecZeroEntries(srk->work1));
        PetscCall(ApplyGradP_Private(seg, srk->Y[i], srk->work1));
        PetscCall(VecAXPY(srk->K_hat_u[i], -1. / srk->rho, srk->work1));
      }
    } else {
      /* --- Implicit stage (i>=1 for L2) --- */

      /* Step 1: Accumulate u_{i,*} in Z */
      PetscCall(VecCopy(srk->U_prev, srk->Z));
      for (j = 0; j < i; j++) {
        at_ij = srk->At[i * s + j];
        a_ij  = srk->A[i * s + j];
        if (at_ij != 0.) PetscCall(VecAXPY(srk->Z, h * at_ij, srk->K_u[j]));
        if (a_ij != 0.) PetscCall(VecAXPY(srk->Z, h * a_ij, srk->K_hat_u[j]));
      }

      /* Step 2: Helmholtz solve — purely viscous, NO pressure gradient */
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

      /* Step 5: Pressure prediction */
      p_tilde_coeff = ComputePTildeCoeff_Private(srk, i, h);

      /* Step 6: Pressure Poisson solve
         A_pres * delta_p = -(1/rho) * fd_div(R_vel + alpha * U_prev_vel)
         where R_vel = K_u[i] + C^u[i] - G(p_tilde)/rho */

      /* work2 = K_u[i] + C^u[i] */
      PetscCall(VecCopy(srk->K_u[i], srk->work2));
      PetscCall(VecAXPY(srk->work2, 1., srk->K_hat_u[i]));

      /* Subtract G(p_tilde)/rho if p_tilde != 0 */
      if (p_tilde_coeff != 0.) {
        PetscCall(VecZeroEntries(srk->work3));
        PetscCall(VecGetSubVector(srk->work3, srk->is_p, &w_p));
        PetscCall(VecGetSubVector(srk->U_prev, srk->is_p, &up_p));
        PetscCall(VecCopy(up_p, w_p));
        PetscCall(VecScale(w_p, p_tilde_coeff));
        PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_p, &up_p));
        PetscCall(VecRestoreSubVector(srk->work3, srk->is_p, &w_p));

        PetscCall(VecZeroEntries(srk->work1));
        PetscCall(ApplyGradP_Private(seg, srk->work3, srk->work1));
        PetscCall(VecAXPY(srk->work2, -1. / srk->rho, srk->work1));
      }

      /* Baumgarte: add α·u_prev to R_vel so fd_div captures α·D·u^{n-1} */
      if (alpha != 0.) {
        PetscCall(VecGetSubVector(srk->work2, srk->is_vel, &w_vel));
        PetscCall(VecGetSubVector(srk->U_prev, srk->is_vel, &up_vel));
        PetscCall(VecAXPY(w_vel, alpha, up_vel));
        PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_vel, &up_vel));
        PetscCall(VecRestoreSubVector(srk->work2, srk->is_vel, &w_vel));
      }

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
      if (p_tilde_coeff != 0.) {
        PetscCall(VecGetSubVector(srk->U_prev, srk->is_p, &up_p));
        PetscCall(VecCopy(up_p, w_p));
        PetscCall(VecScale(w_p, p_tilde_coeff));
        PetscCall(VecRestoreSubVector(srk->U_prev, srk->is_p, &up_p));
        PetscCall(VecAXPY(w_p, 1., sol_p));
      } else {
        PetscCall(VecCopy(sol_p, w_p));
      }
      PetscCall(VecRestoreSubVector(srk->Y[i], srk->is_p, &w_p));

      /* Step 8: K_hat_u[i] = C^u[i] - G(p[i])/rho */
      PetscCall(VecZeroEntries(srk->work1));
      PetscCall(ApplyGradP_Private(seg, srk->Y[i], srk->work1));
      PetscCall(VecAXPY(srk->K_hat_u[i], -1. / srk->rho, srk->work1));
    }
  }

  /* Solution update */
  if (srk->stiffly_accurate) PetscCall(VecCopy(srk->Y[s - 1], seg->sol));

  /* FSAL: save K_hat_u for next step */
  if (srk->fsal) PetscCall(VecCopy(srk->K_hat_u[s - 1], srk->K_hat_prev));

  srk->first_step = PETSC_FALSE;

  PetscCall(VecDestroy(&rhs_comp));
  PetscCall(VecDestroy(&sol_comp));
  PetscCall(VecDestroy(&rhs_p));
  PetscCall(VecDestroy(&sol_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}
