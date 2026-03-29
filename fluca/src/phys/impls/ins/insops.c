#include <fluca/private/physinsimpl.h>
#include <petscts.h>

/* Face stencil locations indexed by direction: LEFT for x, DOWN for y, BACK for z */
static const DMStagStencilLocation face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* --- BC adapter functions ------------------------------------------------- */

static PetscErrorCode PhysINS_BCAdapterFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PhysINS_BCAdapter *a = (PhysINS_BCAdapter *)ctx;

  PetscFunctionBegin;
  PetscCall(a->fn(dim, t, x, a->comp, value, a->fn_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysINS_BCAdapterFnDot(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PhysINS_BCAdapter *a = (PhysINS_BCAdapter *)ctx;

  PetscFunctionBegin;
  PetscCall(a->fn_dot(dim, t, x, a->comp, value, a->fn_dot_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set velocity Dirichlet BCs on a FlucaFD operator for a specific velocity component.
   Uses the BC adapter to bridge PhysINSBCFn (has comp) to FlucaFDBCValueFn (no comp). */
static PetscErrorCode SetVelocityDirichletBCs(Phys phys, FlucaFD fd, PetscInt comp)
{
  Phys_INS                *ins                          = (Phys_INS *)phys->data;
  FlucaFDBoundaryCondition fd_bcs[2 * PHYS_INS_MAX_DIM] = {{0}};
  PetscInt                 f;

  PetscFunctionBegin;
  for (f = 0; f < 2 * phys->dim; f++) {
    if (ins->bcs[f].type == PHYS_INS_BC_VELOCITY && ins->bcs[f].fn) {
      ins->bc_adapters[comp][f].fn         = ins->bcs[f].fn;
      ins->bc_adapters[comp][f].fn_dot     = ins->bcs[f].fn_dot;
      ins->bc_adapters[comp][f].fn_ctx     = ins->bcs[f].ctx;
      ins->bc_adapters[comp][f].fn_dot_ctx = ins->bcs[f].fn_dot_ctx;
      ins->bc_adapters[comp][f].comp       = comp;
      fd_bcs[f].type                       = FLUCAFD_BC_DIRICHLET;
      fd_bcs[f].fn                         = PhysINS_BCAdapterFn;
      fd_bcs[f].fn_ctx                     = &ins->bc_adapters[comp][f];
      fd_bcs[f].fn_dot                     = ins->bcs[f].fn_dot ? PhysINS_BCAdapterFnDot : NULL;
      fd_bcs[f].fn_dot_ctx                 = &ins->bc_adapters[comp][f];
    } else if (ins->bcs[f].type == PHYS_INS_BC_VELOCITY) {
      /* Constant zero velocity BC */
      fd_bcs[f].type  = FLUCAFD_BC_DIRICHLET;
      fd_bcs[f].value = 0.;
    }
  }
  PetscCall(FlucaFDSetBoundaryConditions(fd, comp, fd_bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set pressure Neumann (zero normal derivative) BCs on a FlucaFD operator */
static PetscErrorCode SetPressureNeumannBCs(Phys phys, FlucaFD fd, PetscInt comp)
{
  Phys_INS                *ins                          = (Phys_INS *)phys->data;
  FlucaFDBoundaryCondition fd_bcs[2 * PHYS_INS_MAX_DIM] = {{0}};
  PetscInt                 f;

  PetscFunctionBegin;
  for (f = 0; f < 2 * phys->dim; f++) {
    if (ins->bcs[f].type == PHYS_INS_BC_VELOCITY) {
      fd_bcs[f].type  = FLUCAFD_BC_NEUMANN;
      fd_bcs[f].value = 0.;
    }
  }
  PetscCall(FlucaFDSetBoundaryConditions(fd, comp, fd_bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Operator construction ------------------------------------------------ */

PetscErrorCode PhysINSBuildOperators_Internal(Phys phys)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d, e;
  PetscReal mu = ins->mu, rho = ins->rho;

  PetscFunctionBegin;
  /* --- fd_laplacian[d] = sum_e d/dx_e(-mu * d(u_d)/dx_e) --- */
  for (d = 0; d < dim; d++) {
    FlucaFD comp_ops[PHYS_INS_MAX_DIM];

    for (e = 0; e < dim; e++) {
      FlucaFD inner, scaled, outer;

      /* d(u_d)/dx_e */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, d, face_loc[e], 0, &inner));
      PetscCall(FlucaFDSetUp(inner));

      /* -mu * d(u_d)/dx_e */
      PetscCall(FlucaFDScaleCreateConstant(inner, -mu, &scaled));
      PetscCall(FlucaFDSetUp(scaled));

      /* d/dx_e(...) back to element */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], 0, DMSTAG_ELEMENT, d, &outer));
      PetscCall(FlucaFDSetUp(outer));

      /* d/dx_e(-mu * d(u_d)/dx_e) */
      PetscCall(FlucaFDCompositionCreate(scaled, outer, &comp_ops[e]));
      PetscCall(FlucaFDSetUp(comp_ops[e]));

      PetscCall(FlucaFDDestroy(&outer));
      PetscCall(FlucaFDDestroy(&scaled));
      PetscCall(FlucaFDDestroy(&inner));
    }

    PetscCall(FlucaFDSumCreate(dim, comp_ops, &ins->fd_laplacian[d]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_laplacian[d], d));
    PetscCall(FlucaFDSetUp(ins->fd_laplacian[d]));

    for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&comp_ops[e]));
  }

  /* --- fd_grad_p[d] = dp/dx_d --- */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, d, &ins->fd_grad_p[d]));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_grad_p[d], dim));
    PetscCall(FlucaFDSetUp(ins->fd_grad_p[d]));
  }

  /* --- fd_div = rho * sum_d d/dx_d(interp_d(u_d)) --- */
  {
    FlucaFD div_comp[PHYS_INS_MAX_DIM];

    for (d = 0; d < dim; d++) {
      FlucaFD interp, face_deriv, div_raw;

      /* interp_d(u_d) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 0, 2, DMSTAG_ELEMENT, d, face_loc[d], 0, &interp));
      PetscCall(FlucaFDSetUp(interp));

      /* d/dx_d(...) -> pressure DOF */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, face_loc[d], 0, DMSTAG_ELEMENT, dim, &face_deriv));
      PetscCall(FlucaFDSetUp(face_deriv));

      /* d/dx_d(interp_d(u_d)) */
      PetscCall(FlucaFDCompositionCreate(interp, face_deriv, &div_raw));
      PetscCall(FlucaFDSetUp(div_raw));

      /* rho * d/dx_d(interp_d(u_d)) */
      PetscCall(FlucaFDScaleCreateConstant(div_raw, rho, &div_comp[d]));
      PetscCall(FlucaFDSetUp(div_comp[d]));

      PetscCall(FlucaFDDestroy(&div_raw));
      PetscCall(FlucaFDDestroy(&face_deriv));
      PetscCall(FlucaFDDestroy(&interp));
    }

    PetscCall(FlucaFDSumCreate(dim, div_comp, &ins->fd_div));
    for (d = 0; d < dim; d++) PetscCall(SetVelocityDirichletBCs(phys, ins->fd_div, d));
    PetscCall(FlucaFDSetUp(ins->fd_div));

    for (d = 0; d < dim; d++) PetscCall(FlucaFDDestroy(&div_comp[d]));
  }

  /* --- fd_pstab = sigma_0 * S(p), S(p) = sum_d [d(dp/dx_d)/dx_d - d^2p/dx_d^2] --- */
  {
    FlucaFD pstab_dir[PHYS_INS_MAX_DIM];
    FlucaFD pstab_sum;

    for (d = 0; d < dim; d++) {
      FlucaFD cell_grad_p, cell_div_p, wide_d;
      FlucaFD compact_d, neg_compact;
      FlucaFD diff_ops[2];

      /* d(dp/dx_d)/dx_d: wide second derivative via two cell-centered first derivatives */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, d, &cell_grad_p));
      PetscCall(FlucaFDSetUp(cell_grad_p));
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, d, DMSTAG_ELEMENT, dim, &cell_div_p));
      PetscCall(FlucaFDSetUp(cell_div_p));
      PetscCall(FlucaFDCompositionCreate(cell_grad_p, cell_div_p, &wide_d));
      PetscCall(FlucaFDSetUp(wide_d));

      /* d^2p/dx_d^2: compact second derivative */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 2, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, dim, &compact_d));
      PetscCall(FlucaFDSetUp(compact_d));

      /* S_d(p) = d(dp/dx_d)/dx_d - d^2p/dx_d^2 */
      PetscCall(FlucaFDScaleCreateConstant(compact_d, -1., &neg_compact));
      PetscCall(FlucaFDSetUp(neg_compact));
      diff_ops[0] = wide_d;
      diff_ops[1] = neg_compact;
      PetscCall(FlucaFDSumCreate(2, diff_ops, &pstab_dir[d]));
      PetscCall(FlucaFDSetUp(pstab_dir[d]));

      PetscCall(FlucaFDDestroy(&neg_compact));
      PetscCall(FlucaFDDestroy(&wide_d));
      PetscCall(FlucaFDDestroy(&compact_d));
      PetscCall(FlucaFDDestroy(&cell_div_p));
      PetscCall(FlucaFDDestroy(&cell_grad_p));
    }

    /* sigma_0 * sum_d S_d(p); sigma_0 initially 0, updated to dt by TSPreStep */
    PetscCall(FlucaFDSumCreate(dim, pstab_dir, &pstab_sum));
    PetscCall(FlucaFDSetUp(pstab_sum));
    PetscCall(FlucaFDScaleCreateConstant(pstab_sum, 0., &ins->fd_pstab));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_pstab, dim));
    PetscCall(FlucaFDSetUp(ins->fd_pstab));

    PetscCall(FlucaFDDestroy(&pstab_sum));
    for (d = 0; d < dim; d++) PetscCall(FlucaFDDestroy(&pstab_dir[d]));
  }

  /* --- dm_face (single), mass_flux = F_d = rho * interp_d(u_d), fd_interp[d] --- */
  {
    DM cdm;

    switch (dim) {
    case 2:
      PetscCall(DMStagCreateCompatibleDMStag(sol_dm, 0, 1, 0, 0, &ins->dm_face));
      break;
    case 3:
      PetscCall(DMStagCreateCompatibleDMStag(sol_dm, 0, 0, 1, 0, &ins->dm_face));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
    }
    PetscCall(DMStagSetCoordinateDMType(ins->dm_face, DMPRODUCT));
    PetscCall(DMGetCoordinateDM(sol_dm, &cdm));
    PetscCall(DMSetCoordinateDM(ins->dm_face, cdm));
    PetscCall(DMCreateGlobalVector(ins->dm_face, &ins->mass_flux));
  }
  for (e = 0; e < dim; e++) {
    /* interp_e(u_e) */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, e, face_loc[e], 0, &ins->fd_interp[e]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_interp[e], e));
    PetscCall(FlucaFDSetUp(ins->fd_interp[e]));
  }

  /* --- fd_conv[d] = sum_e d/dx_e(F_e * TVD_e(u_d)) --- */
  for (d = 0; d < dim; d++) {
    FlucaFD conv_comp[PHYS_INS_MAX_DIM];

    for (e = 0; e < dim; e++) {
      FlucaFD face_deriv;

      /* TVD_e(u_d) */
      PetscCall(FlucaFDSecondOrderTVDCreate(sol_dm, (FlucaFDDirection)e, d, 0, &ins->fd_tvd[d][e]));
      PetscCall(FlucaFDAppendOptionsPrefix(ins->fd_tvd[d][e], "phys_ins_"));
      PetscCall(FlucaFDSetFromOptions(ins->fd_tvd[d][e]));
      PetscCall(FlucaFDSetUp(ins->fd_tvd[d][e]));

      /* F_e * TVD_e(u_d) */
      PetscCall(FlucaFDScaleCreateVector(ins->fd_tvd[d][e], ins->mass_flux, 0, &ins->fd_momentum_flux[d][e]));
      PetscCall(FlucaFDSetUp(ins->fd_momentum_flux[d][e]));

      /* d/dx_e(...) back to element */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], 0, DMSTAG_ELEMENT, d, &face_deriv));
      PetscCall(FlucaFDSetUp(face_deriv));

      /* d/dx_e(F_e * TVD_e(u_d)) */
      PetscCall(FlucaFDCompositionCreate(ins->fd_momentum_flux[d][e], face_deriv, &conv_comp[e]));
      PetscCall(FlucaFDSetUp(conv_comp[e]));

      PetscCall(FlucaFDDestroy(&face_deriv));
    }

    /* sum_e d/dx_e(F_e * TVD_e(u_d)) */
    PetscCall(FlucaFDSumCreate(dim, conv_comp, &ins->fd_conv[d]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_conv[d], d));
    PetscCall(FlucaFDSetUp(ins->fd_conv[d]));

    for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&conv_comp[e]));
  }

  /* Create temp vector for residual assembly */
  PetscCall(DMCreateGlobalVector(sol_dm, &ins->temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSDestroyOperators_Internal(Phys phys)
{
  Phys_INS *ins = (Phys_INS *)phys->data;
  PetscInt  d, e;

  PetscFunctionBegin;
  for (d = 0; d < PHYS_INS_MAX_DIM; d++) {
    PetscCall(FlucaFDDestroy(&ins->fd_laplacian[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_grad_p[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_interp[d]));
    for (e = 0; e < PHYS_INS_MAX_DIM; e++) {
      PetscCall(FlucaFDDestroy(&ins->fd_tvd[d][e]));
      PetscCall(FlucaFDDestroy(&ins->fd_momentum_flux[d][e]));
    }
  }
  PetscCall(FlucaFDDestroy(&ins->fd_div));
  PetscCall(FlucaFDDestroy(&ins->fd_pstab));
  PetscCall(VecDestroy(&ins->mass_flux));
  PetscCall(DMDestroy(&ins->dm_face));
  PetscCall(VecDestroy(&ins->temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Update convection operators with current velocity ------------------- */

static PetscErrorCode UpdateConvectionVelocity_Internal(Phys phys, PetscReal t, Vec U)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscReal rho    = ins->rho;
  PetscInt  dim    = phys->dim, d, e;

  PetscFunctionBegin;
  /* mass_flux = sum_d interp_d(u_d), then scale by rho to get F_d at each face */
  PetscCall(VecZeroEntries(ins->mass_flux));
  for (e = 0; e < dim; e++) PetscCall(FlucaFDApply(ins->fd_interp[e], t, sol_dm, ins->dm_face, U, ins->mass_flux));
  PetscCall(VecScale(ins->mass_flux, rho));

  /* Update TVD_e(u_d) and F_e * TVD_e(u_d) with current state */
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) {
      PetscCall(FlucaFDSecondOrderTVDSetMassFlux(ins->fd_tvd[d][e], ins->mass_flux, 0));
      PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ins->fd_tvd[d][e], U));
      PetscCall(FlucaFDScaleSetVector(ins->fd_momentum_flux[d][e], ins->mass_flux, face_loc[e], 0));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- IFunction: implicit part (viscous + pressure + ODE-transformed continuity) --- */

PetscErrorCode PhysComputeIFunction_INS(Phys phys, PetscReal t, Vec U, Vec U_t, Vec F)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;
  Vec       temp   = ins->temp;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(F));

  /* F_momentum_d = fd_laplacian[d](u) + fd_grad_p[d](p) */
  for (d = 0; d < dim; d++) {
    PetscCall(VecZeroEntries(temp));
    PetscCall(FlucaFDApply(ins->fd_laplacian[d], t, sol_dm, sol_dm, U, temp));
    PetscCall(VecAXPY(F, 1., temp));
    PetscCall(VecZeroEntries(temp));
    PetscCall(FlucaFDApply(ins->fd_grad_p[d], t, sol_dm, sol_dm, U, temp));
    PetscCall(VecAXPY(F, 1., temp));
  }

  /* F_momentum_d += rho * du_d/dt */
  {
    Vec F_vel, Ut_vel;

    PetscCall(VecGetSubVector(F, ins->is_vel, &F_vel));
    PetscCall(VecGetSubVector(U_t, ins->is_vel, &Ut_vel));
    PetscCall(VecAXPY(F_vel, ins->rho, Ut_vel));
    PetscCall(VecRestoreSubVector(U_t, ins->is_vel, &Ut_vel));
    PetscCall(VecRestoreSubVector(F, ins->is_vel, &F_vel));
  }

  /* F_continuity = alpha * [D(u) + sigma_0 * S(p)] + [D(du/dt) + sigma_0 * S(dp/dt)]
     with alpha = 1 (not 1/dt) so that alpha * sigma_0 = dt, avoiding the O(1) pressure error
     from the alpha * sigma_0 = 1 cancellation. */
  PetscCall(VecZeroEntries(temp));
  PetscCall(FlucaFDApply(ins->fd_div, t, sol_dm, sol_dm, U, temp));
  PetscCall(VecAXPY(F, ins->alpha, temp));
  PetscCall(VecZeroEntries(temp));
  PetscCall(FlucaFDApply(ins->fd_pstab, t, sol_dm, sol_dm, U, temp));
  PetscCall(VecAXPY(F, ins->alpha, temp));

  PetscCall(VecZeroEntries(temp));
  PetscCall(FlucaFDApplyDot(ins->fd_div, t, sol_dm, sol_dm, U_t, temp));
  PetscCall(VecAXPY(F, 1., temp));
  PetscCall(VecZeroEntries(temp));
  PetscCall(FlucaFDApplyDot(ins->fd_pstab, t, sol_dm, sol_dm, U_t, temp));
  PetscCall(VecAXPY(F, 1., temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- IJacobian ------------------------------------------------------------ */

PetscErrorCode PhysComputeIJacobian_INS(Phys phys, PetscReal t, Vec U, Vec U_t, PetscReal shift, Mat Amat, Mat Pmat)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(Pmat));

  /* Velocity rows: fd_laplacian[d] + fd_grad_p[d] (output_c = d < dim) */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDGetOperator(ins->fd_laplacian[d], sol_dm, sol_dm, Pmat));
    PetscCall(FlucaFDGetOperator(ins->fd_grad_p[d], sol_dm, sol_dm, Pmat));
  }

  /* Continuity rows: fd_div + fd_pstab (output_c = dim, no overlap with velocity rows).
     Scaled by (shift + alpha) from alpha*C(U) + dC/dt(U_t) linearization. */
  PetscCall(FlucaFDGetOperator(ins->fd_div, sol_dm, sol_dm, Pmat));
  PetscCall(FlucaFDGetOperator(ins->fd_pstab, sol_dm, sol_dm, Pmat));

  PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));

  /* Velocity diagonal: + shift * rho (from rho * du_d/dt) */
  {
    Vec diag_shift;

    PetscCall(MatCreateVecs(Pmat, NULL, &diag_shift));
    PetscCall(VecZeroEntries(diag_shift));
    {
      Vec vel_sub;

      PetscCall(VecGetSubVector(diag_shift, ins->is_vel, &vel_sub));
      PetscCall(VecSet(vel_sub, shift * ins->rho));
      PetscCall(VecRestoreSubVector(diag_shift, ins->is_vel, &vel_sub));
    }
    PetscCall(MatDiagonalSet(Pmat, diag_shift, ADD_VALUES));
    PetscCall(VecDestroy(&diag_shift));
  }

  /* Continuity rows *= (shift + alpha) */
  {
    Vec diag_scale;
    PetscCall(MatCreateVecs(Pmat, NULL, &diag_scale));
    PetscCall(VecSet(diag_scale, 1.));
    {
      Vec subvec;
      PetscCall(VecGetSubVector(diag_scale, ins->is_p, &subvec));
      PetscCall(VecSet(subvec, shift + ins->alpha));
      PetscCall(VecRestoreSubVector(diag_scale, ins->is_p, &subvec));
    }
    PetscCall(MatDiagonalScale(Pmat, diag_scale, NULL));
    PetscCall(VecDestroy(&diag_scale));
  }
  if (Amat != Pmat) {
    PetscCall(MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- RHSFunction: explicit part (convection + body force) ----------------- */

PetscErrorCode PhysComputeRHSFunction_INS(Phys phys, PetscReal t, Vec U, Vec G)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;
  Vec       temp   = ins->temp;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(G));

  /* Update convection operators with current velocity */
  PetscCall(UpdateConvectionVelocity_Internal(phys, t, U));

  /* G_momentum_d = -fd_conv[d](u) */
  for (d = 0; d < dim; d++) {
    PetscCall(VecZeroEntries(temp));
    PetscCall(FlucaFDApply(ins->fd_conv[d], t, sol_dm, sol_dm, U, temp));
    PetscCall(VecAXPY(G, -1., temp));
  }

  /* G_momentum_d += f_d(t) */
  if (phys->bodyforce) {
    const PetscScalar **arrc[3] = {NULL, NULL, NULL};
    PetscInt            xs, ys, zs, xm, ym, zm, slot_elem;
    PetscInt            i, j, k;

    PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
    PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
    PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));

    for (k = zs; k < zs + zm; k++) {
      for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
          PetscReal     coords[3] = {0};
          PetscScalar   force[3];
          DMStagStencil row_s;

          coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
          coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
          if (dim == 3) coords[2] = PetscRealPart(arrc[2][k][slot_elem]);
          PetscCall(phys->bodyforce(dim, t, coords, force, phys->bodyforce_ctx));

          row_s.j   = j;
          row_s.k   = k;
          row_s.loc = DMSTAG_ELEMENT;
          for (d = 0; d < dim; d++) {
            row_s.i = i;
            row_s.c = d;
            PetscCall(DMStagVecSetValuesStencil(sol_dm, G, 1, &row_s, &force[d], ADD_VALUES));
          }
        }
      }
    }

    PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
    PetscCall(VecAssemblyBegin(G));
    PetscCall(VecAssemblyEnd(G));
  }

  /* G_continuity = 0 */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- TS callbacks --------------------------------------------------------- */

static PetscErrorCode UpdatePressureStabilizationDt_Internal(TS ts, Phys_INS *ins)
{
  PetscReal dt;

  PetscFunctionBegin;
  PetscCall(TSGetTimeStep(ts, &dt));
  if (dt != ins->dt_current) {
    PetscCall(FlucaFDScaleSetConstant(ins->fd_pstab, dt));
    ins->alpha      = 1. / dt;
    ins->dt_current = dt;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IFunction_INS(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, void *ctx)
{
  Phys      phys = (Phys)ctx;
  Phys_INS *ins  = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  PetscCall(UpdatePressureStabilizationDt_Internal(ts, ins));
  PetscCall(PhysComputeIFunction_INS(phys, t, U, U_t, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IJacobian_INS(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal shift, Mat Amat, Mat Pmat, void *ctx)
{
  Phys      phys = (Phys)ctx;
  Phys_INS *ins  = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  PetscCall(UpdatePressureStabilizationDt_Internal(ts, ins));
  PetscCall(PhysComputeIJacobian_INS(phys, t, U, U_t, shift, Amat, Pmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction_INS(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  Phys phys = (Phys)ctx;

  PetscFunctionBegin;
  PetscCall(PhysComputeRHSFunction_INS(phys, t, U, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Solver data creation ------------------------------------------------- */

PetscErrorCode PhysINSCreateSolverData_Internal(Phys phys)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;
  MPI_Comm  comm;

  PetscFunctionBegin;
  if (ins->J) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)phys, &comm));

  /* Create Jacobian matrices */
  PetscCall(DMCreateMatrix(sol_dm, &ins->J));
  PetscCall(MatSetOption(ins->J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(DMCreateMatrix(sol_dm, &ins->J_rhs));
  PetscCall(MatSetOption(ins->J_rhs, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  /* Create IS for field decomposition */
  {
    DMStagStencil vel_stencils[PHYS_INS_MAX_DIM], p_stencil[1];

    for (d = 0; d < dim; d++) {
      vel_stencils[d].i   = 0;
      vel_stencils[d].j   = 0;
      vel_stencils[d].k   = 0;
      vel_stencils[d].loc = DMSTAG_ELEMENT;
      vel_stencils[d].c   = d;
    }
    p_stencil[0].i   = 0;
    p_stencil[0].j   = 0;
    p_stencil[0].k   = 0;
    p_stencil[0].loc = DMSTAG_ELEMENT;
    p_stencil[0].c   = dim;

    PetscCall(DMStagCreateISFromStencils(sol_dm, dim, vel_stencils, &ins->is_vel));
    PetscCall(DMStagCreateISFromStencils(sol_dm, 1, p_stencil, &ins->is_p));

    /* Per-component velocity IS */
    for (d = 0; d < dim; d++) {
      DMStagStencil comp_stencil;

      comp_stencil.i   = 0;
      comp_stencil.j   = 0;
      comp_stencil.k   = 0;
      comp_stencil.loc = DMSTAG_ELEMENT;
      comp_stencil.c   = d;
      PetscCall(DMStagCreateISFromStencils(sol_dm, 1, &comp_stencil, &ins->is_comp[d]));
    }
  }

  /* Null space for pressure (when all-velocity Dirichlet) */
  ins->has_pressure_outlet = PETSC_FALSE;
  if (!ins->has_pressure_outlet) {
    Vec      nullvec, subvec;
    PetscInt np;

    PetscCall(DMCreateGlobalVector(sol_dm, &nullvec));
    PetscCall(VecZeroEntries(nullvec));
    PetscCall(VecGetSubVector(nullvec, ins->is_p, &subvec));
    PetscCall(VecGetSize(subvec, &np));
    PetscCall(VecSet(subvec, 1. / PetscSqrtReal((PetscReal)np)));
    PetscCall(VecRestoreSubVector(nullvec, ins->is_p, &subvec));
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, 1, &nullvec, &ins->nullspace));
    PetscCall(VecDestroy(&nullvec));
    PetscCall(MatSetNullSpace(ins->J, ins->nullspace));

    /* Compose pressure-only null space onto is_p so that PCFieldSplit
       auto-propagates it to the Schur complement diagonal sub-block.
       "nullspace": KSP projects it out from the Schur complement RHS.
       "nearnullspace": GAMG uses it for multigrid coarsening. */
    {
      MatNullSpace p_nullspace;

      PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &p_nullspace));
      PetscCall(PetscObjectCompose((PetscObject)ins->is_p, "nullspace", (PetscObject)p_nullspace));
      PetscCall(PetscObjectCompose((PetscObject)ins->is_p, "nearnullspace", (PetscObject)p_nullspace));
      PetscCall(MatNullSpaceDestroy(&p_nullspace));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- PhysSetUpTS_INS ------------------------------------------------------ */

PetscErrorCode PhysSetUpTS_INS(Phys phys, TS ts)
{
  Phys_INS *ins = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  /* Wire DM */
  PetscCall(TSSetDM(ts, phys->sol_dm));

  /* Wire IMEX callbacks */
  PetscCall(TSSetIFunction(ts, NULL, IFunction_INS, phys));
  PetscCall(TSSetIJacobian(ts, ins->J, ins->J, IJacobian_INS, phys));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_INS, phys));

  /* Default to TSARKIMEX */
  PetscCall(TSSetType(ts, TSARKIMEX));

  /* Default solver: KSPONLY + nested fieldsplit.
     Outer: velocity (u,v,[w]) vs pressure (p).
     Inner: additive fieldsplit on velocity — u, v, [w] independently
     (the implicit operator has no cross terms between velocity components). */
  {
    SNES        snes;
    KSP         ksp, *outer_sub;
    PC          pc, vel_pc;
    PetscInt    dim          = phys->dim, n_splits, n_vel, n_cells;
    const char *comp_names[] = {"u", "v", "w"};

    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESSetType(snes, SNESKSPONLY));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCFIELDSPLIT));
    PetscCall(PCFieldSplitSetIS(pc, "velocity", ins->is_vel));
    PetscCall(PCFieldSplitSetIS(pc, "pressure", ins->is_p));

    /* Inner split: u, v, [w] within the velocity sub-block.
       DMStagCreateISFromStencils orders indices by (location, component) per cell,
       so the velocity sub-block interleaves components: [u0,v0, u1,v1, ...].
       Each component d occupies stride positions: start=d, step=dim, count=n_cells. */
    PetscCall(ISGetLocalSize(ins->is_vel, &n_vel));
    n_cells = n_vel / dim;

    PetscCall(PCFieldSplitGetSubKSP(pc, &n_splits, &outer_sub));

    /* Velocity sub-PC: additive fieldsplit over u, v, [w] */
    PetscCall(KSPGetPC(outer_sub[0], &vel_pc));
    PetscCall(PCSetType(vel_pc, PCFIELDSPLIT));
    for (PetscInt d = 0; d < dim; d++) {
      IS is_d;

      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)phys), n_cells, d, dim, &is_d));
      PetscCall(PCFieldSplitSetIS(vel_pc, comp_names[d], is_d));
      PetscCall(ISDestroy(&is_d));
    }

    /* Default leaf-level sub-PCs to Jacobi (ILU fails on periodic grids) */
    {
      PetscInt n_vel_splits;
      KSP     *vel_sub;
      PC       sub_pc;

      PetscCall(PCFieldSplitGetSubKSP(vel_pc, &n_vel_splits, &vel_sub));
      for (PetscInt d = 0; d < n_vel_splits; d++) {
        PetscCall(KSPGetPC(vel_sub[d], &sub_pc));
        PetscCall(PCSetType(sub_pc, PCJACOBI));
      }
      PetscCall(PetscFree(vel_sub));

      PetscCall(KSPGetPC(outer_sub[1], &sub_pc));
      PetscCall(PCSetType(sub_pc, PCJACOBI));
    }
    PetscCall(PetscFree(outer_sub));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- PhysSetUpSeg_INS: wire FlucaFD operators into Seg ------------------- */

/* SRK explicit RHS callback: -(conv(u) + G(p) - source) / rho */
static PetscErrorCode SRKExplicitRHS(PetscReal t, Vec Y, Vec F, void *ctx)
{
  Phys      phys = (Phys)ctx;
  Phys_INS *ins  = (Phys_INS *)phys->data;
  DM        dm   = phys->sol_dm;
  PetscInt  dim  = phys->dim, d;

  PetscFunctionBegin;
  PetscCheck(ins->rho > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Density must be positive");
  PetscCall(VecZeroEntries(F));

  /* Update convection operators with stage velocity */
  PetscCall(UpdateConvectionVelocity_Internal(phys, t, Y));

  /* F_vel -= conv(u) / rho */
  for (d = 0; d < dim; d++) {
    PetscCall(VecZeroEntries(ins->temp));
    PetscCall(FlucaFDApply(ins->fd_conv[d], t, dm, dm, Y, ins->temp));
    PetscCall(VecAXPY(F, -1.0 / ins->rho, ins->temp));
  }

  /* Note: pressure gradient G(p) is NOT included here — it is handled inside the
     SRK stage as a constraint force in the Helmholtz and pressure solves. */

  /* F_vel += source / rho */
  if (phys->bodyforce) {
    const PetscScalar **arrc[3] = {NULL, NULL, NULL};
    PetscInt            xs, ys, zs, xm, ym, zm, slot_elem;
    PetscInt            i, j, k;

    PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
    PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
    PetscCall(DMStagGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));

    for (k = zs; k < zs + zm; k++) {
      for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
          PetscReal     coords[3] = {0};
          PetscScalar   force[3];
          DMStagStencil row_s;

          coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
          coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
          if (dim == 3) coords[2] = PetscRealPart(arrc[2][k][slot_elem]);
          PetscCall(phys->bodyforce(dim, t, coords, force, phys->bodyforce_ctx));

          row_s.j   = j;
          row_s.k   = k;
          row_s.loc = DMSTAG_ELEMENT;
          for (d = 0; d < dim; d++) {
            PetscScalar val = force[d] / ins->rho;

            row_s.i = i;
            row_s.c = d;
            PetscCall(DMStagVecSetValuesStencil(dm, F, 1, &row_s, &val, ADD_VALUES));
          }
        }
      }
    }

    PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
    PetscCall(VecAssemblyBegin(F));
    PetscCall(VecAssemblyEnd(F));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetUpSeg_INS(Phys phys, Seg seg)
{
  Phys_INS *ins = (Phys_INS *)phys->data;
  PetscInt  dim = phys->dim, d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(seg, SEG_CLASSID, 2);
  /* Wire DM and solution data */
  PetscCall(SegSetDM(seg, phys->sol_dm));

  /* Wire density */
  PetscCall(SegSRKSetDensity(seg, ins->rho));

  /* Wire FlucaFD operators */
  for (d = 0; d < dim; d++) {
    PetscCall(SegSRKSetLaplacian(seg, d, ins->fd_laplacian[d]));
    PetscCall(SegSRKSetGradient(seg, d, ins->fd_grad_p[d]));
  }
  PetscCall(SegSRKSetDivergence(seg, ins->fd_div));
  PetscCall(SegSRKSetStabilization(seg, ins->fd_pstab));

  /* Wire field IS */
  PetscCall(SegSRKSetFieldIS(seg, dim, ins->is_vel, ins->is_p, ins->is_comp));

  /* Wire explicit RHS callback */
  PetscCall(SegSetRHSFunction(seg, SRKExplicitRHS, phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}
