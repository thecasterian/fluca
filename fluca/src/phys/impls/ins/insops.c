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
      fd_bcs[f].value = 0.0;
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
      fd_bcs[f].value = 0.0;
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
  /* --- Viscous Laplacian per velocity direction: -mu * nabla^2 u_d --- */
  for (d = 0; d < dim; d++) {
    FlucaFD comp_ops[PHYS_INS_MAX_DIM];

    for (e = 0; e < dim; e++) {
      FlucaFD inner, scaled, outer;

      /* inner: d/de from (ELEMENT, d) to (face_loc[e], 0) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, d, face_loc[e], 0, &inner));
      PetscCall(FlucaFDSetUp(inner));

      /* scaled: -mu * inner */
      PetscCall(FlucaFDScaleCreateConstant(inner, -mu, &scaled));
      PetscCall(FlucaFDSetUp(scaled));

      /* outer: d/de from (face_loc[e], 0) to (ELEMENT, d) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], 0, DMSTAG_ELEMENT, d, &outer));
      PetscCall(FlucaFDSetUp(outer));

      /* composition: outer(scaled(x)) = d/de(-mu * du_d/de) */
      PetscCall(FlucaFDCompositionCreate(scaled, outer, &comp_ops[e]));
      PetscCall(FlucaFDSetUp(comp_ops[e]));

      PetscCall(FlucaFDDestroy(&outer));
      PetscCall(FlucaFDDestroy(&scaled));
      PetscCall(FlucaFDDestroy(&inner));
    }

    /* Sum over directions: -mu * nabla^2 u_d */
    PetscCall(FlucaFDSumCreate(dim, comp_ops, &ins->fd_laplacian[d]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_laplacian[d], d));
    PetscCall(FlucaFDSetUp(ins->fd_laplacian[d]));

    for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&comp_ops[e]));
  }

  /* --- Pressure gradient per velocity direction: dp/dx_d --- */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, d, &ins->fd_grad_p[d]));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_grad_p[d], dim));
    PetscCall(FlucaFDSetUp(ins->fd_grad_p[d]));
  }

  /* --- Divergence: rho * sum_d d(interp(u_d))/dx_d --- */
  {
    FlucaFD div_comp[PHYS_INS_MAX_DIM];

    for (d = 0; d < dim; d++) {
      FlucaFD interp, face_deriv, div_raw;

      /* interp: interpolate u_d from element to face (0th derivative = interpolation) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 0, 2, DMSTAG_ELEMENT, d, face_loc[d], 0, &interp));
      PetscCall(FlucaFDSetUp(interp));

      /* face_deriv: d/dx_d from face to element pressure DOF */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, face_loc[d], 0, DMSTAG_ELEMENT, dim, &face_deriv));
      PetscCall(FlucaFDSetUp(face_deriv));

      /* composition: face_deriv(interp(x)) = d(interp(u_d))/dx_d */
      PetscCall(FlucaFDCompositionCreate(interp, face_deriv, &div_raw));
      PetscCall(FlucaFDSetUp(div_raw));

      /* scale by rho */
      PetscCall(FlucaFDScaleCreateConstant(div_raw, rho, &div_comp[d]));
      PetscCall(FlucaFDSetUp(div_comp[d]));

      PetscCall(FlucaFDDestroy(&div_raw));
      PetscCall(FlucaFDDestroy(&face_deriv));
      PetscCall(FlucaFDDestroy(&interp));
    }

    /* Sum over directions with velocity BCs per input component */
    PetscCall(FlucaFDSumCreate(dim, div_comp, &ins->fd_div));
    for (d = 0; d < dim; d++) PetscCall(SetVelocityDirichletBCs(phys, ins->fd_div, d));
    PetscCall(FlucaFDSetUp(ins->fd_div));

    for (d = 0; d < dim; d++) PetscCall(FlucaFDDestroy(&div_comp[d]));
  }

  /* --- Pressure stabilization: sigma_0 * sum_d (DTG_d - DG^st_d)(p) --- */
  {
    FlucaFD pstab_dir[PHYS_INS_MAX_DIM];
    FlucaFD pstab_sum;

    for (d = 0; d < dim; d++) {
      FlucaFD face_grad_p, face_div_p, compact_d;
      FlucaFD cell_grad_p, interp_p, td_p, wide_d;
      FlucaFD neg_compact;
      FlucaFD diff_ops[2];

      /* Compact Laplacian DG^st_d: face_div_p(face_grad_p(p)) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, face_loc[d], 0, &face_grad_p));
      PetscCall(FlucaFDSetUp(face_grad_p));
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, face_loc[d], 0, DMSTAG_ELEMENT, dim, &face_div_p));
      PetscCall(FlucaFDSetUp(face_div_p));
      PetscCall(FlucaFDCompositionCreate(face_grad_p, face_div_p, &compact_d));
      PetscCall(FlucaFDSetUp(compact_d));

      /* Wide Laplacian DTG_d: face_div_p(interp_p(cell_grad_p(p))) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, d, &cell_grad_p));
      PetscCall(FlucaFDSetUp(cell_grad_p));
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 0, 2, DMSTAG_ELEMENT, d, face_loc[d], 0, &interp_p));
      PetscCall(FlucaFDSetUp(interp_p));
      PetscCall(FlucaFDCompositionCreate(interp_p, face_div_p, &td_p));
      PetscCall(FlucaFDSetUp(td_p));
      PetscCall(FlucaFDCompositionCreate(cell_grad_p, td_p, &wide_d));
      PetscCall(FlucaFDSetUp(wide_d));

      /* S_d = wide_d - compact_d */
      PetscCall(FlucaFDScaleCreateConstant(compact_d, -1.0, &neg_compact));
      PetscCall(FlucaFDSetUp(neg_compact));
      diff_ops[0] = wide_d;
      diff_ops[1] = neg_compact;
      PetscCall(FlucaFDSumCreate(2, diff_ops, &pstab_dir[d]));
      PetscCall(FlucaFDSetUp(pstab_dir[d]));

      PetscCall(FlucaFDDestroy(&neg_compact));
      PetscCall(FlucaFDDestroy(&wide_d));
      PetscCall(FlucaFDDestroy(&td_p));
      PetscCall(FlucaFDDestroy(&interp_p));
      PetscCall(FlucaFDDestroy(&cell_grad_p));
      PetscCall(FlucaFDDestroy(&compact_d));
      PetscCall(FlucaFDDestroy(&face_div_p));
      PetscCall(FlucaFDDestroy(&face_grad_p));
    }

    /* Sum over directions and scale by sigma_0 (initially 0, updated by TSPreStep) */
    PetscCall(FlucaFDSumCreate(dim, pstab_dir, &pstab_sum));
    PetscCall(FlucaFDSetUp(pstab_sum));
    PetscCall(FlucaFDScaleCreateConstant(pstab_sum, 0.0, &ins->fd_pstab));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_pstab, dim));
    PetscCall(FlucaFDSetUp(ins->fd_pstab));

    PetscCall(FlucaFDDestroy(&pstab_sum));
    for (d = 0; d < dim; d++) PetscCall(FlucaFDDestroy(&pstab_dir[d]));
  }

  /* --- Face DMs and interpolation operators for convection --- */
  for (e = 0; e < dim; e++) {
    DM cdm;

    /* Face DM: 1 DOF at edge (2D) or face (3D) location */
    switch (dim) {
    case 2:
      PetscCall(DMStagCreateCompatibleDMStag(sol_dm, 0, 1, 0, 0, &ins->dm_face[e]));
      break;
    case 3:
      PetscCall(DMStagCreateCompatibleDMStag(sol_dm, 0, 0, 1, 0, &ins->dm_face[e]));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
    }
    PetscCall(DMStagSetCoordinateDMType(ins->dm_face[e], DMPRODUCT));
    PetscCall(DMGetCoordinateDM(sol_dm, &cdm));
    PetscCall(DMSetCoordinateDM(ins->dm_face[e], cdm));
    PetscCall(DMCreateGlobalVector(ins->dm_face[e], &ins->mass_flux_face[e]));

    /* Interpolation: u_e from ELEMENT,e to face_loc[e],0 */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, e, face_loc[e], 0, &ins->fd_interp[e]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_interp[e], e));
    PetscCall(FlucaFDSetUp(ins->fd_interp[e]));
  }

  /* --- Convection operators: C_d = sum_e d/dx_e(F_e * u_d_TVD) --- */
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) {
      FlucaFD face_deriv;

      /* TVD interpolation: u_d (ELEMENT,d) -> u_d_TVD (face_loc[e],0) */
      PetscCall(FlucaFDSecondOrderTVDCreate(sol_dm, (FlucaFDDirection)e, d, 0, &ins->fd_tvd[d][e]));
      PetscCall(FlucaFDAppendOptionsPrefix(ins->fd_tvd[d][e], "phys_ins_"));
      PetscCall(FlucaFDSetFromOptions(ins->fd_tvd[d][e]));
      PetscCall(FlucaFDSetUp(ins->fd_tvd[d][e]));

      /* Scale by face mass flux: u_d_TVD * F_e */
      PetscCall(FlucaFDScaleCreateVector(ins->fd_tvd[d][e], ins->mass_flux_face[e], 0, &ins->fd_scale_vel[d][e]));
      PetscCall(FlucaFDSetUp(ins->fd_scale_vel[d][e]));

      /* Face derivative: d/dx_e (face_loc[e],0 -> ELEMENT,d) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], 0, DMSTAG_ELEMENT, d, &face_deriv));
      PetscCall(FlucaFDSetUp(face_deriv));

      /* Compose: d/dx_e(F_e * u_d_TVD) */
      PetscCall(FlucaFDCompositionCreate(ins->fd_scale_vel[d][e], face_deriv, &ins->fd_conv_comp[d][e]));
      PetscCall(FlucaFDSetUp(ins->fd_conv_comp[d][e]));

      PetscCall(FlucaFDDestroy(&face_deriv));
    }

    /* Sum over e: C_d = sum_e d/dx_e(F_e * u_d_TVD) */
    PetscCall(FlucaFDSumCreate(dim, ins->fd_conv_comp[d], &ins->fd_conv[d]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_conv[d], d));
    PetscCall(FlucaFDSetUp(ins->fd_conv[d]));
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
    PetscCall(VecDestroy(&ins->mass_flux_face[d]));
    PetscCall(DMDestroy(&ins->dm_face[d]));
    for (e = 0; e < PHYS_INS_MAX_DIM; e++) {
      PetscCall(FlucaFDDestroy(&ins->fd_tvd[d][e]));
      PetscCall(FlucaFDDestroy(&ins->fd_scale_vel[d][e]));
      PetscCall(FlucaFDDestroy(&ins->fd_conv_comp[d][e]));
    }
  }
  PetscCall(FlucaFDDestroy(&ins->fd_div));
  PetscCall(FlucaFDDestroy(&ins->fd_pstab));
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
  /* Interpolate each velocity component to faces and scale by rho to get mass flux */
  for (e = 0; e < dim; e++) {
    PetscCall(FlucaFDApply(ins->fd_interp[e], t, sol_dm, ins->dm_face[e], U, ins->mass_flux_face[e]));
    PetscCall(VecScale(ins->mass_flux_face[e], rho));
  }

  /* Update TVD and scale operators with current mass flux and solution */
  for (d = 0; d < dim; d++) {
    for (e = 0; e < dim; e++) {
      PetscCall(FlucaFDSecondOrderTVDSetMassFlux(ins->fd_tvd[d][e], ins->mass_flux_face[e], 0));
      PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(ins->fd_tvd[d][e], U));
      PetscCall(FlucaFDScaleSetVector(ins->fd_scale_vel[d][e], ins->mass_flux_face[e], face_loc[e], 0));
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

  /* Momentum: -mu * L * u_d + dp/dx_d for each velocity direction */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDApply(ins->fd_laplacian[d], t, sol_dm, sol_dm, U, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
    PetscCall(FlucaFDApply(ins->fd_grad_p[d], t, sol_dm, sol_dm, U, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
  }

  /* Add mass term: rho * U_dot for velocity DOFs only, using IS to extract velocity sub-vector */
  {
    Vec F_vel, Ut_vel;

    PetscCall(VecGetSubVector(F, ins->is_vel, &F_vel));
    PetscCall(VecGetSubVector(U_t, ins->is_vel, &Ut_vel));
    PetscCall(VecAXPY(F_vel, ins->rho, Ut_vel));
    PetscCall(VecRestoreSubVector(U_t, ins->is_vel, &Ut_vel));
    PetscCall(VecRestoreSubVector(F, ins->is_vel, &F_vel));
  }

  /* Continuity: alpha * D(u) + D_dot(u_dot) + alpha * sigma_0 * S(p) + sigma_0 * S_dot(p_dot)
     Each FlucaFDApply uses INSERT_VALUES, so operators sharing the same output DOF (ELEMENT, dim)
     must be applied and accumulated separately to avoid overwriting. */

  /* Constraint feedback: alpha * D(u) */
  PetscCall(FlucaFDApply(ins->fd_div, t, sol_dm, sol_dm, U, temp));
  PetscCall(VecAXPY(F, ins->alpha, temp));
  /* Constraint feedback: alpha * sigma_0 * S(p) */
  PetscCall(FlucaFDApply(ins->fd_pstab, t, sol_dm, sol_dm, U, temp));
  PetscCall(VecAXPY(F, ins->alpha, temp));

  /* Time derivative part: D_dot(u_dot) */
  PetscCall(FlucaFDApplyDot(ins->fd_div, t, sol_dm, sol_dm, U_t, temp));
  PetscCall(VecAXPY(F, 1.0, temp));
  /* Time derivative part: sigma_0 * S_dot(p_dot) */
  PetscCall(FlucaFDApplyDot(ins->fd_pstab, t, sol_dm, sol_dm, U_t, temp));
  PetscCall(VecAXPY(F, 1.0, temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- IJacobian ------------------------------------------------------------ */

PetscErrorCode PhysComputeIJacobian_INS(Phys phys, PetscReal t, Vec U, Vec U_t, PetscReal shift, Mat Amat, Mat Pmat)
{
  Phys_INS     *ins    = (Phys_INS *)phys->data;
  DM            sol_dm = phys->sol_dm;
  PetscInt      dim    = phys->dim, d;
  PetscInt      xs, ys, zs, xm, ym, zm;
  PetscInt      i, j, k;
  DMStagStencil row;
  PetscScalar   val;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(Pmat));

  /* Velocity rows: -mu * L + G (no overlap with continuity rows since output_c < dim) */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDGetOperator(ins->fd_laplacian[d], sol_dm, sol_dm, Pmat));
    PetscCall(FlucaFDGetOperator(ins->fd_grad_p[d], sol_dm, sol_dm, Pmat));
  }

  /* Continuity rows: D and S (output_c = dim, no overlap with velocity rows).
     FlucaFDGetOperator ADDs the natural operator coefficients (rho for D, sigma_0 for S).
     We need (shift + alpha) * [D + S], so we add them first, then scale the continuity
     rows by (shift + alpha) after assembly. */
  PetscCall(FlucaFDGetOperator(ins->fd_div, sol_dm, sol_dm, Pmat));
  PetscCall(FlucaFDGetOperator(ins->fd_pstab, sol_dm, sol_dm, Pmat));

  /* Add shift * rho to velocity diagonal entries */
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));
  row.loc = DMSTAG_ELEMENT;
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        row.i = i;
        row.j = j;
        row.k = k;
        for (d = 0; d < dim; d++) {
          row.c = d;
          val   = shift * ins->rho;
          PetscCall(DMStagMatSetValuesStencil(sol_dm, Pmat, 1, &row, 1, &row, &val, ADD_VALUES));
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));

  /* Scale continuity rows by (shift + alpha) using MatDiagonalScale with the IS */
  {
    Vec diag_scale;
    PetscCall(MatCreateVecs(Pmat, NULL, &diag_scale));
    PetscCall(VecSet(diag_scale, 1.0));
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

  /* Momentum: -C_d for each velocity direction */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDApply(ins->fd_conv[d], t, sol_dm, sol_dm, U, temp));
    PetscCall(VecAXPY(G, -1.0, temp));
  }

  /* Add body force f(t) if set */
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

  /* G_continuity = 0 (constraint feedback is in IFunction) */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- TS callbacks --------------------------------------------------------- */

static PetscErrorCode UpdatePressureStabilizationDt_Internal(TS ts, Phys_INS *ins)
{
  PetscReal dt;

  PetscFunctionBegin;
  PetscCall(TSGetTimeStep(ts, &dt));
  if (dt != ins->dt_current) {
    PetscCall(FlucaFDScaleSetConstant(ins->fd_pstab, dt / ins->rho));
    ins->alpha      = 1.0 / dt;
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
    PetscCall(VecSet(subvec, 1.0 / PetscSqrtReal((PetscReal)np)));
    PetscCall(VecRestoreSubVector(nullvec, ins->is_p, &subvec));
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, 1, &nullvec, &ins->nullspace));
    PetscCall(VecDestroy(&nullvec));
    PetscCall(MatSetNullSpace(ins->J, ins->nullspace));
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

  /* Default PC: ILU (user can override via options) */
  {
    SNES snes;
    KSP  ksp;
    PC   pc;

    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCILU));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
