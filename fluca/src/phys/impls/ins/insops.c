#include <fluca/private/physinsimpl.h>

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
  PetscReal nu     = ins->mu / ins->rho;

  PetscFunctionBegin;
  /* --- fd_diff[d] = F_diff_d = sum_e d/dx_e(nu * d(u_d)/dx_e) --- */
  for (d = 0; d < dim; d++) {
    FlucaFD comp_ops[PHYS_INS_MAX_DIM];

    for (e = 0; e < dim; e++) {
      FlucaFD inner, scaled, outer;

      /* d(u_d)/dx_e */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, d, face_loc[e], 0, &inner));
      PetscCall(FlucaFDSetUp(inner));

      /* nu * d(u_d)/dx_e */
      PetscCall(FlucaFDScaleCreateConstant(inner, nu, &scaled));
      PetscCall(FlucaFDSetUp(scaled));

      /* d/dx_e(...) back to element */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], 0, DMSTAG_ELEMENT, d, &outer));
      PetscCall(FlucaFDSetUp(outer));

      /* d/dx_e(nu * d(u_d)/dx_e) */
      PetscCall(FlucaFDCompositionCreate(scaled, outer, &comp_ops[e]));
      PetscCall(FlucaFDSetUp(comp_ops[e]));

      PetscCall(FlucaFDDestroy(&outer));
      PetscCall(FlucaFDDestroy(&scaled));
      PetscCall(FlucaFDDestroy(&inner));
    }

    PetscCall(FlucaFDSumCreate(dim, comp_ops, &ins->fd_diff[d]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_diff[d], d));
    PetscCall(FlucaFDSetUp(ins->fd_diff[d]));

    for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&comp_ops[e]));
  }

  /* --- fd_grad_p[d] = (1/rho) * dp/dx_d --- */
  for (d = 0; d < dim; d++) {
    FlucaFD bare_grad;

    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, d, &bare_grad));
    PetscCall(FlucaFDSetUp(bare_grad));
    PetscCall(FlucaFDScaleCreateConstant(bare_grad, 1. / ins->rho, &ins->fd_grad_p[d]));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_grad_p[d], dim));
    PetscCall(FlucaFDSetUp(ins->fd_grad_p[d]));
    PetscCall(FlucaFDDestroy(&bare_grad));
  }

  /* --- fd_div = rho * sum_d d(u_d)/dx_d (cell-to-cell, wide divergence) --- */
  {
    FlucaFD div_comp[PHYS_INS_MAX_DIM];
    FlucaFD div_sum;

    for (d = 0; d < dim; d++) {
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, d, DMSTAG_ELEMENT, dim, &div_comp[d]));
      PetscCall(FlucaFDSetUp(div_comp[d]));
    }

    PetscCall(FlucaFDSumCreate(dim, div_comp, &div_sum));
    PetscCall(FlucaFDSetUp(div_sum));
    PetscCall(FlucaFDScaleCreateConstant(div_sum, ins->rho, &ins->fd_div));
    for (d = 0; d < dim; d++) PetscCall(SetVelocityDirichletBCs(phys, ins->fd_div, d));
    PetscCall(FlucaFDSetUp(ins->fd_div));

    PetscCall(FlucaFDDestroy(&div_sum));
    for (d = 0; d < dim; d++) PetscCall(FlucaFDDestroy(&div_comp[d]));
  }

  /* --- fd_pres_lap = L = sum_d d^2p/dx_d^2 (compact pressure Laplacian) --- */
  {
    FlucaFD compact_dir[PHYS_INS_MAX_DIM];

    for (d = 0; d < dim; d++) {
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 2, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, dim, &compact_dir[d]));
      PetscCall(FlucaFDSetUp(compact_dir[d]));
    }
    PetscCall(FlucaFDSumCreate(dim, compact_dir, &ins->fd_pres_lap));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_pres_lap, dim));
    PetscCall(FlucaFDSetUp(ins->fd_pres_lap));
    for (d = 0; d < dim; d++) PetscCall(FlucaFDDestroy(&compact_dir[d]));
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
    PetscCall(FlucaFDDestroy(&ins->fd_diff[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_grad_p[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_interp[d]));
    for (e = 0; e < PHYS_INS_MAX_DIM; e++) {
      PetscCall(FlucaFDDestroy(&ins->fd_tvd[d][e]));
      PetscCall(FlucaFDDestroy(&ins->fd_momentum_flux[d][e]));
    }
  }
  PetscCall(FlucaFDDestroy(&ins->fd_div));
  PetscCall(FlucaFDDestroy(&ins->fd_pres_lap));
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

PetscErrorCode PhysINSCreateSolverData_Internal(Phys phys)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;

  PetscFunctionBegin;
  if (ins->is_vel) PetscFunctionReturn(PETSC_SUCCESS);

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
    PetscCall(SegSRKSetDiffusion(seg, d, ins->fd_diff[d]));
    PetscCall(SegSRKSetGradient(seg, d, ins->fd_grad_p[d]));
  }
  PetscCall(SegSRKSetDivergence(seg, ins->fd_div));
  PetscCall(SegSRKSetPressureLaplacian(seg, ins->fd_pres_lap));

  /* Wire field IS */
  PetscCall(SegSRKSetFieldIS(seg, dim, ins->is_vel, ins->is_p, ins->is_comp));

  /* Wire explicit RHS callback */
  PetscCall(SegSetRHSFunction(seg, SRKExplicitRHS, phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}
