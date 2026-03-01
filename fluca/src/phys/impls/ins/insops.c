#include <fluca/private/physinsimpl.h>
#include <petscts.h>

/* Face stencil locations indexed by direction: LEFT for x, DOWN for y, BACK for z */
static const DMStagStencilLocation face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* --- BC adapter ----------------------------------------------------------- */

static PetscErrorCode PhysINSBCAdapterFn(PetscInt dim, const PetscReal x[], PetscScalar *val, void *ctx)
{
  PhysINSBCAdapter *adapter = (PhysINSBCAdapter *)ctx;

  PetscFunctionBegin;
  PetscCall(adapter->fn(dim, x, adapter->comp, val, adapter->ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set velocity Dirichlet BCs on an FlucaFD operator for velocity component d */
static PetscErrorCode SetVelocityDirichletBCs(Phys phys, PetscInt d, FlucaFD fd)
{
  Phys_INS                *ins                          = (Phys_INS *)phys->data;
  FlucaFDBoundaryCondition fd_bcs[2 * PHYS_INS_MAX_DIM] = {{0}};
  PetscInt                 f;

  PetscFunctionBegin;
  for (f = 0; f < 2 * phys->dim; f++) {
    if (ins->bcs[f].type == PHYS_INS_BC_VELOCITY) {
      fd_bcs[f].type = FLUCAFD_BC_DIRICHLET;
      if (ins->bcs[f].fn) {
        ins->bc_adapters[d * PHYS_INS_MAX_FACES + f].fn   = ins->bcs[f].fn;
        ins->bc_adapters[d * PHYS_INS_MAX_FACES + f].ctx  = ins->bcs[f].ctx;
        ins->bc_adapters[d * PHYS_INS_MAX_FACES + f].comp = d;
        fd_bcs[f].fn                                      = PhysINSBCAdapterFn;
        fd_bcs[f].ctx                                     = &ins->bc_adapters[d * PHYS_INS_MAX_FACES + f];
      }
    }
  }
  PetscCall(FlucaFDSetBoundaryConditions(fd, fd_bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set pressure Neumann (zero normal derivative) BCs on an FlucaFD operator */
static PetscErrorCode SetPressureNeumannBCs(Phys phys, FlucaFD fd)
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
  PetscCall(FlucaFDSetBoundaryConditions(fd, fd_bcs));
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
  /* --- Viscous Laplacian per velocity direction --- */
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
    PetscCall(SetVelocityDirichletBCs(phys, d, ins->fd_laplacian[d]));
    PetscCall(FlucaFDSetUp(ins->fd_laplacian[d]));

    for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&comp_ops[e]));
  }

  /* --- Pressure gradient per velocity direction --- */
  for (d = 0; d < dim; d++) {
    /* dp/dx_d from (ELEMENT, dim) to (ELEMENT, d) — 3-point centered */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, d, &ins->fd_grad_p[d]));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_grad_p[d]));
    PetscCall(FlucaFDSetUp(ins->fd_grad_p[d]));
  }

  /* --- Divergence per direction (wide stencil via face interpolation) --- */
  for (d = 0; d < dim; d++) {
    FlucaFD interp, face_deriv;

    /* interp: interpolate u_d from element to face (0th derivative, 2nd order) */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 0, 2, DMSTAG_ELEMENT, d, face_loc[d], 0, &interp));
    PetscCall(FlucaFDSetUp(interp));

    /* face_deriv: d/dx_d from face to element pressure DOF */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, face_loc[d], 0, DMSTAG_ELEMENT, dim, &face_deriv));
    PetscCall(FlucaFDSetUp(face_deriv));

    /* composition: face_deriv(interp(x)) = d(interp(u_d))/dx_d */
    PetscCall(FlucaFDCompositionCreate(interp, face_deriv, &ins->fd_div[d]));
    PetscCall(SetVelocityDirichletBCs(phys, d, ins->fd_div[d]));
    PetscCall(FlucaFDSetUp(ins->fd_div[d]));

    PetscCall(FlucaFDDestroy(&face_deriv));
    PetscCall(FlucaFDDestroy(&interp));
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
    PetscCall(SetVelocityDirichletBCs(phys, e, ins->fd_interp[e]));
    PetscCall(FlucaFDSetUp(ins->fd_interp[e]));
  }

  /* --- Convection operators: C_d = sum_e d/dx_e(F_e * u_d_TVD) where F_e = rho * u_e --- */
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
    PetscCall(SetVelocityDirichletBCs(phys, d, ins->fd_conv[d]));
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
    PetscCall(FlucaFDDestroy(&ins->fd_div[d]));
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
  PetscCall(VecDestroy(&ins->temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Update convection operators with current velocity ------------------- */

static PetscErrorCode UpdateConvectionVelocity_Internal(Phys phys, Vec U)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscReal rho    = ins->rho;
  PetscInt  dim    = phys->dim, d, e;

  PetscFunctionBegin;
  /* Interpolate each velocity component to faces and scale by rho to get mass flux */
  for (e = 0; e < dim; e++) {
    PetscCall(FlucaFDApply(ins->fd_interp[e], sol_dm, ins->dm_face[e], U, ins->mass_flux_face[e]));
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

/* --- Steady residual (shared by SNES and TS paths) ----------------------- */

static PetscErrorCode ComputeSteadyResidual(Phys phys, PetscReal t, Vec x, Vec F)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;
  Vec       temp   = ins->temp;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(F));

  /* Update convection operators with current solution */
  PetscCall(UpdateConvectionVelocity_Internal(phys, x));

  /* Momentum: rho * C_d - mu * nabla^2 u_d + dp/dx_d */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDApply(ins->fd_conv[d], sol_dm, sol_dm, x, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
    PetscCall(FlucaFDApply(ins->fd_laplacian[d], sol_dm, sol_dm, x, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
    PetscCall(FlucaFDApply(ins->fd_grad_p[d], sol_dm, sol_dm, x, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
  }

  /* Continuity: div(u) */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDApply(ins->fd_div[d], sol_dm, sol_dm, x, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
  }

  /* Subtract body force from momentum components */
  if (phys->bodyforce) {
    const PetscScalar **arrc[3] = {NULL, NULL, NULL};
    PetscInt            xs, ys, zs, xm, ym, zm, slot_elem;
    PetscInt            i, j, k;

    PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
    PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
    PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));

    if (dim == 2) {
      for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
          PetscReal     coords[3] = {0};
          PetscScalar   force[3], neg_force;
          DMStagStencil row;

          coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
          coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
          PetscCall(phys->bodyforce(dim, t, coords, force, phys->bodyforce_ctx));

          row.j   = j;
          row.k   = 0;
          row.loc = DMSTAG_ELEMENT;
          for (d = 0; d < dim; d++) {
            row.i     = i;
            row.c     = d;
            neg_force = -force[d];
            PetscCall(DMStagVecSetValuesStencil(sol_dm, F, 1, &row, &neg_force, ADD_VALUES));
          }
        }
      }
    } else if (dim == 3) {
      for (k = zs; k < zs + zm; k++) {
        for (j = ys; j < ys + ym; j++) {
          for (i = xs; i < xs + xm; i++) {
            PetscReal     coords[3];
            PetscScalar   force[3], neg_force;
            DMStagStencil row;

            coords[0] = PetscRealPart(arrc[0][i][slot_elem]);
            coords[1] = PetscRealPart(arrc[1][j][slot_elem]);
            coords[2] = PetscRealPart(arrc[2][k][slot_elem]);
            PetscCall(phys->bodyforce(dim, t, coords, force, phys->bodyforce_ctx));

            row.j   = j;
            row.k   = k;
            row.loc = DMSTAG_ELEMENT;
            for (d = 0; d < dim; d++) {
              row.i     = i;
              row.c     = d;
              neg_force = -force[d];
              PetscCall(DMStagVecSetValuesStencil(sol_dm, F, 1, &row, &neg_force, ADD_VALUES));
            }
          }
        }
      }
    }

    PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
    PetscCall(VecAssemblyBegin(F));
    PetscCall(VecAssemblyEnd(F));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Ops implementations -------------------------------------------------- */

PetscErrorCode PhysComputeIFunction_INS(Phys phys, PetscReal t, Vec U, Vec U_t, Vec F)
{
  Phys_INS     *ins    = (Phys_INS *)phys->data;
  DM            sol_dm = phys->sol_dm;
  PetscInt      dim    = phys->dim, d;
  PetscInt      xs, ys, zs, xm, ym, zm;
  PetscInt      i, j, k;
  DMStagStencil stencil;
  PetscScalar   u_t_val, mass_val;

  PetscFunctionBegin;
  PetscCall(ComputeSteadyResidual(phys, t, U, F));

  /* Add mass term: rho * U_dot for velocity DOFs only (no mass on pressure) */
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));
  stencil.loc = DMSTAG_ELEMENT;
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        stencil.i = i;
        stencil.j = j;
        stencil.k = k;
        for (d = 0; d < dim; d++) {
          stencil.c = d;
          PetscCall(DMStagVecGetValuesStencil(sol_dm, U_t, 1, &stencil, &u_t_val));
          mass_val = ins->rho * u_t_val;
          PetscCall(DMStagVecSetValuesStencil(sol_dm, F, 1, &stencil, &mass_val, ADD_VALUES));
        }
      }
    }
  }
  PetscCall(VecAssemblyBegin(F));
  PetscCall(VecAssemblyEnd(F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  /* Update convection operators with current velocity (Picard linearization) */
  PetscCall(UpdateConvectionVelocity_Internal(phys, U));

  /* Assemble steady Jacobian: convection (Picard) + viscous + pressure gradient + divergence */
  PetscCall(MatZeroEntries(Pmat));
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDGetOperator(ins->fd_conv[d], sol_dm, sol_dm, Pmat));
    PetscCall(FlucaFDGetOperator(ins->fd_laplacian[d], sol_dm, sol_dm, Pmat));
    PetscCall(FlucaFDGetOperator(ins->fd_grad_p[d], sol_dm, sol_dm, Pmat));
  }
  for (d = 0; d < dim; d++) PetscCall(FlucaFDGetOperator(ins->fd_div[d], sol_dm, sol_dm, Pmat));

  /* Add shift * rho to velocity diagonal entries (mass matrix contribution) */
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
  if (Amat != Pmat) {
    PetscCall(MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- TS callbacks --------------------------------------------------------- */

static PetscErrorCode IFunction_INS(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(PhysComputeIFunction_INS((Phys)ctx, t, U, U_t, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IJacobian_INS(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal shift, Mat Amat, Mat Pmat, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(PhysComputeIJacobian_INS((Phys)ctx, t, U, U_t, shift, Amat, Pmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Solver setup --------------------------------------------------------- */

/* Create solver data shared by SNES and TS paths (idempotent) */
static PetscErrorCode PhysINSCreateSolverData_Internal(Phys phys)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;
  MPI_Comm  comm;

  PetscFunctionBegin;
  if (ins->J) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)phys, &comm));

  /* Create Jacobian matrix */
  PetscCall(DMCreateMatrix(sol_dm, &ins->J));
  PetscCall(MatSetOption(ins->J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

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

  /* Null space (if no pressure outlet — all velocity Dirichlet) */
  ins->has_pressure_outlet = PETSC_FALSE;
  for (d = 0; d < 2 * dim; d++) { /* Future: check for PHYS_INS_BC_PRESSURE_OUTLET */
  }
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

PetscErrorCode PhysSetUpTS_INS(Phys phys, TS ts)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  SNES      snes;
  KSP       ksp;
  PC        pc;

  PetscFunctionBegin;
  PetscCall(PhysINSCreateSolverData_Internal(phys));

  /* Wire DM */
  PetscCall(TSSetDM(ts, sol_dm));

  /* Wire TS callbacks */
  PetscCall(TSSetIFunction(ts, NULL, IFunction_INS, phys));
  PetscCall(TSSetIJacobian(ts, ins->J, ins->J, IJacobian_INS, phys));

  /* Default PC: ILU (user can override via options) */
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCILU));
  PetscFunctionReturn(PETSC_SUCCESS);
}
