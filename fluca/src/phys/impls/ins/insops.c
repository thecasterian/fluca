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

/* Set velocity Dirichlet BCs of velocity component d on a FlucaFD operator.
   Uses the BC adapter to bridge PhysINSBCFn (has comp) to FlucaFDBCValueFn (no comp). */
static PetscErrorCode SetVelocityDirichletBCs(Phys phys, FlucaFD fd, PetscInt d)
{
  Phys_INS                *ins                          = (Phys_INS *)phys->data;
  FlucaFDBoundaryCondition fd_bcs[2 * PHYS_INS_MAX_DIM] = {{0}};
  PetscInt                 f;

  PetscFunctionBegin;
  for (f = 0; f < 2 * phys->dim; f++) {
    if (ins->bcs[f].type == PHYS_INS_BC_VELOCITY && ins->bcs[f].fn) {
      ins->bc_adapters[d][f].fn         = ins->bcs[f].fn;
      ins->bc_adapters[d][f].fn_dot     = ins->bcs[f].fn_dot;
      ins->bc_adapters[d][f].fn_ctx     = ins->bcs[f].ctx;
      ins->bc_adapters[d][f].fn_dot_ctx = ins->bcs[f].fn_dot_ctx;
      ins->bc_adapters[d][f].comp       = d;
      fd_bcs[f].type                    = FLUCAFD_BC_DIRICHLET;
      fd_bcs[f].fn                      = PhysINS_BCAdapterFn;
      fd_bcs[f].fn_ctx                  = &ins->bc_adapters[d][f];
      fd_bcs[f].fn_dot                  = ins->bcs[f].fn_dot ? PhysINS_BCAdapterFnDot : NULL;
      fd_bcs[f].fn_dot_ctx              = &ins->bc_adapters[d][f];
    } else if (ins->bcs[f].type == PHYS_INS_BC_VELOCITY) {
      /* Constant zero velocity BC */
      fd_bcs[f].type  = FLUCAFD_BC_DIRICHLET;
      fd_bcs[f].value = 0.;
    }
  }
  PetscCall(FlucaFDSetBoundaryConditions(fd, ins->c_vel + d, fd_bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set pressure Neumann (zero normal derivative) BCs on a FlucaFD operator */
static PetscErrorCode SetPressureNeumannBCs(Phys phys, FlucaFD fd)
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
  PetscCall(FlucaFDSetBoundaryConditions(fd, ins->c_p, fd_bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Operator construction ------------------------------------------------ */

/* Operators of the momentum rows: A = I + (dt/2) J - (dt/2) nu lap and G = (dt/rho) grad.
   Coefficients depending on dt and the linearization state are set per step. */
static PetscErrorCode BuildMomentumOperators_Private(Phys phys)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d, e;
  DM        cdm;

  PetscFunctionBegin;
  PetscCall(PhysGetFieldIS_Internal(phys, PHYS_FIELD_VELOCITY, &ins->is_vel));
  PetscCall(DMCreateGlobalVector(sol_dm, &ins->zero));
  PetscCall(VecZeroEntries(ins->zero));

  /* ubar_d^n lives on a one-DOF-per-face DM */
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
  for (d = 0; d < dim; d++) {
    PetscCall(DMCreateGlobalVector(ins->dm_face, &ins->ubar[d]));
    for (e = 0; e < dim; e++) {
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, ins->c_vel + d, face_loc[e], 0, &ins->fd_interp_vel[d][e]));
      PetscCall(SetVelocityDirichletBCs(phys, ins->fd_interp_vel[d][e], d));
      PetscCall(FlucaFDSetUp(ins->fd_interp_vel[d][e]));
    }
  }

  /* Viscous and pressure-gradient blocks */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDScaleCreateConstant(ins->fd_laplacian[d], 0., &ins->fd_visc[d]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_visc[d], d));
    PetscCall(FlucaFDSetUp(ins->fd_visc[d]));
    PetscCall(FlucaFDScaleCreateConstant(ins->fd_grad_p[d], 0., &ins->fd_grad[d]));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_grad[d]));
    PetscCall(FlucaFDSetUp(ins->fd_grad[d]));
  }

  /* Linearized convection, guide eq. (5) and section Spatial Discretization:
     d/dx_e(ubar_d^{n+1} U_e^n + ubar_d^n ubar_e^{n+1}) */
  for (d = 0; d < dim; d++) {
    FlucaFD terms[2 * PHYS_INS_MAX_DIM], sum;

    for (e = 0; e < dim; e++) {
      FlucaFD interp_d, interp_e, outer;

      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, ins->c_vel + d, face_loc[e], ins->c_U, &interp_d));
      PetscCall(FlucaFDSetUp(interp_d));
      PetscCall(FlucaFDScaleCreateVector(interp_d, ins->zero, ins->c_U, &ins->fd_conv_U[d][e]));
      PetscCall(FlucaFDSetUp(ins->fd_conv_U[d][e]));
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, ins->c_vel + e, face_loc[e], ins->c_U, &interp_e));
      PetscCall(FlucaFDSetUp(interp_e));
      PetscCall(FlucaFDScaleCreateVector(interp_e, ins->ubar[d], 0, &ins->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDSetUp(ins->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], ins->c_U, DMSTAG_ELEMENT, ins->c_vel + d, &outer));
      PetscCall(FlucaFDSetUp(outer));
      PetscCall(FlucaFDCompositionCreate(ins->fd_conv_U[d][e], outer, &terms[2 * e]));
      PetscCall(FlucaFDSetUp(terms[2 * e]));
      PetscCall(FlucaFDCompositionCreate(ins->fd_conv_ubar[d][e], outer, &terms[2 * e + 1]));
      PetscCall(FlucaFDSetUp(terms[2 * e + 1]));
      PetscCall(FlucaFDDestroy(&outer));
      PetscCall(FlucaFDDestroy(&interp_e));
      PetscCall(FlucaFDDestroy(&interp_d));
    }
    PetscCall(FlucaFDSumCreate(2 * dim, terms, &sum));
    PetscCall(FlucaFDSetUp(sum));
    PetscCall(FlucaFDScaleCreateConstant(sum, 0., &ins->fd_conv[d]));
    for (e = 0; e < dim; e++) PetscCall(SetVelocityDirichletBCs(phys, ins->fd_conv[d], e));
    PetscCall(FlucaFDSetUp(ins->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&sum));
    for (e = 0; e < 2 * dim; e++) PetscCall(FlucaFDDestroy(&terms[e]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSBuildOperators_Internal(Phys phys)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d, e;
  PetscReal mu     = ins->mu;

  PetscFunctionBegin;
  /* --- fd_laplacian[d] = sum_e d/dx_e(-mu * d(u_d)/dx_e) --- */
  for (d = 0; d < dim; d++) {
    FlucaFD comp_ops[PHYS_INS_MAX_DIM];

    for (e = 0; e < dim; e++) {
      FlucaFD inner, scaled, outer;

      /* d(u_d)/dx_e */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, ins->c_vel + d, face_loc[e], ins->c_U, &inner));
      PetscCall(FlucaFDSetUp(inner));

      /* -mu * d(u_d)/dx_e */
      PetscCall(FlucaFDScaleCreateConstant(inner, -mu, &scaled));
      PetscCall(FlucaFDSetUp(scaled));

      /* d/dx_e(...) back to element */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], ins->c_U, DMSTAG_ELEMENT, ins->c_vel + d, &outer));
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
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, ins->c_p, DMSTAG_ELEMENT, ins->c_vel + d, &ins->fd_grad_p[d]));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_grad_p[d]));
    PetscCall(FlucaFDSetUp(ins->fd_grad_p[d]));
  }

  PetscCall(BuildMomentumOperators_Private(phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSDestroyOperators_Internal(Phys phys)
{
  Phys_INS *ins = (Phys_INS *)phys->data;
  PetscInt  d, e;

  PetscFunctionBegin;
  for (d = 0; d < PHYS_INS_MAX_DIM; d++) {
    PetscCall(FlucaFDDestroy(&ins->fd_conv[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_grad[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_visc[d]));
    for (e = 0; e < PHYS_INS_MAX_DIM; e++) {
      PetscCall(FlucaFDDestroy(&ins->fd_conv_ubar[d][e]));
      PetscCall(FlucaFDDestroy(&ins->fd_conv_U[d][e]));
      PetscCall(FlucaFDDestroy(&ins->fd_interp_vel[d][e]));
    }
    PetscCall(VecDestroy(&ins->ubar[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_laplacian[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_grad_p[d]));
  }
  PetscCall(DMDestroy(&ins->dm_face));
  PetscCall(VecDestroy(&ins->zero));
  PetscCall(ISDestroy(&ins->is_vel));
  PetscFunctionReturn(PETSC_SUCCESS);
}
