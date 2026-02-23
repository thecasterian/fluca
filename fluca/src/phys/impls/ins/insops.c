#include "insimpl.h"
#include <petscsnes.h>
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
  Phys_INS           *ins    = (Phys_INS *)phys->data;
  DM                  sol_dm = phys->sol_dm;
  PetscInt            dim    = phys->dim, d, e;
  PetscReal           mu = ins->mu, rho = ins->rho;
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            slot_prev;
  PetscReal           h[3], V_cell, a_P;

  PetscFunctionBegin;
  /* Compute grid spacing from first cell for stabilization coefficient */
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_LEFT, &slot_prev));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  V_cell = 1.0;
  a_P    = 0.0;
  for (d = 0; d < dim; d++) {
    PetscInt xs;

    PetscCall(DMStagGetCorners(sol_dm, (d == 0) ? &xs : NULL, (d == 1) ? &xs : NULL, (d == 2) ? &xs : NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    h[d] = PetscRealPart(arrc[d][xs + 1][slot_prev] - arrc[d][xs][slot_prev]);
    V_cell *= h[d];
    a_P += 2.0 / (h[d] * h[d]);
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  a_P *= mu;
  ins->alpha = V_cell / a_P;

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
    FlucaFD deriv;

    /* d/dx_d from (ELEMENT, dim) to (ELEMENT, d) — 3-point centered */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, DMSTAG_ELEMENT, d, &deriv));
    PetscCall(FlucaFDSetUp(deriv));

    /* (1/rho) * dp/dx_d */
    PetscCall(FlucaFDScaleCreateConstant(deriv, 1.0 / rho, &ins->fd_grad_p[d]));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_grad_p[d]));
    PetscCall(FlucaFDSetUp(ins->fd_grad_p[d]));

    PetscCall(FlucaFDDestroy(&deriv));
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

  /* --- Pressure stabilization: -alpha * compact Laplacian of p --- */
  {
    FlucaFD comp_ops[PHYS_INS_MAX_DIM];
    FlucaFD pstab_sum;

    for (d = 0; d < dim; d++) {
      FlucaFD inner_d, outer_d;

      /* inner_d: d/dx_d from (ELEMENT, dim) to (face_loc[d], 0) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, dim, face_loc[d], 0, &inner_d));
      PetscCall(FlucaFDSetUp(inner_d));

      /* outer_d: d/dx_d from (face_loc[d], 0) to (ELEMENT, dim) */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, face_loc[d], 0, DMSTAG_ELEMENT, dim, &outer_d));
      PetscCall(FlucaFDSetUp(outer_d));

      /* composition: d^2p/dx_d^2 */
      PetscCall(FlucaFDCompositionCreate(inner_d, outer_d, &comp_ops[d]));
      PetscCall(FlucaFDSetUp(comp_ops[d]));

      PetscCall(FlucaFDDestroy(&outer_d));
      PetscCall(FlucaFDDestroy(&inner_d));
    }

    /* Sum: compact Laplacian of p */
    PetscCall(FlucaFDSumCreate(dim, comp_ops, &pstab_sum));
    PetscCall(FlucaFDSetUp(pstab_sum));

    /* Scale by -alpha */
    PetscCall(FlucaFDScaleCreateConstant(pstab_sum, -ins->alpha, &ins->fd_pstab));
    PetscCall(SetPressureNeumannBCs(phys, ins->fd_pstab));
    PetscCall(FlucaFDSetUp(ins->fd_pstab));

    PetscCall(FlucaFDDestroy(&pstab_sum));
    for (d = 0; d < dim; d++) PetscCall(FlucaFDDestroy(&comp_ops[d]));
  }

  /* Create temp vector for residual assembly */
  PetscCall(DMCreateGlobalVector(sol_dm, &ins->temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSDestroyOperators_Internal(Phys phys)
{
  Phys_INS *ins = (Phys_INS *)phys->data;
  PetscInt  d;

  PetscFunctionBegin;
  for (d = 0; d < PHYS_INS_MAX_DIM; d++) {
    PetscCall(FlucaFDDestroy(&ins->fd_laplacian[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_grad_p[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_div[d]));
  }
  PetscCall(FlucaFDDestroy(&ins->fd_pstab));
  PetscCall(VecDestroy(&ins->temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Steady residual (shared by SNES and TS paths) ----------------------- */

static PetscErrorCode ComputeSteadyResidual_Stokes(Phys phys, PetscReal t, Vec x, Vec F)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;
  Vec       temp   = ins->temp;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(F));

  /* Momentum: -mu * nabla^2 u_d + (1/rho) * dp/dx_d */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDApply(ins->fd_laplacian[d], sol_dm, sol_dm, x, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
    PetscCall(FlucaFDApply(ins->fd_grad_p[d], sol_dm, sol_dm, x, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
  }

  /* Continuity: div(u) - alpha * nabla^2 p */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDApply(ins->fd_div[d], sol_dm, sol_dm, x, temp));
    PetscCall(VecAXPY(F, 1.0, temp));
  }
  PetscCall(FlucaFDApply(ins->fd_pstab, sol_dm, sol_dm, x, temp));
  PetscCall(VecAXPY(F, 1.0, temp));

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

/* --- SNES callbacks ------------------------------------------------------- */

static PetscErrorCode FormFunction_Stokes(SNES snes, Vec x, Vec F, void *ctx)
{
  Phys phys = (Phys)ctx;

  PetscFunctionBegin;
  PetscCall(ComputeSteadyResidual_Stokes(phys, 0.0, x, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian_Stokes(SNES snes, Vec x, Mat J, Mat Jpre, void *ctx)
{
  Phys      phys   = (Phys)ctx;
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, d;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(Jpre));

  /* Momentum rows */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDGetOperator(ins->fd_laplacian[d], sol_dm, sol_dm, Jpre));
    PetscCall(FlucaFDGetOperator(ins->fd_grad_p[d], sol_dm, sol_dm, Jpre));
  }

  /* Continuity rows */
  for (d = 0; d < dim; d++) PetscCall(FlucaFDGetOperator(ins->fd_div[d], sol_dm, sol_dm, Jpre));
  PetscCall(FlucaFDGetOperator(ins->fd_pstab, sol_dm, sol_dm, Jpre));

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- TS callbacks --------------------------------------------------------- */

static PetscErrorCode IFunction_Stokes(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, void *ctx)
{
  Phys          phys   = (Phys)ctx;
  Phys_INS     *ins    = (Phys_INS *)phys->data;
  DM            sol_dm = phys->sol_dm;
  PetscInt      dim    = phys->dim, d;
  PetscInt      xs, ys, zs, xm, ym, zm;
  PetscInt      i, j, k;
  DMStagStencil stencil;
  PetscScalar   u_t_val, mass_val;

  PetscFunctionBegin;
  PetscCall(ComputeSteadyResidual_Stokes(phys, t, U, F));

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

static PetscErrorCode IJacobian_Stokes(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal shift, Mat Amat, Mat Pmat, void *ctx)
{
  Phys          phys   = (Phys)ctx;
  Phys_INS     *ins    = (Phys_INS *)phys->data;
  DM            sol_dm = phys->sol_dm;
  PetscInt      dim    = phys->dim, d;
  PetscInt      xs, ys, zs, xm, ym, zm;
  PetscInt      i, j, k;
  DMStagStencil row;
  PetscScalar   val;

  PetscFunctionBegin;
  /* Assemble steady Stokes Jacobian */
  PetscCall(MatZeroEntries(Pmat));
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDGetOperator(ins->fd_laplacian[d], sol_dm, sol_dm, Pmat));
    PetscCall(FlucaFDGetOperator(ins->fd_grad_p[d], sol_dm, sol_dm, Pmat));
  }
  for (d = 0; d < dim; d++) PetscCall(FlucaFDGetOperator(ins->fd_div[d], sol_dm, sol_dm, Pmat));
  PetscCall(FlucaFDGetOperator(ins->fd_pstab, sol_dm, sol_dm, Pmat));

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

PetscErrorCode PhysSetUpSNES_INS(Phys phys, SNES snes)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  KSP       ksp;
  PC        pc;

  PetscFunctionBegin;
  PetscCall(PhysINSCreateSolverData_Internal(phys));

  /* Wire DM */
  PetscCall(SNESSetDM(snes, sol_dm));

  /* Pre-assemble Jacobian (Stokes is linear) */
  PetscCall(FormJacobian_Stokes(snes, NULL, ins->J, ins->J, phys));

  /* Wire SNES callbacks */
  PetscCall(SNESSetFunction(snes, NULL, FormFunction_Stokes, phys));
  PetscCall(SNESSetJacobian(snes, ins->J, ins->J, FormJacobian_Stokes, phys));

  /* Default PC: ILU (user can override via options) */
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCILU));
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
  PetscCall(TSSetIFunction(ts, NULL, IFunction_Stokes, phys));
  PetscCall(TSSetIJacobian(ts, ins->J, ins->J, IJacobian_Stokes, phys));

  /* Default PC: ILU (user can override via options) */
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCILU));
  PetscFunctionReturn(PETSC_SUCCESS);
}
