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

/* Accuracy order of the cell-to-face interpolation T of the Rhie-Chow row, guide eq. (11).

   The continuity row of a cell is the flux difference (U_{f+1} - U_f)/h (guide eq. (10)), so its
   truncation error is the difference of the face errors of T, not the face errors themselves. The
   two-point average leaves e_f = (h^2/8) d2u/dn2 at every interior face; that is a smooth field, its
   difference across a cell is O(h^3), and the interior rows come out second order even though each
   face is only second order. A prescribed-velocity boundary face breaks the argument: there U is the
   boundary datum itself, e = 0 exactly, so the wall cell differences an O(h^2) face error against
   zero and its row is only first order - a one-cell-thick O(h) layer in the truncation error of the
   whole continuity operator, injected into the pressure Poisson right-hand side at every step.

   Neither of the two invariants may be traded away to remove it: U must equal u_b . n exactly on a
   prescribed-velocity face, and D must stay a flux difference so that the continuity rows still sum
   to the net boundary flux. With e_0 = 0 fixed and the differences required to be O(h^3), the face
   errors are forced to be O(h^3) at every face, which is what a fourth-order accurate interpolation
   delivers: T is the four-point interpolation over cells i-2..i+1 for face i.

   T deliberately carries no boundary condition, so next to a wall it is the cubic through the four
   interior cells 0..3 rather than the cubic through the wall datum and cells 0..2. Both are O(h^4)
   accurate at that face, but only the first keeps T a pure interior operator, and T has to be one:
   the Rhie-Chow correction R = T G_c - G^st (guide eq. (11)) applies T to the cell pressure gradient,
   for which no boundary datum exists, and a T whose weights no longer sum to one there would leave a
   spurious O(1) part in R at a wall-adjacent face instead of the O(h^2 d3p/dn3) correction R is meant
   to be. With T interior, R annihilates any pressure that is quadratic in the face-normal direction,
   at every face. The boundary-face rows of the Rhie-Chow block are the boundary condition itself and
   are overwritten with U = u_b . n, so T is never asked for a value there.

   Every continuity row is then second order, on uniform and on smoothly stretched grids alike,
   because FlucaFD builds the stencils from the actual coordinates by a Vandermonde solve. */
static const PetscInt interp_accu_order = 4;

/* --- Operator construction ------------------------------------------------

   The cell-centered pressure gradient carries no boundary condition. At a wall cell FlucaFD
   then resolves the off-grid pressure by quadratic extrapolation, which turns the central
   difference into the second-order one-sided (-3 p_0 + 4 p_1 - p_2)/(2 h). A homogeneous
   Neumann ghost p_{-1} = p_0 would instead leave the first-order (p_1 - p_0)/(2 h): dp/dn is
   genuinely nonzero at a wall, so that ghost belongs to the Poisson problem of the pressure
   increment, not to the pressure gradient of the momentum equation. */

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

/* Local indices of the locally owned face-velocity rows that lie on a non-periodic boundary */
static PetscErrorCode CreateBoundaryFaceRows_Private(Phys phys)
{
  Phys_INS      *ins    = (Phys_INS *)phys->data;
  DM             sol_dm = phys->sol_dm;
  PetscInt       dim    = phys->dim;
  DMBoundaryType bt[3]  = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscInt       N[3] = {1, 1, 1}, s[3] = {0, 0, 0}, m[3] = {1, 1, 1}, extra[3] = {0, 0, 0};
  PetscInt       pass, n, e, d, i, j, k, hi[3], idx[3];
  DMStagStencil  st;

  PetscFunctionBegin;
  PetscCall(DMStagGetBoundaryTypes(sol_dm, &bt[0], &bt[1], &bt[2]));
  PetscCall(DMStagGetGlobalSizes(sol_dm, &N[0], &N[1], &N[2]));
  PetscCall(DMStagGetCorners(sol_dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2], &extra[0], &extra[1], &extra[2]));
  for (d = dim; d < 3; ++d) {
    s[d]     = 0;
    m[d]     = 1;
    extra[d] = 0;
  }
  n = 0;
  for (pass = 0; pass < 2; ++pass) {
    if (pass == 1) {
      PetscCall(PetscMalloc1(n, &ins->bface));
      ins->nbface = n;
      n           = 0;
    }
    for (e = 0; e < dim; ++e) {
      if (bt[e] == DM_BOUNDARY_PERIODIC) continue;
      for (d = 0; d < 3; ++d) hi[d] = s[d] + m[d] + (d == e ? extra[d] : 0);
      for (k = s[2]; k < hi[2]; ++k) {
        for (j = s[1]; j < hi[1]; ++j) {
          for (i = s[0]; i < hi[0]; ++i) {
            idx[0] = i;
            idx[1] = j;
            idx[2] = k;
            if (idx[e] != 0 && idx[e] != N[e]) continue;
            if (pass == 1) {
              st.i   = i;
              st.j   = j;
              st.k   = k;
              st.loc = face_loc[e];
              st.c   = ins->c_U;
              PetscCall(DMStagStencilToIndexLocal(sol_dm, dim, 1, &st, &ins->bface[n]));
            }
            ++n;
          }
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* An empty matrix with the solution DM's parallel layout and index mapping, preallocated for the
   widest block row of T, G_c, G^st or their product. Cheaper and much sparser than DMCreateMatrix(),
   which lays the whole stencil out as explicit zeros; the product of two such matrices would then
   reach the diagonal neighbours that the star preallocation of the system matrix has no room for. */
static PetscErrorCode CreateBlockMatrix_Private(Phys phys, Mat *A)
{
  Phys_INS              *ins = (Phys_INS *)phys->data;
  ISLocalToGlobalMapping ltog;
  PetscInt               n, N;
  const PetscInt         nz = 12;

  PetscFunctionBegin;
  PetscCall(DMGetLocalToGlobalMapping(phys->sol_dm, &ltog));
  PetscCall(VecGetLocalSize(ins->zero, &n));
  PetscCall(VecGetSize(ins->zero, &N));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)phys), A));
  PetscCall(MatSetSizes(*A, n, n, N, N));
  PetscCall(MatSetType(*A, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*A, nz, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*A, nz, NULL, nz, NULL));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltog, ltog));
  PetscCall(MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Operators of the Rhie-Chow rows (guide eq. (11)) and continuity rows (guide eq. (10)).

   R = T G_c - G^st is assembled as an explicit matrix product rather than as a stencil composition.
   A FlucaFD composition expands the raw stencils of its operands and only then resolves the points
   that fall off the grid: next to a wall it resolves off-grid *pressure* cells inside the composed
   stencil, while the matrix T resolves its own off-grid *velocity* cell first and is only then
   multiplied by the cell gradient. The two coincide only while T has no off-grid point there, which was
   true of the two-point average and is not true of the four-point interpolation. Guide eq. (11) means
   the T of the system matrix, so the product is the faithful reading, and only the product keeps
   (-T) G - (-R) equal to -G^st in every row - the identity that makes the fractional step method
   leave the continuity equation unperturbed (guide eq. (17), (19)). R does not depend on dt; the time
   step only scales it by dt/rho, so it is built once here. */
static PetscErrorCode BuildCouplingOperators_Private(Phys phys)
{
  Phys_INS *ins    = (Phys_INS *)phys->data;
  DM        sol_dm = phys->sol_dm;
  PetscInt  dim    = phys->dim, e;
  FlucaFD   div[PHYS_INS_MAX_DIM];
  Mat       Tmat, Gmat, Gstmat;

  PetscFunctionBegin;
  PetscCall(CreateBoundaryFaceRows_Private(phys));
  PetscCall(CreateBlockMatrix_Private(phys, &Tmat));
  PetscCall(CreateBlockMatrix_Private(phys, &Gmat));
  PetscCall(CreateBlockMatrix_Private(phys, &Gstmat));
  for (e = 0; e < dim; e++) {
    FlucaFD Gst;

    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, interp_accu_order, DMSTAG_ELEMENT, ins->c_vel + e, face_loc[e], ins->c_U, &ins->fd_T[e]));
    PetscCall(FlucaFDSetUp(ins->fd_T[e]));
    PetscCall(FlucaFDScaleCreateConstant(ins->fd_T[e], -1., &ins->fd_negT[e]));
    PetscCall(FlucaFDSetUp(ins->fd_negT[e]));

    /* Boundary-face right-hand side. The two-point interpolation has no off-grid point at any
       interior face, so applying it to the zero vector is zero there and exactly u_b . n on a
       boundary face, where the Dirichlet datum carries the whole weight. */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, ins->c_vel + e, face_loc[e], ins->c_U, &ins->fd_bface[e]));
    PetscCall(SetVelocityDirichletBCs(phys, ins->fd_bface[e], e));
    PetscCall(FlucaFDSetUp(ins->fd_bface[e]));

    /* Blocks of R = T G_c - G^st on faces normal to e */
    PetscCall(FlucaFDGetOperator(ins->fd_T[e], sol_dm, sol_dm, Tmat));
    PetscCall(FlucaFDGetOperator(ins->fd_grad_p[e], sol_dm, sol_dm, Gmat));
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, ins->c_p, face_loc[e], ins->c_U, &Gst));
    PetscCall(FlucaFDSetUp(Gst));
    PetscCall(FlucaFDGetOperator(Gst, sol_dm, sol_dm, Gstmat));
    PetscCall(FlucaFDDestroy(&Gst));
  }
  PetscCall(MatAssemblyBegin(Tmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Tmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Gmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Gstmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Gstmat, MAT_FINAL_ASSEMBLY));

  /* A boundary-face row is the boundary condition, not a Rhie-Chow row: PhysComputeCouplingSystem
     replaces it by a unit row. Emptying it here keeps R off the cells that only the one-sided
     interpolation at a boundary face reaches, which lie outside the star the system matrix is
     preallocated for. */
  PetscCall(MatZeroRowsLocal(Tmat, ins->nbface, ins->bface, 0., NULL, NULL));
  PetscCall(MatZeroRowsLocal(Gstmat, ins->nbface, ins->bface, 0., NULL, NULL));

  PetscCall(MatMatMult(Tmat, Gmat, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &ins->negR));
  PetscCall(MatAXPY(ins->negR, -1., Gstmat, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatScale(ins->negR, -1.));
  PetscCall(MatEliminateZeros(ins->negR, PETSC_FALSE));
  PetscCall(MatDestroy(&Gstmat));
  PetscCall(MatDestroy(&Gmat));
  PetscCall(MatDestroy(&Tmat));

  for (e = 0; e < dim; e++) {
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], ins->c_U, DMSTAG_ELEMENT, ins->c_p, &div[e]));
    PetscCall(FlucaFDSetUp(div[e]));
  }
  PetscCall(FlucaFDSumCreate(dim, div, &ins->fd_D));
  PetscCall(FlucaFDSetUp(ins->fd_D));
  for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&div[e]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSBuildOperators_Internal(Phys phys)
{
  Phys_INS   *ins    = (Phys_INS *)phys->data;
  DM          sol_dm = phys->sol_dm;
  PetscInt    dim    = phys->dim, d, e;
  PetscScalar mu;

  PetscFunctionBegin;
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
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
    PetscCall(FlucaFDSetUp(ins->fd_grad_p[d]));
  }

  PetscCall(BuildMomentumOperators_Private(phys));
  PetscCall(BuildCouplingOperators_Private(phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSDestroyOperators_Internal(Phys phys)
{
  Phys_INS *ins = (Phys_INS *)phys->data;
  PetscInt  d, e;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ins->negR));
  for (d = 0; d < PHYS_INS_MAX_DIM; d++) {
    PetscCall(FlucaFDDestroy(&ins->fd_bface[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_negT[d]));
    PetscCall(FlucaFDDestroy(&ins->fd_T[d]));
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
  PetscCall(FlucaFDDestroy(&ins->fd_D));
  PetscCall(PetscFree(ins->bface));
  PetscCall(DMDestroy(&ins->dm_face));
  PetscCall(VecDestroy(&ins->zero));
  PetscCall(ISDestroy(&ins->is_vel));
  PetscFunctionReturn(PETSC_SUCCESS);
}
