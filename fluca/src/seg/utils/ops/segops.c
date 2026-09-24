#include <fluca/private/segopsimpl.h>

/* --- BC adapter functions ------------------------------------------------- */

static PetscErrorCode SegSpatialOpsBCAdapterFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  Seg_BCAdapter *a = (Seg_BCAdapter *)ctx;

  PetscFunctionBegin;
  PetscCall(a->fn(dim, t, x, a->comp, value, a->fn_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegSpatialOpsBCAdapterFnDot(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  Seg_BCAdapter *a = (Seg_BCAdapter *)ctx;

  PetscFunctionBegin;
  PetscCall(a->fn_dot(dim, t, x, a->comp, value, a->fn_dot_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set velocity Dirichlet BCs of velocity component d on a FlucaFD operator.
   Uses the BC adapter of ops to bridge PhysBCFn (has comp) to FlucaFDBCValueFn (no comp). */
PetscErrorCode SegSpatialOpsSetVelocityBCs_Internal(Phys phys, SegSpatialOps *ops, FlucaFD fd, PetscInt d)
{
  FlucaFDBoundaryCondition fd_bcs[2 * FLUCA_MAX_DIM] = {{0}};
  PhysBC                   bc;
  PetscInt                 f;

  PetscFunctionBegin;
  for (f = 0; f < 2 * ops->dim; f++) {
    PetscCall(PhysGetBoundaryCondition(phys, f, &bc));
    if (bc.type == PHYS_BC_VELOCITY && bc.fn) {
      ops->bc_adapters[d][f].fn         = bc.fn;
      ops->bc_adapters[d][f].fn_dot     = bc.fn_dot;
      ops->bc_adapters[d][f].fn_ctx     = bc.ctx;
      ops->bc_adapters[d][f].fn_dot_ctx = bc.fn_dot_ctx;
      ops->bc_adapters[d][f].comp       = d;
      fd_bcs[f].type                    = FLUCAFD_BC_DIRICHLET;
      fd_bcs[f].fn                      = SegSpatialOpsBCAdapterFn;
      fd_bcs[f].fn_ctx                  = &ops->bc_adapters[d][f];
      fd_bcs[f].fn_dot                  = bc.fn_dot ? SegSpatialOpsBCAdapterFnDot : NULL;
      fd_bcs[f].fn_dot_ctx              = &ops->bc_adapters[d][f];
    } else if (bc.type == PHYS_BC_VELOCITY) {
      /* Constant zero velocity BC */
      fd_bcs[f].type  = FLUCAFD_BC_DIRICHLET;
      fd_bcs[f].value = 0.;
    }
  }
  PetscCall(FlucaFDSetBoundaryConditions(fd, ops->c_vel + d, fd_bcs));
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

/* Local indices of the locally owned face-velocity rows that lie on a non-periodic boundary */
static PetscErrorCode CreateBoundaryFaceRows_Private(Phys phys, SegSpatialOps *ops)
{
  PetscInt       dim   = ops->dim;
  DMBoundaryType bt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscInt       N[3] = {1, 1, 1}, s[3] = {0, 0, 0}, m[3] = {1, 1, 1}, extra[3] = {0, 0, 0};
  PetscInt       pass, n, e, d, i, j, k, hi[3], idx[3];
  DMStagStencil  st;
  DM             sol_dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
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
      PetscCall(PetscMalloc1(n, &ops->bface));
      ops->nbface = n;
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
              st.c   = ops->c_U;
              PetscCall(DMStagStencilToIndexLocal(sol_dm, dim, 1, &st, &ops->bface[n]));
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
static PetscErrorCode CreateBlockMatrix_Private(Phys phys, SegSpatialOps *ops, Mat *A)
{
  ISLocalToGlobalMapping ltog;
  PetscInt               n, N;
  const PetscInt         nz = 12;
  DM                     sol_dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMGetLocalToGlobalMapping(sol_dm, &ltog));
  PetscCall(VecGetLocalSize(ops->zero, &n));
  PetscCall(VecGetSize(ops->zero, &N));
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
static PetscErrorCode BuildCouplingOperators_Private(Phys phys, SegSpatialOps *ops)
{
  PetscInt dim = ops->dim, e;
  FlucaFD  div[FLUCA_MAX_DIM];
  Mat      Tmat, Gmat, Gstmat;
  DM       sol_dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(CreateBoundaryFaceRows_Private(phys, ops));
  PetscCall(CreateBlockMatrix_Private(phys, ops, &Tmat));
  PetscCall(CreateBlockMatrix_Private(phys, ops, &Gmat));
  PetscCall(CreateBlockMatrix_Private(phys, ops, &Gstmat));
  for (e = 0; e < dim; e++) {
    FlucaFD Gst;

    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, interp_accu_order, DMSTAG_ELEMENT, ops->c_vel + e, face_loc[e], ops->c_U, &ops->fd_T[e]));
    PetscCall(FlucaFDSetUp(ops->fd_T[e]));
    PetscCall(FlucaFDScaleCreateConstant(ops->fd_T[e], -1., &ops->fd_negT[e]));
    PetscCall(FlucaFDSetUp(ops->fd_negT[e]));

    /* Boundary-face right-hand side. The two-point interpolation has no off-grid point at any
       interior face, so applying it to the zero vector is zero there and exactly u_b . n on a
       boundary face, where the Dirichlet datum carries the whole weight. */
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, ops->c_vel + e, face_loc[e], ops->c_U, &ops->fd_bface[e]));
    PetscCall(SegSpatialOpsSetVelocityBCs_Internal(phys, ops, ops->fd_bface[e], e));
    PetscCall(FlucaFDSetUp(ops->fd_bface[e]));

    /* Blocks of R = T G_c - G^st on faces normal to e */
    PetscCall(FlucaFDGetOperator(ops->fd_T[e], sol_dm, sol_dm, Tmat));
    PetscCall(FlucaFDGetOperator(ops->fd_grad_p[e], sol_dm, sol_dm, Gmat));
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, ops->c_p, face_loc[e], ops->c_U, &Gst));
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

  /* A boundary-face row is the boundary condition, not a Rhie-Chow row: the coupling rows of the
     system replace it by a unit row. Emptying it here keeps R off the cells that only the one-sided
     interpolation at a boundary face reaches, which lie outside the star the system matrix is
     preallocated for. */
  PetscCall(MatZeroRowsLocal(Tmat, ops->nbface, ops->bface, 0., NULL, NULL));
  PetscCall(MatZeroRowsLocal(Gstmat, ops->nbface, ops->bface, 0., NULL, NULL));

  PetscCall(MatMatMult(Tmat, Gmat, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &ops->negR));
  PetscCall(MatAXPY(ops->negR, -1., Gstmat, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatScale(ops->negR, -1.));
  PetscCall(MatEliminateZeros(ops->negR, PETSC_FALSE));
  PetscCall(MatDestroy(&Gstmat));
  PetscCall(MatDestroy(&Gmat));
  PetscCall(MatDestroy(&Tmat));

  for (e = 0; e < dim; e++) {
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], ops->c_U, DMSTAG_ELEMENT, ops->c_p, &div[e]));
    PetscCall(FlucaFDSetUp(div[e]));
  }
  PetscCall(FlucaFDSumCreate(dim, div, &ops->fd_D));
  PetscCall(FlucaFDSetUp(ops->fd_D));
  for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&div[e]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSpatialOpsBuild_Internal(Phys phys, SegSpatialOps *ops)
{
  DMBoundaryType   bt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PhysBC           bc_lo, bc_hi;
  PhysEquationRole role;
  const char      *name;
  PetscScalar      mu;
  PetscInt         dim, sw, d, e, k, nfields;
  DM               sol_dm;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMGetDimension(sol_dm, &dim));
  ops->dim = dim;

  /* The Rhie-Chow correction R = T G_c - G^st composes the four-point interpolation T with the
     three-point cell gradient. Next to a wall the interpolation is folded onto four interior cells,
     and a face row then reaches four elements away on the side the folding points into. */
  PetscCall(DMStagGetStencilWidth(sol_dm, &sw));
  PetscCheck(sw >= 4, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "The Seg spatial operators require a base DM stencil width of at least 4, got %" PetscInt_FMT, sw);
  /* Only velocity boundary conditions are supported: every non-periodic boundary needs one */
  PetscCall(DMStagGetBoundaryTypes(sol_dm, &bt[0], &bt[1], &bt[2]));
  for (d = 0; d < dim; ++d) {
    if (bt[d] == DM_BOUNDARY_PERIODIC) continue;
    PetscCall(PhysGetBoundaryCondition(phys, 2 * d, &bc_lo));
    PetscCall(PhysGetBoundaryCondition(phys, 2 * d + 1, &bc_hi));
    PetscCheck(bc_lo.type == PHYS_BC_VELOCITY && bc_hi.type == PHYS_BC_VELOCITY, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "The Seg spatial operators require a velocity boundary condition on both non-periodic boundaries in direction %" PetscInt_FMT, d);
  }

  /* These operators only cover fields with a momentum, pressure or auxiliary role; a
     transported-scalar field, or any other field they do not know how to handle, would be laid into
     the solution DM with no rows ever written for it, leaving a singular system with no diagnostic.
     Every declared field must be checked, not just the ones looked up by name here, since a Phys
     subtype may declare additional fields these operators are unaware of. */
  PetscCall(PhysGetNumFields(phys, &nfields));
  for (k = 0; k < nfields; ++k) {
    PetscCall(PhysGetFieldName(phys, k, &name));
    PetscCall(PhysGetFieldRole(phys, name, &role));
    PetscCheck(role == PHYS_EQN_MOMENTUM || role == PHYS_EQN_PRESSURE || role == PHYS_EQN_AUXILIARY, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "The Seg spatial operators cannot be built for field %s with equation role %s", name, PhysEquationRoles[role]);
  }

  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &ops->c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &ops->c_p, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &ops->c_U, NULL));

  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
  /* --- fd_laplacian[d] = sum_e d/dx_e(-mu * d(u_d)/dx_e) --- */
  for (d = 0; d < dim; d++) {
    FlucaFD comp_ops[FLUCA_MAX_DIM];

    for (e = 0; e < dim; e++) {
      FlucaFD inner, outer;

      /* d(u_d)/dx_e */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, DMSTAG_ELEMENT, ops->c_vel + d, face_loc[e], ops->c_U, &inner));
      PetscCall(FlucaFDSetUp(inner));

      /* -mu * d(u_d)/dx_e. Kept (not destroyed) so SegSpatialOpsUpdateProperties_Internal can
         rescale it when the viscosity changes after this operator is built. */
      PetscCall(FlucaFDScaleCreateConstant(inner, -mu, &ops->fd_negmu[d][e]));
      PetscCall(FlucaFDSetUp(ops->fd_negmu[d][e]));

      /* d/dx_e(...) back to element */
      PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)e, 1, 2, face_loc[e], ops->c_U, DMSTAG_ELEMENT, ops->c_vel + d, &outer));
      PetscCall(FlucaFDSetUp(outer));

      /* d/dx_e(-mu * d(u_d)/dx_e) */
      PetscCall(FlucaFDCompositionCreate(ops->fd_negmu[d][e], outer, &comp_ops[e]));
      PetscCall(FlucaFDSetUp(comp_ops[e]));

      PetscCall(FlucaFDDestroy(&outer));
      PetscCall(FlucaFDDestroy(&inner));
    }

    PetscCall(FlucaFDSumCreate(dim, comp_ops, &ops->fd_laplacian[d]));
    PetscCall(SegSpatialOpsSetVelocityBCs_Internal(phys, ops, ops->fd_laplacian[d], d));
    PetscCall(FlucaFDSetUp(ops->fd_laplacian[d]));

    for (e = 0; e < dim; e++) PetscCall(FlucaFDDestroy(&comp_ops[e]));
  }

  /* --- fd_grad_p[d] = dp/dx_d --- */
  for (d = 0; d < dim; d++) {
    PetscCall(FlucaFDDerivativeCreate(sol_dm, (FlucaFDDirection)d, 1, 2, DMSTAG_ELEMENT, ops->c_p, DMSTAG_ELEMENT, ops->c_vel + d, &ops->fd_grad_p[d]));
    PetscCall(FlucaFDSetUp(ops->fd_grad_p[d]));
  }

  /* --- zero solution vector --- */
  PetscCall(DMCreateGlobalVector(sol_dm, &ops->zero));
  PetscCall(VecZeroEntries(ops->zero));

  PetscCall(BuildCouplingOperators_Private(phys, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Point where the spatial operators pick up the current material properties at each assembly.
   fd_negmu[d][e] is the scale nested inside fd_laplacian[d]; a composition or sum fetches its
   operands' stencils at apply time, so rescaling it here changes what fd_laplacian[d] (and any
   operator built on it, such as Seg_CNLinear's fd_visc[d]) produce on the next apply. */
PetscErrorCode SegSpatialOpsUpdateProperties_Internal(Phys phys, SegSpatialOps *ops)
{
  PetscScalar mu;
  PetscInt    d, e;

  PetscFunctionBegin;
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
  for (d = 0; d < ops->dim; d++) {
    for (e = 0; e < ops->dim; e++) PetscCall(FlucaFDScaleSetConstant(ops->fd_negmu[d][e], -mu));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSpatialOpsDestroy_Internal(SegSpatialOps *ops)
{
  PetscInt d, e;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ops->negR));
  for (d = 0; d < FLUCA_MAX_DIM; d++) {
    PetscCall(FlucaFDDestroy(&ops->fd_bface[d]));
    PetscCall(FlucaFDDestroy(&ops->fd_negT[d]));
    PetscCall(FlucaFDDestroy(&ops->fd_T[d]));
    for (e = 0; e < FLUCA_MAX_DIM; e++) { PetscCall(FlucaFDDestroy(&ops->fd_negmu[d][e])); }
    PetscCall(FlucaFDDestroy(&ops->fd_laplacian[d]));
    PetscCall(FlucaFDDestroy(&ops->fd_grad_p[d]));
  }
  PetscCall(FlucaFDDestroy(&ops->fd_D));
  PetscCall(PetscFree(ops->bface));
  PetscCall(VecDestroy(&ops->zero));
  PetscFunctionReturn(PETSC_SUCCESS);
}
