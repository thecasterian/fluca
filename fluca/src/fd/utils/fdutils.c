#include <fluca/private/flucafdimpl.h>

PetscErrorCode FlucaFDValidateStencilLocation_Internal(DMStagStencilLocation loc)
{
  PetscFunctionBegin;
  /* Only ELEMENT, LEFT, DOWN, BACK, and their combinations are allowed */
  switch (loc) {
  case DMSTAG_ELEMENT:
  case DMSTAG_LEFT:
  case DMSTAG_DOWN:
  case DMSTAG_BACK:
  case DMSTAG_DOWN_LEFT:
  case DMSTAG_BACK_LEFT:
  case DMSTAG_BACK_DOWN:
  case DMSTAG_BACK_DOWN_LEFT:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid stencil location %s; only ELEMENT, LEFT, DOWN, BACK, and their combinations are allowed", DMStagStencilLocations[loc]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDUseFaceCoordinate_Internal(DMStagStencilLocation loc, PetscInt dir, PetscBool *use_face)
{
  PetscFunctionBegin;
  switch (dir) {
  case 0:
    /* Check for LEFT */
    *use_face = (loc == DMSTAG_LEFT || loc == DMSTAG_DOWN_LEFT || loc == DMSTAG_BACK_LEFT || loc == DMSTAG_BACK_DOWN_LEFT);
    break;
  case 1:
    /* Check for DOWN */
    *use_face = (loc == DMSTAG_DOWN || loc == DMSTAG_DOWN_LEFT || loc == DMSTAG_BACK_DOWN || loc == DMSTAG_BACK_DOWN_LEFT);
    break;
  case 2:
    /* Check for BACK */
    *use_face = (loc == DMSTAG_BACK || loc == DMSTAG_BACK_LEFT || loc == DMSTAG_BACK_DOWN || loc == DMSTAG_BACK_DOWN_LEFT);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid dir %" PetscInt_FMT, dir);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetCoordinate_Internal(const PetscScalar **arr_coord, PetscInt idx, PetscInt slot, PetscInt x, PetscInt n, PetscScalar h_prev, PetscScalar h_next, PetscScalar *coord)
{
  PetscFunctionBegin;
  if (x <= idx && idx < x + n) *coord = arr_coord[idx][slot];
  /* Before the local grid start; extrapolate using uniform grid with h_prev */
  else if (idx < x) *coord = arr_coord[x][slot] - (x - idx) * h_prev;
  /* After the local grid end; extrapolate using uniform grid with h_next */
  else *coord = arr_coord[x + n - 1][slot] + (idx - (x + n - 1)) * h_next;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSolveLinearSystem_Internal(PetscInt n, PetscScalar A[], PetscScalar b[], PetscScalar x[])
{
  PetscInt    i, j, k;
  PetscScalar factor, sum;

  PetscFunctionBegin;
  /* Forward elimination */
  for (k = 0; k < n - 1; ++k) {
    PetscCheck(PetscAbs(A[k * n + k]) > FLUCAFD_ZERO_PIVOT_TOL, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Zero pivot encountered at row %" PetscInt_FMT " during forward elimination", k);
    for (i = k + 1; i < n; ++i) {
      factor = A[i * n + k] / A[k * n + k];
      for (j = k; j < n; ++j) A[i * n + j] -= factor * A[k * n + j];
      b[i] -= factor * b[k];
    }
  }
  /* Back substitution */
  for (i = n - 1; i >= 0; --i) {
    PetscCheck(PetscAbs(A[i * n + i]) > FLUCAFD_ZERO_PIVOT_TOL, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Zero pivot encountered at row %" PetscInt_FMT " during back substitution", i);
    sum = 0.;
    for (j = i + 1; j < n; ++j) sum += A[i * n + j] * x[j];
    x[i] = (b[i] - sum) / A[i * n + i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IsOffGrid_Private(FlucaFD fd, const DMStagStencil *col, PetscInt *off_dir, PetscBool *is_low, PetscBool *is_off_grid)
{
  PetscInt  idx, xg, ng, extrag, d;
  PetscBool use_face_coord, periodic;

  PetscFunctionBegin;
  *is_off_grid = PETSC_FALSE;

  for (d = 0; d < fd->dim; ++d) {
    switch (d) {
    case 0:
      idx = col->i;
      break;
    case 1:
      idx = col->j;
      break;
    case 2:
      idx = col->k;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported dim");
    }
    PetscCall(FlucaFDUseFaceCoordinate_Internal(col->loc, d, &use_face_coord));

    periodic = fd->bcs[2 * d].type == FLUCAFD_BC_PERIODIC;

    /* Local grid info */
    xg     = fd->x[d] - ((fd->is_first_rank[d] && !periodic) ? 0 : fd->stencil_width);
    ng     = fd->n[d] + ((fd->is_first_rank[d] && !periodic) ? 0 : fd->stencil_width) + ((fd->is_last_rank[d] && !periodic) ? 0 : fd->stencil_width);
    extrag = (fd->is_last_rank[d] && use_face_coord && !periodic) ? 1 : 0;

    if (idx < xg) {
      *is_off_grid = PETSC_TRUE;
      *off_dir     = d;
      *is_low      = PETSC_TRUE;
      break;
    } else if (idx >= xg + ng + extrag) {
      *is_off_grid = PETSC_TRUE;
      *off_dir     = d;
      *is_low      = PETSC_FALSE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetStencilSizeForOffGridPoint_Private(FlucaFD fd, const DMStagStencil *col, PetscInt dir, FlucaFDBoundaryConditionType bc_type, PetscInt *stencil_size)
{
  FlucaFDTermLink term;
  PetscInt        order, min_order;

  PetscFunctionBegin;
  min_order = PETSC_INT_MAX;
  for (term = fd->termlink; term; term = term->next)
    if (term->deriv_order[dir] != -1 && term->accu_order[dir] != PETSC_INT_MAX && term->input_loc == col->loc && term->input_c == col->c) {
      order     = term->deriv_order[dir] + term->accu_order[dir];
      min_order = PetscMin(order, min_order);
    }
  PetscCheck(min_order != PETSC_INT_MAX, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONG, "Cannot find a term from the given stencil point");

  switch (bc_type) {
  case FLUCAFD_BC_NONE:
  case FLUCAFD_BC_DIRICHLET:
  case FLUCAFD_BC_NEUMANN:
    *stencil_size = min_order;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported boundary condition type");
  }
  if (*stencil_size < 1) *stencil_size = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddStencilPoint_Private(PetscInt *ncols, DMStagStencil col[], PetscScalar v[], const DMStagStencil *new_col, PetscScalar new_v)
{
  PetscInt  c;
  PetscBool found;

  PetscFunctionBegin;
  found = PETSC_FALSE;
  for (c = 0; c < *ncols; ++c) {
    if (col[c].i == new_col->i && col[c].j == new_col->j && col[c].k == new_col->k && col[c].c == new_col->c && col[c].loc == new_col->loc) {
      v[c] += new_v;
      found = PETSC_TRUE;
      break;
    }
  }
  if (!found) {
    PetscCheck(*ncols < FLUCAFD_MAX_STENCIL_SIZE, PETSC_COMM_SELF, PETSC_ERR_SUP, "Resulting stencil is too large");
    col[*ncols] = *new_col;
    v[*ncols]   = new_v;
    ++(*ncols);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetBoundaryStencilLocation_Private(DMStagStencilLocation ref_loc, PetscInt dir, DMStagStencilLocation *bnd_loc)
{
  PetscFunctionBegin;
  // clang-format off
  switch (dir) {
  case 0:
    /* Add LEFT */
    switch (ref_loc) {
    case DMSTAG_ELEMENT:        *bnd_loc = DMSTAG_LEFT;           break;
    case DMSTAG_LEFT:           *bnd_loc = DMSTAG_LEFT;           break;
    case DMSTAG_DOWN:           *bnd_loc = DMSTAG_DOWN_LEFT;      break;
    case DMSTAG_BACK:           *bnd_loc = DMSTAG_BACK_LEFT;      break;
    case DMSTAG_DOWN_LEFT:      *bnd_loc = DMSTAG_DOWN_LEFT;      break;
    case DMSTAG_BACK_LEFT:      *bnd_loc = DMSTAG_BACK_LEFT;      break;
    case DMSTAG_BACK_DOWN:      *bnd_loc = DMSTAG_BACK_DOWN_LEFT; break;
    case DMSTAG_BACK_DOWN_LEFT: *bnd_loc = DMSTAG_BACK_DOWN_LEFT; break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported DMStagStencilLocation");
    }
    break;
  case 1:
    /* Add DOWN */
    switch (ref_loc) {
    case DMSTAG_ELEMENT:        *bnd_loc = DMSTAG_DOWN;           break;
    case DMSTAG_LEFT:           *bnd_loc = DMSTAG_DOWN_LEFT;      break;
    case DMSTAG_DOWN:           *bnd_loc = DMSTAG_DOWN;           break;
    case DMSTAG_BACK:           *bnd_loc = DMSTAG_BACK_DOWN;      break;
    case DMSTAG_DOWN_LEFT:      *bnd_loc = DMSTAG_DOWN_LEFT;      break;
    case DMSTAG_BACK_LEFT:      *bnd_loc = DMSTAG_BACK_DOWN_LEFT; break;
    case DMSTAG_BACK_DOWN:      *bnd_loc = DMSTAG_BACK_DOWN;      break;
    case DMSTAG_BACK_DOWN_LEFT: *bnd_loc = DMSTAG_BACK_DOWN_LEFT; break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported DMStagStencilLocation");
    }
    break;
  case 2:
    /* Add BACK */
    switch (ref_loc) {
    case DMSTAG_ELEMENT:        *bnd_loc = DMSTAG_BACK;           break;
    case DMSTAG_LEFT:           *bnd_loc = DMSTAG_BACK_LEFT;      break;
    case DMSTAG_DOWN:           *bnd_loc = DMSTAG_BACK_DOWN;      break;
    case DMSTAG_BACK:           *bnd_loc = DMSTAG_BACK;           break;
    case DMSTAG_DOWN_LEFT:      *bnd_loc = DMSTAG_BACK_DOWN_LEFT; break;
    case DMSTAG_BACK_LEFT:      *bnd_loc = DMSTAG_BACK_LEFT;      break;
    case DMSTAG_BACK_DOWN:      *bnd_loc = DMSTAG_BACK_DOWN;      break;
    case DMSTAG_BACK_DOWN_LEFT: *bnd_loc = DMSTAG_BACK_DOWN_LEFT; break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported DMStagStencilLocation");
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported direction");
  }
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDRemoveOffGridPoints_Internal(FlucaFD fd, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscInt       iter;
  const PetscInt max_iter = 100;

  PetscFunctionBegin;
  for (iter = 0; iter < max_iter; ++iter) {
    PetscInt  off_idx, off_dir, c;
    PetscBool is_low, is_off_grid;

    /* Find an off-grid point */
    off_idx = -1;
    for (c = 0; c < *ncols; ++c) {
      PetscCall(IsOffGrid_Private(fd, &col[c], &off_dir, &is_low, &is_off_grid));
      if (is_off_grid) {
        off_idx = c;
        break;
      }
    }
    if (off_idx < 0) break;

    /* Process this off-grid point */
    {
      const DMStagStencil          off_col = col[off_idx];
      const PetscScalar            off_v   = v[off_idx];
      FlucaFDBoundaryConditionType bc_type;
      const PetscScalar          **arr_coord;
      PetscBool                    periodic, use_face_coord;
      PetscInt                     coord_slot, stencil_size, xg, ng, extrag, first_face_idx, last_face_idx, off_grid_idx, start_idx, bnd_idx;
      PetscScalar                  h_prev, h_next, off_coord, bnd_coord, a_off;
      PetscScalar                  extrap_coords[FLUCAFD_MAX_STENCIL_SIZE];
      PetscScalar                  extrap_coeffs[FLUCAFD_MAX_STENCIL_SIZE];
      PetscScalar                  A[FLUCAFD_MAX_STENCIL_SIZE * FLUCAFD_MAX_STENCIL_SIZE];
      PetscScalar                  b[FLUCAFD_MAX_STENCIL_SIZE];
      DMStagStencil                new_col;
      PetscInt                     n, r;

      periodic = fd->bcs[2 * off_dir].type == FLUCAFD_BC_PERIODIC;

      if (is_low && fd->is_first_rank[off_dir] && !periodic) bc_type = fd->bcs[2 * off_dir].type;
      else if (!is_low && fd->is_last_rank[off_dir] && !periodic) bc_type = fd->bcs[2 * off_dir + 1].type;
      else bc_type = FLUCAFD_BC_NONE;

      arr_coord = fd->arr_coord[off_dir];
      PetscCall(FlucaFDUseFaceCoordinate_Internal(off_col.loc, off_dir, &use_face_coord));
      coord_slot = use_face_coord ? fd->slot_coord_prev : fd->slot_coord_elem;

      PetscCall(GetStencilSizeForOffGridPoint_Private(fd, &off_col, off_dir, bc_type, &stencil_size));
      PetscCheck(stencil_size <= FLUCAFD_MAX_STENCIL_SIZE, PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Stencil size %" PetscInt_FMT " exceeds maximum %d", stencil_size, FLUCAFD_MAX_STENCIL_SIZE);

      /* Local grid info */
      xg     = fd->x[off_dir] - ((fd->is_first_rank[off_dir] && !periodic) ? 0 : fd->stencil_width);
      ng     = fd->n[off_dir] + ((fd->is_first_rank[off_dir] && !periodic) ? 0 : fd->stencil_width) + ((fd->is_last_rank[off_dir] && !periodic) ? 0 : fd->stencil_width);
      extrag = (fd->is_last_rank[off_dir] && use_face_coord && !periodic) ? 1 : 0;

      first_face_idx = xg;
      last_face_idx  = xg + ng - ((fd->is_last_rank[off_dir] && !periodic) ? 0 : 1);
      h_prev         = arr_coord[first_face_idx + 1][fd->slot_coord_prev] - arr_coord[first_face_idx][fd->slot_coord_prev];
      h_next         = arr_coord[last_face_idx][fd->slot_coord_prev] - arr_coord[last_face_idx - 1][fd->slot_coord_prev];
      switch (off_dir) {
      case 0:
        off_grid_idx = off_col.i;
        break;
      case 1:
        off_grid_idx = off_col.j;
        break;
      case 2:
        off_grid_idx = off_col.k;
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported direction");
      }
      PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, off_grid_idx, coord_slot, xg, ng + extrag, h_prev, h_next, &off_coord));

      /* Remove the off-grid point from stencil */
      for (c = off_idx; c < *ncols - 1; ++c) {
        col[c] = col[c + 1];
        v[c]   = v[c + 1];
      }
      --(*ncols);

      /* Handle based on BC type */
      switch (bc_type) {
      case FLUCAFD_BC_NONE:
        /* Extrapolate the value on the off-grid point using n consecutive on-grid points */
        start_idx = is_low ? xg : (xg + ng + extrag - stencil_size);
        for (n = 0; n < stencil_size; ++n) PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, start_idx + n, coord_slot, xg, ng + extrag, h_prev, h_next, &extrap_coords[n]));

        /* Build and solve Vandermonde matrix */
        for (r = 0; r < stencil_size; ++r) {
          for (n = 0; n < stencil_size; ++n) A[r * stencil_size + n] = PetscPowScalarInt(extrap_coords[n] - off_coord, r);
          b[r] = (r == 0) ? 1. : 0.;
        }
        PetscCall(FlucaFDSolveLinearSystem_Internal(stencil_size, A, b, extrap_coeffs));

        /* Add on-grid points to stencil */
        for (n = 0; n < stencil_size; ++n) {
          new_col = off_col;
          switch (off_dir) {
          case 0:
            new_col.i = start_idx + n;
            break;
          case 1:
            new_col.j = start_idx + n;
            break;
          case 2:
            new_col.k = start_idx + n;
            break;
          }
          PetscCall(AddStencilPoint_Private(ncols, col, v, &new_col, off_v * extrap_coeffs[n]));
        }
        break;

      case FLUCAFD_BC_DIRICHLET:
        /* Extrapolate the value on the off-grid point using the boundary value and (n-1) consecutive on-grid points */
        start_idx = is_low ? xg : (xg + ng + extrag - (stencil_size - 1));
        if (use_face_coord) {
          /* Remove duplicate */
          if (is_low) ++start_idx;
          else --start_idx;
        }
        bnd_idx = is_low ? 0 : fd->N[off_dir];
        PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, bnd_idx, fd->slot_coord_prev, xg, ng + extrag, h_prev, h_next, &bnd_coord));
        extrap_coords[0] = bnd_coord;
        for (n = 0; n < stencil_size - 1; ++n) PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, start_idx + n, coord_slot, xg, ng + extrag, h_prev, h_next, &extrap_coords[n + 1]));

        /* Build and solve Vandermonde matrix */
        for (r = 0; r < stencil_size; ++r) {
          for (n = 0; n < stencil_size; ++n) A[r * stencil_size + n] = PetscPowScalarInt(extrap_coords[n] - off_coord, r);
          b[r] = (r == 0) ? 1. : 0.;
        }
        PetscCall(FlucaFDSolveLinearSystem_Internal(stencil_size, A, b, extrap_coeffs));

        /* Add the boundary point to stencil */
        new_col.i = (off_dir == 0) ? bnd_idx : off_col.i;
        new_col.j = (off_dir == 1) ? bnd_idx : off_col.j;
        new_col.k = (off_dir == 2) ? bnd_idx : off_col.k;
        PetscCall(GetBoundaryStencilLocation_Private(off_col.loc, off_dir, &new_col.loc));
        new_col.c = -(2 * off_dir + (is_low ? 1 : 2)); /* Boundary value marker: -1=left, -2=right, -3=down, -4=up, -5=back, -6=front */
        PetscCall(AddStencilPoint_Private(ncols, col, v, &new_col, off_v * extrap_coeffs[0]));

        /* Add on-grid points to stencil */
        for (n = 0; n < stencil_size - 1; ++n) {
          new_col = off_col;
          switch (off_dir) {
          case 0:
            new_col.i = start_idx + n;
            break;
          case 1:
            new_col.j = start_idx + n;
            break;
          case 2:
            new_col.k = start_idx + n;
            break;
          }
          PetscCall(AddStencilPoint_Private(ncols, col, v, &new_col, off_v * extrap_coeffs[n + 1]));
        }
        break;

      case FLUCAFD_BC_NEUMANN:
        /* Build FD stencil for first derivative on the boundary using n consecutive on-grid points */
        start_idx = is_low ? xg : (xg + ng + extrag - (stencil_size - 1));
        bnd_idx   = is_low ? 0 : fd->N[off_dir];
        PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, bnd_idx, fd->slot_coord_prev, xg, ng + extrag, h_prev, h_next, &bnd_coord));
        extrap_coords[0] = off_coord;
        for (n = 0; n < stencil_size - 1; ++n) PetscCall(FlucaFDGetCoordinate_Internal(arr_coord, start_idx + n, coord_slot, xg, ng + extrag, h_prev, h_next, &extrap_coords[n + 1]));

        /* Build Vandermonde matrix for first derivative on the boundary */
        for (r = 0; r < stencil_size; ++r) {
          for (n = 0; n < stencil_size; ++n) { A[r * stencil_size + n] = PetscPowScalarInt(extrap_coords[n] - bnd_coord, r); }
          b[r] = (r == 1) ? 1.0 : 0.0; /* First derivative */
        }
        /* Solve for FD coefficients: v'[bnd] = a_off*v[off] + a[0]*v[0] + ... */
        PetscCall(FlucaFDSolveLinearSystem_Internal(stencil_size, A, b, extrap_coeffs));
        a_off = extrap_coeffs[0];
        PetscCheck(PetscAbs(a_off) >= FLUCAFD_COEFF_ATOL, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Neumann BC coefficient for off-grid point is too small");

        /* v[off] = (v'[bnd] - a[0]*v[0] - ... ) / a_off */

        /* Add the boundary point to stencil */
        new_col.i = (off_dir == 0) ? bnd_idx : off_col.i;
        new_col.j = (off_dir == 1) ? bnd_idx : off_col.j;
        new_col.k = (off_dir == 2) ? bnd_idx : off_col.k;
        PetscCall(GetBoundaryStencilLocation_Private(off_col.loc, off_dir, &new_col.loc));
        new_col.c = -(2 * off_dir + (is_low ? 1 : 2)); /* Boundary value marker: -1=left, -2=right, -3=down, -4=up, -5=back, -6=front */
        PetscCall(AddStencilPoint_Private(ncols, col, v, &new_col, off_v / a_off));

        /* Add on-grid points to stencil */
        for (n = 0; n < stencil_size - 1; ++n) {
          new_col = off_col;
          switch (off_dir) {
          case 0:
            new_col.i = start_idx + n;
            break;
          case 1:
            new_col.j = start_idx + n;
            break;
          case 2:
            new_col.k = start_idx + n;
            break;
          }
          PetscCall(AddStencilPoint_Private(ncols, col, v, &new_col, -off_v * extrap_coeffs[n + 1] / a_off));
        }
        break;

      default:
        SETERRQ(PetscObjectComm((PetscObject)fd), PETSC_ERR_SUP, "Unsupported boundary condition type");
      }
    }
  }

  PetscCheck(iter < max_iter, PetscObjectComm((PetscObject)fd), PETSC_ERR_CONV_FAILED, "Failed to remove all off-grid points after %" PetscInt_FMT " iterations", max_iter);
  PetscCall(FlucaFDRemoveZeroStencilPoints_Internal(ncols, col, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDRemoveZeroStencilPoints_Internal(PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscScalar       v_abssum;
  PetscInt          ncols_new, c;
  PetscBool         remove;
  const PetscScalar atol = FLUCAFD_COEFF_ATOL, rtol = FLUCAFD_COEFF_RTOL;

  PetscFunctionBegin;
  v_abssum = 0.;
  for (c = 0; c < *ncols; ++c) v_abssum += PetscAbs(v[c]);
  ncols_new = 0;
  for (c = 0; c < *ncols; ++c) {
    remove = PetscAbs(v[c]) < atol || PetscAbs(v[c] / v_abssum) < rtol;
    if (!remove) {
      col[ncols_new] = col[c];
      v[ncols_new]   = v[c];
      ++ncols_new;
    }
  }
  *ncols = ncols_new;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkCreate_Internal(FlucaFDTermLink *term)
{
  FlucaFDTermLink t;
  PetscInt        d;

  PetscFunctionBegin;
  PetscCall(PetscNew(&t));
  for (d = 0; d < FLUCAFD_MAX_DIM; ++d) {
    t->deriv_order[d] = -1;
    t->accu_order[d]  = PETSC_INT_MAX;
  }
  t->input_loc = DMSTAG_ELEMENT;
  t->input_c   = 0;
  t->next      = NULL;
  *term        = t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkDuplicate_Internal(FlucaFDTermLink src, FlucaFDTermLink *dst)
{
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(FlucaFDTermLinkCreate_Internal(dst));
  for (d = 0; d < FLUCAFD_MAX_DIM; ++d) {
    (*dst)->deriv_order[d] = src->deriv_order[d];
    (*dst)->accu_order[d]  = src->accu_order[d];
  }
  (*dst)->input_loc = src->input_loc;
  (*dst)->input_c   = src->input_c;
  (*dst)->next      = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkAppend_Internal(FlucaFDTermLink *link, FlucaFDTermLink term)
{
  FlucaFDTermLink curr;

  PetscFunctionBegin;
  if (!*link) {
    *link = term;
  } else {
    curr = *link;
    while (curr->next) curr = curr->next;
    curr->next = term;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDTermInfoEqual_Private(FlucaFDTermLink a, FlucaFDTermLink b, PetscBool *equal)
{
  PetscInt d;

  PetscFunctionBegin;
  *equal = PETSC_TRUE;
  for (d = 0; d < FLUCAFD_MAX_DIM; ++d)
    if (a->deriv_order[d] != b->deriv_order[d] || a->accu_order[d] != b->accu_order[d]) *equal = PETSC_FALSE;
  if (a->input_loc != b->input_loc || a->input_c != b->input_c) *equal = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkFind_Internal(FlucaFDTermLink link, FlucaFDTermLink term, PetscBool *found)
{
  FlucaFDTermLink t;
  PetscBool       equal;

  PetscFunctionBegin;
  *found = PETSC_FALSE;
  for (t = link; t; t = t->next) {
    PetscCall(FlucaFDTermInfoEqual_Private(t, term, &equal));
    if (equal) {
      *found = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkDestroy_Internal(FlucaFDTermLink *link)
{
  FlucaFDTermLink curr, next;

  PetscFunctionBegin;
  curr = *link;
  while (curr) {
    next = curr->next;
    PetscCall(PetscFree(curr));
    curr = next;
  }
  *link = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
