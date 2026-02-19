#include <fluca/private/flucafdimpl.h>

static PetscErrorCode GetOutputLoopRange_Private(FlucaFD fd, DM output_dm, PetscInt *i_start, PetscInt *j_start, PetscInt *k_start, PetscInt *i_end, PetscInt *j_end, PetscInt *k_end)
{
  PetscInt  xs, ys, zs, xm, ym, zm, nExtrax, nExtray, nExtraz;
  PetscBool use_face_x, use_face_y, use_face_z;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(output_dm, &xs, &ys, &zs, &xm, &ym, &zm, &nExtrax, &nExtray, &nExtraz));
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, 0, &use_face_x));
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, 1, &use_face_y));
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, 2, &use_face_z));
  *i_start = xs;
  *j_start = (fd->dim >= 2) ? ys : 0;
  *k_start = (fd->dim >= 3) ? zs : 0;
  *i_end   = xs + xm + (use_face_x ? nExtrax : 0);
  *j_end   = (fd->dim >= 2) ? (ys + ym + (use_face_y ? nExtray : 0)) : 1;
  *k_end   = (fd->dim >= 3) ? (zs + zm + (use_face_z ? nExtraz : 0)) : 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetStencilRaw(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscAssertPointer(ncols, 5);
  PetscAssertPointer(col, 6);
  PetscAssertPointer(v, 7);
  PetscUseTypeMethod(fd, getstencilraw, i, j, k, ncols, col, v);
  PetscCall(FlucaFDRemoveZeroStencilPoints_Internal(ncols, col, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetStencil(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscAssertPointer(ncols, 5);
  PetscAssertPointer(col, 6);
  PetscAssertPointer(v, 7);
  PetscCall(FlucaFDGetStencilRaw(fd, i, j, k, ncols, col, v));
  PetscCall(FlucaFDRemoveOffGridPoints_Internal(fd, ncols, col, v));
  PetscCall(FlucaFDRemoveZeroStencilPoints_Internal(ncols, col, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDApply(FlucaFD fd, DM input_dm, DM output_dm, Vec x, Vec y)
{
  Vec                x_local, y_local;
  const PetscScalar *x_arr;
  PetscScalar       *y_arr;
  PetscInt           i_start, j_start, k_start, i_end, j_end, k_end;
  PetscInt           i, j, k, c;
  DMStagStencil      row;
  DMStagStencil      col[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar        v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt           ncols;
  PetscInt           ir, idx, bnd_idx;
  PetscScalar        result;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecificType(input_dm, DM_CLASSID, 2, DMSTAG);
  PetscValidHeaderSpecificType(output_dm, DM_CLASSID, 3, DMSTAG);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 5);
  PetscCheck(fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FlucaFD not setup");

  /* Scatter input to local vector (fills ghost values) */
  PetscCall(DMGetLocalVector(input_dm, &x_local));
  PetscCall(DMGlobalToLocal(input_dm, x, INSERT_VALUES, x_local));
  PetscCall(VecGetArrayRead(x_local, &x_arr));

  /* Prepare local output vector */
  PetscCall(DMGetLocalVector(output_dm, &y_local));
  PetscCall(VecZeroEntries(y_local));
  PetscCall(VecGetArray(y_local, &y_arr));

  PetscCall(GetOutputLoopRange_Private(fd, output_dm, &i_start, &j_start, &k_start, &i_end, &j_end, &k_end));

  for (k = k_start; k < k_end; k++) {
    for (j = j_start; j < j_end; j++) {
      for (i = i_start; i < i_end; i++) {
        result = 0.;

        row.i   = i;
        row.j   = j;
        row.k   = k;
        row.c   = fd->output_c;
        row.loc = fd->output_loc;
        PetscCall(FlucaFDGetStencil(fd, i, j, k, &ncols, col, v));

        for (c = 0; c < ncols; ++c) {
          if (col[c].c >= 0) {
            /* Interior point: read from input vector */
            PetscCall(DMStagStencilToIndexLocal(input_dm, fd->dim, 1, &col[c], &idx));
            result += v[c] * x_arr[idx];
          } else if (col[c].c == FLUCAFD_CONSTANT) {
            /* Constant term */
            result += v[c];
          } else {
            /* Boundary point: coefficient * boundary value */
            bnd_idx = -col[c].c - 1;
            PetscCheck(bnd_idx >= 0 && bnd_idx < 2 * FLUCAFD_MAX_DIM, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary marker %" PetscInt_FMT " in stencil", col[c].c);
            result += v[c] * fd->bcs[bnd_idx].value;
          }
        }

        PetscCall(DMStagStencilToIndexLocal(output_dm, fd->dim, 1, &row, &ir));
        y_arr[ir] = result;
      }
    }
  }

  PetscCall(VecRestoreArray(y_local, &y_arr));
  PetscCall(VecRestoreArrayRead(x_local, &x_arr));
  PetscCall(DMLocalToGlobal(output_dm, y_local, INSERT_VALUES, y));
  PetscCall(DMRestoreLocalVector(output_dm, &y_local));
  PetscCall(DMRestoreLocalVector(input_dm, &x_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* FlucaFDGetOperator - Assembles only the interior stencil coefficients into op.
   Boundary and constant terms are omitted because they do not depend on the
   solution vector (they contribute to the residual, not the Jacobian).
   Use FlucaFDApply() for full operator application including boundary terms. */
PetscErrorCode FlucaFDGetOperator(FlucaFD fd, DM input_dm, DM output_dm, Mat op)
{
  PetscInt      i_start, j_start, k_start, i_end, j_end, k_end;
  PetscInt      i, j, k, c;
  DMStagStencil row;
  DMStagStencil col[FLUCAFD_MAX_STENCIL_SIZE];
  DMStagStencil mat_col[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar   v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar   mat_v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt      ncols, mat_ncols;
  PetscInt      ir;
  PetscInt      ic[FLUCAFD_MAX_STENCIL_SIZE];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecificType(input_dm, DM_CLASSID, 2, DMSTAG);
  PetscValidHeaderSpecificType(output_dm, DM_CLASSID, 3, DMSTAG);
  PetscValidHeaderSpecific(op, MAT_CLASSID, 4);
  PetscCheck(fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FlucaFD not setup");

  PetscCall(GetOutputLoopRange_Private(fd, output_dm, &i_start, &j_start, &k_start, &i_end, &j_end, &k_end));

  for (k = k_start; k < k_end; k++) {
    for (j = j_start; j < j_end; j++) {
      for (i = i_start; i < i_end; i++) {
        row.i   = i;
        row.j   = j;
        row.k   = k;
        row.c   = fd->output_c;
        row.loc = fd->output_loc;
        PetscCall(FlucaFDGetStencil(fd, i, j, k, &ncols, col, v));

        /* Collect only interior stencil points (skip boundary and constant terms) */
        mat_ncols = 0;
        for (c = 0; c < ncols; ++c) {
          if (col[c].c >= 0) {
            mat_col[mat_ncols] = col[c];
            mat_v[mat_ncols]   = v[c];
            ++mat_ncols;
          }
        }

        /* Set matrix values for interior points */
        if (mat_ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(output_dm, fd->dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(input_dm, fd->dim, mat_ncols, mat_col, ic));
          PetscCall(MatSetValuesLocal(op, 1, &ir, mat_ncols, ic, mat_v, ADD_VALUES));
        }
      }
    }
  }
  /* The user must call MatAssemblyBegin/End */
  PetscFunctionReturn(PETSC_SUCCESS);
}
