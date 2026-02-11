#include <fluca/private/flucafdimpl.h>

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

PetscErrorCode FlucaFDApply(FlucaFD fd, DM input_dm, DM output_dm, Mat op, Vec vbc)
{
  PetscInt      x, y, z, m, n, p, nExtrax, nExtray, nExtraz;
  PetscBool     use_face_x, use_face_y, use_face_z;
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
  PetscValidHeaderSpecific(vbc, VEC_CLASSID, 5);
  PetscCheck(fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FlucaFD not setup");

  PetscCall(DMStagGetCorners(output_dm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz));

  /* Compute loop ranges based on output stencil location */
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, 0, &use_face_x));
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, 1, &use_face_y));
  PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, 2, &use_face_z));
  i_start = x;
  j_start = (fd->dim >= 2) ? y : 0;
  k_start = (fd->dim >= 3) ? z : 0;
  i_end   = x + m + (use_face_x ? nExtrax : 0);
  j_end   = (fd->dim >= 2) ? (y + n + (use_face_y ? nExtray : 0)) : 1;
  k_end   = (fd->dim >= 3) ? (z + p + (use_face_z ? nExtraz : 0)) : 1;

  for (k = k_start; k < k_end; k++) {
    for (j = j_start; j < j_end; j++) {
      for (i = i_start; i < i_end; i++) {
        row.i   = i;
        row.j   = j;
        row.k   = k;
        row.c   = fd->output_c;
        row.loc = fd->output_loc;
        PetscCall(FlucaFDGetStencil(fd, i, j, k, &ncols, col, v));

        /* Separate boundary points, constant terms, and matrix points */
        mat_ncols = 0;
        for (c = 0; c < ncols; ++c) {
          if (col[c].c < 0) {
            if (col[c].c == FLUCAFD_CONSTANT) {
              /* Constant term: coefficient goes directly to vbc vector */
              PetscCall(DMStagVecSetValuesStencil(output_dm, vbc, 1, &row, &v[c], ADD_VALUES));
            } else {
              /* Boundary point: coefficient * boundary value -> vbc vector */
              PetscInt    bnd_idx;
              PetscScalar bnd_contribution;

              bnd_idx = -col[c].c - 1; /* Convert boundary marker to index: -1 -> 0, -2 -> 1, etc. */
              PetscCheck(bnd_idx >= 0 && bnd_idx < 2 * FLUCAFD_MAX_DIM, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary marker %" PetscInt_FMT " in stencil", col[c].c);
              bnd_contribution = v[c] * fd->bcs[bnd_idx].value;
              PetscCall(DMStagVecSetValuesStencil(output_dm, vbc, 1, &row, &bnd_contribution, ADD_VALUES));
            }
          } else {
            /* Matrix point: add to mat_col/mat_v arrays */
            mat_col[mat_ncols] = col[c];
            mat_v[mat_ncols]   = v[c];
            ++mat_ncols;
          }
        }

        /* Set matrix values for non-boundary points */
        if (mat_ncols > 0) {
          PetscCall(DMStagStencilToIndexLocal(output_dm, fd->dim, 1, &row, &ir));
          PetscCall(DMStagStencilToIndexLocal(input_dm, fd->dim, mat_ncols, mat_col, ic));
          PetscCall(MatSetValuesLocal(op, 1, &ir, mat_ncols, ic, mat_v, ADD_VALUES));
        }
      }
    }
  }
  /* The user must call MatAssemblyBegin/End and VecAssemblyBegin/End */
  PetscFunctionReturn(PETSC_SUCCESS);
}
