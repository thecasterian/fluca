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
  PetscCall(FlucaFDRemoveOffGridPoints_Internal(fd, i, j, k, ncols, col, v));
  PetscCall(FlucaFDRemoveZeroStencilPoints_Internal(ncols, col, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDApply(FlucaFD fd, DM input_dm, DM output_dm, Mat mat)
{
  PetscInt              x, y, z, m, n, p;
  PetscInt              i, j, k;
  DMStagStencil         row;
  DMStagStencil         col[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar           v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt              ncols;
  PetscInt              ir;
  PetscInt              ic[FLUCAFD_MAX_STENCIL_SIZE];
  DMStagStencilLocation stag_loc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 2);
  PetscCheck(fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FlucaFD not setup");

  PetscCall(FlucaFDStencilLocationToDMStagStencilLocation_Internal(fd->output_loc, &stag_loc));
  PetscCall(DMStagGetCorners(output_dm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(MatZeroEntries(mat));

  /* Loop over all local grid points */
  for (k = z; k < z + p; k++) {
    for (j = y; j < y + n; j++) {
      for (i = x; i < x + m; i++) {
        row.i   = i;
        row.j   = j;
        row.k   = k;
        row.c   = fd->output_c;
        row.loc = stag_loc;
        PetscCall(FlucaFDGetStencil(fd, i, j, k, &ncols, col, v));

        PetscCall(DMStagStencilToIndexLocal(output_dm, fd->dim, 1, &row, &ir));
        PetscCall(DMStagStencilToIndexLocal(input_dm, fd->dim, ncols, col, ic));
        PetscCall(MatSetValuesLocal(mat, 1, &ir, ncols, ic, v, ADD_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
