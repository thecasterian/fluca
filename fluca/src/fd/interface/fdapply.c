#include <fluca/private/flucafdimpl.h>

// clang-format off
#define FlucaFDStencilPointToStencil(pt, st) \
  do { \
    (st)->loc = (pt)->loc; \
    (st)->i   = (pt)->i; \
    (st)->j   = (pt)->j; \
    (st)->k   = (pt)->k; \
    (st)->c   = (pt)->c; \
  } while (0)
// clang-format on

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

/* With pattern true, every scale, constant or vector, is a deferred ref, so points merge only under the same scales
   and the zeros removed here come only from fixed data: mesh weights */
PetscErrorCode FlucaFDGetStencilRaw(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscBool pattern, PetscInt *npoints, FlucaFDStencilPoint points[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscAssertPointer(npoints, 6);
  PetscAssertPointer(points, 7);
  PetscUseTypeMethod(fd, getstencilraw, i, j, k, pattern, npoints, points);
  PetscCall(FlucaFDRemoveZeroStencilPoints_Internal(pattern, npoints, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* With pattern true, scale and TVD refs are left unevaluated and coefficients that may change at run time keep their
   points (both TVD upwind candidates stay grid points): only zeros that come from fixed data are removed */
static PetscErrorCode GetStencil_Private(FlucaFD fd, PetscReal t, PetscInt i, PetscInt j, PetscInt k, PetscBool pattern, PetscInt *npoints, FlucaFDStencilPoint points[])
{
  FlucaFDStencilPoint merged[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt            merged_npoints, c;
  const FlucaFDBoundaryCondition(*bcs)[2 * FLUCAFD_MAX_DIM];

  PetscFunctionBegin;
  PetscCall(FlucaFDGetStencilRaw(fd, i, j, k, pattern, npoints, points));
  PetscCall(FlucaFDGetBoundaryConditions_Internal(fd, i, j, k, &bcs));
  PetscCall(FlucaFDRemoveOffGridPoints_Internal(fd, bcs, pattern, npoints, points));
  if (pattern) {
    for (c = 0; c < *npoints; c++) {
      points[c].nscales = 0;
      points[c].ntvds   = 0;
    }
  } else {
    PetscCall(FlucaFDResolveScaleRefs_Internal(*npoints, points));
    PetscCall(FlucaFDResolveTVDRefs_Internal(t, *npoints, points));
  }
  /* Re-merge points that are now identical after ref resolution */
  merged_npoints = 0;
  for (c = 0; c < *npoints; c++) PetscCall(FlucaFDAddStencilPoint_Internal(&points[c], &merged_npoints, merged));
  *npoints = merged_npoints;
  PetscCall(PetscArraycpy(points, merged, merged_npoints));
  if (!pattern) PetscCall(FlucaFDRemoveZeroStencilPoints_Internal(PETSC_FALSE, npoints, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetStencil(FlucaFD fd, PetscReal t, PetscInt i, PetscInt j, PetscInt k, PetscInt *npoints, FlucaFDStencilPoint points[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscAssertPointer(npoints, 6);
  PetscAssertPointer(points, 7);
  PetscCall(GetStencil_Private(fd, t, i, j, k, PETSC_FALSE, npoints, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDApply_Private(FlucaFD fd, PetscReal t, DM input_dm, DM output_dm, Vec x, Vec y, PetscBool use_dot_bc)
{
  Vec                 x_local, y_local;
  const PetscScalar  *x_arr;
  PetscScalar        *y_arr;
  PetscInt            i_start, j_start, k_start, i_end, j_end, k_end;
  PetscInt            i, j, k, c;
  DMStagStencil       row, stencil;
  FlucaFDStencilPoint points[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt            npoints;
  PetscInt            ir, idx;
  PetscScalar         result, bc_val;
  const FlucaFDBoundaryCondition(*bcs)[2 * FLUCAFD_MAX_DIM];

  PetscFunctionBegin;
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
        PetscCall(FlucaFDGetStencil(fd, t, i, j, k, &npoints, points));
        PetscCall(FlucaFDGetBoundaryConditions_Internal(fd, i, j, k, &bcs));

        for (c = 0; c < npoints; ++c) {
          switch (points[c].type) {
          case FLUCAFD_STENCIL_GRID:
            FlucaFDStencilPointToStencil(&points[c], &stencil);
            PetscCall(DMStagStencilToIndexLocal(input_dm, fd->dim, 1, &stencil, &idx));
            result += points[c].v * x_arr[idx];
            break;
          case FLUCAFD_STENCIL_CONSTANT:
            result += points[c].v;
            break;
          case FLUCAFD_STENCIL_BOUNDARY:
            if (use_dot_bc) {
              PetscCall(FlucaFDEvaluateBCValueDot_Internal(fd, bcs, t, points[c].c, points[c].boundary_face, points[c].i, points[c].j, points[c].k, &bc_val));
            } else {
              PetscCall(FlucaFDEvaluateBCValue_Internal(fd, bcs, t, points[c].c, points[c].boundary_face, points[c].i, points[c].j, points[c].k, &bc_val));
            }
            result += points[c].v * bc_val;
            break;
          case FLUCAFD_STENCIL_IB:
            PetscCall(FlucaFDEvaluateIBValue_Internal(fd, t, &points[c], use_dot_bc, &bc_val));
            result += points[c].v * bc_val;
            break;
          }
        }

        PetscCall(DMStagStencilToIndexLocal(output_dm, fd->dim, 1, &row, &ir));
        y_arr[ir] = result;
      }
    }
  }

  PetscCall(VecRestoreArray(y_local, &y_arr));
  PetscCall(VecRestoreArrayRead(x_local, &x_arr));
  PetscCall(DMLocalToGlobal(output_dm, y_local, ADD_VALUES, y));
  PetscCall(DMRestoreLocalVector(output_dm, &y_local));
  PetscCall(DMRestoreLocalVector(input_dm, &x_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDApply(FlucaFD fd, PetscReal t, DM input_dm, DM output_dm, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecificType(input_dm, DM_CLASSID, 3, DMSTAG);
  PetscValidHeaderSpecificType(output_dm, DM_CLASSID, 4, DMSTAG);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscCheck(fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FlucaFD not setup");
  PetscCall(FlucaFDApply_Private(fd, t, input_dm, output_dm, x, y, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDApplyDot(FlucaFD fd, PetscReal t, DM input_dm, DM output_dm, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecificType(input_dm, DM_CLASSID, 3, DMSTAG);
  PetscValidHeaderSpecificType(output_dm, DM_CLASSID, 4, DMSTAG);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscCheck(fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FlucaFD not setup");
  PetscCall(FlucaFDApply_Private(fd, t, input_dm, output_dm, x, y, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetOperator_Private(FlucaFD fd, DM input_dm, DM output_dm, PetscBool pattern, Mat op)
{
  PetscInt            i_start, j_start, k_start, i_end, j_end, k_end;
  PetscInt            i, j, k, c;
  DMStagStencil       row;
  FlucaFDStencilPoint points[FLUCAFD_MAX_STENCIL_SIZE];
  DMStagStencil       mat_col[FLUCAFD_MAX_STENCIL_SIZE];
  PetscScalar         mat_v[FLUCAFD_MAX_STENCIL_SIZE];
  PetscInt            npoints, mat_ncols;
  PetscInt            ir;
  PetscInt            ic[FLUCAFD_MAX_STENCIL_SIZE];

  PetscFunctionBegin;
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
        PetscCall(GetStencil_Private(fd, 0., i, j, k, pattern, &npoints, points));

        /* Collect only grid stencil points (skip boundary and constant terms) */
        mat_ncols = 0;
        for (c = 0; c < npoints; ++c) {
          if (points[c].type == FLUCAFD_STENCIL_GRID) {
            FlucaFDStencilPointToStencil(&points[c], &mat_col[mat_ncols]);
            mat_v[mat_ncols] = points[c].v;
            ++mat_ncols;
          }
        }

        /* Set matrix values for grid points */
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

PetscErrorCode FlucaFDGetOperator(FlucaFD fd, DM input_dm, DM output_dm, Mat op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecificType(input_dm, DM_CLASSID, 2, DMSTAG);
  PetscValidHeaderSpecificType(output_dm, DM_CLASSID, 3, DMSTAG);
  PetscValidHeaderSpecific(op, MAT_CLASSID, 4);
  PetscCall(GetOperator_Private(fd, input_dm, output_dm, PETSC_FALSE, op));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetOperatorPattern(FlucaFD fd, DM input_dm, DM output_dm, Mat op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscValidHeaderSpecificType(input_dm, DM_CLASSID, 2, DMSTAG);
  PetscValidHeaderSpecificType(output_dm, DM_CLASSID, 3, DMSTAG);
  PetscValidHeaderSpecific(op, MAT_CLASSID, 4);
  PetscCall(GetOperator_Private(fd, input_dm, output_dm, PETSC_TRUE, op));
  PetscFunctionReturn(PETSC_SUCCESS);
}
