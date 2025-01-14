#include <fluca/private/flucaviewer_cgns.h>
#include <fluca/private/sol_fsm.h>
#include <pcgnslib.h>
#include <petscdmstag.h>

static PetscErrorCode GetCGNSDataType_Private(PetscDataType petsc_dtype, CGNS_ENUMT(DataType_t) *cgns_dtype)
{
  PetscFunctionBegin;
  switch (petsc_dtype) {
  case PETSC_DOUBLE:
    *cgns_dtype = CGNS_ENUMV(RealDouble);
    break;
  case PETSC_FLOAT:
    *cgns_dtype = CGNS_ENUMV(RealSingle);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported data type");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetLocalEntries2d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt d, PetscScalar *e)
{
  PetscInt       x, y, m, n, nExtrax, nExtray;
  PetscBool      isLastRankx, isLastRanky;
  PetscScalar ***arr;
  PetscInt       iloc, i, j, cnt = 0;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRankx, &isLastRanky, NULL));
  nExtrax = (loc == DMSTAG_LEFT && isLastRankx) ? 1 : 0;
  nExtray = (loc == DMSTAG_DOWN && isLastRanky) ? 1 : 0;
  PetscCall(DMStagVecGetArrayRead(dm, v, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, d, &iloc));
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m + nExtrax; ++i) e[cnt++] = arr[j][i][iloc];
  PetscCall(DMStagVecRestoreArrayRead(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetLocalEntries3d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt d, PetscScalar *e)
{
  PetscInt        x, y, z, m, n, p, nExtrax, nExtray, nExtraz;
  PetscBool       isLastRankx, isLastRanky, isLastRankz;
  PetscScalar ****arr;
  PetscInt        iloc, i, j, k, cnt = 0;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRankx, &isLastRanky, &isLastRankz));
  nExtrax = (loc == DMSTAG_LEFT && isLastRankx) ? 1 : 0;
  nExtray = (loc == DMSTAG_DOWN && isLastRanky) ? 1 : 0;
  nExtraz = (loc == DMSTAG_BACK && isLastRankz) ? 1 : 0;
  PetscCall(DMStagVecGetArrayRead(dm, v, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, d, &iloc));
  for (k = z; k < z + p + nExtraz; ++k)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m + nExtrax; ++i) e[cnt++] = arr[k][j][i][iloc];
  PetscCall(DMStagVecRestoreArrayRead(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteCellCenteredSolution_Private(DM dm, Vec v, int file_num, int base, int zone, int solution, const char *name)
{
  PetscInt               dim, xs[3], xm[3], d;
  cgsize_t               rmin[3], rmax[3], rsize;
  int                    field;
  PetscScalar           *e;
  CGNS_ENUMT(DataType_t) datatype;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], NULL, NULL, NULL));
  PetscCall(GetCGNSDataType_Private(PETSC_SCALAR, &datatype));

  rsize = 1;
  for (d = 0; d < dim; ++d) {
    rmin[d] = xs[d] + 1;
    rmax[d] = xs[d] + xm[d];
    rsize *= rmax[d] - rmin[d] + 1;
  }

  PetscCall(PetscMalloc1(rsize, &e));
  switch (dim) {
  case 2:
    PetscCall(GetLocalEntries2d_Private(dm, v, DMSTAG_ELEMENT, 0, e));
    break;
  case 3:
    PetscCall(GetLocalEntries3d_Private(dm, v, DMSTAG_ELEMENT, 0, e));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
  }
  CGNSCall(cgp_field_write(file_num, base, zone, solution, datatype, name, &field));
  CGNSCall(cgp_field_write_data(file_num, base, zone, solution, field, rmin, rmax, e));
  PetscCall(PetscFree(e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteFaceCenteredSolution_Private(DM dm, Vec v, int file_num, int base, int zone, int solution, const char *const names[])
{
  PetscInt                    dim, M[3], xs[3], xm[3], nExtra[3], d, l;
  PetscBool                   isLastRank[3];
  cgsize_t                    array_size[3], rmin[3], rmax[3], rsize;
  int                         array;
  PetscScalar                *e;
  CGNS_ENUMT(DataType_t)      datatype;
  const DMStagStencilLocation locs[3] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetGlobalSizes(dm, &M[0], &M[1], &M[2]));
  PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRank[0], &isLastRank[1], &isLastRank[2]));
  for (d = 0; d < dim; ++d) nExtra[d] = isLastRank[d] ? 1 : 0;
  PetscCall(GetCGNSDataType_Private(PETSC_SCALAR, &datatype));

  for (l = 0; l < dim; ++l) {
    rsize = 1;
    for (d = 0; d < dim; ++d) {
      array_size[d] = M[d] + (d == l ? 1 : 0);
      rmin[d]       = xs[d] + 1;
      rmax[d]       = xs[d] + xm[d] + (d == l ? nExtra[d] : 0);
      rsize *= rmax[d] - rmin[d] + 1;
    }

    PetscCall(PetscMalloc1(rsize, &e));
    switch (dim) {
    case 2:
      PetscCall(GetLocalEntries2d_Private(dm, v, locs[l], 0, e));
      break;
    case 3:
      PetscCall(GetLocalEntries3d_Private(dm, v, locs[l], 0, e));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, "UserDefinedData_t", l + 1, NULL));
    CGNSCall(cgp_array_write(names[l], datatype, dim, array_size, &array));
    CGNSCall(cgp_array_write_data(array, rmin, rmax, e));
    PetscCall(PetscFree(e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolView_FSMCGNS(Sol sol, PetscViewer viewer)
{
  Sol_FSM               *fsm = (Sol_FSM *)sol->data;
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  DM                     dm, fdm;
  PetscInt               dim, d;
  PetscReal              time;
  PetscInt               step;
  PetscReal             *time_slot;
  size_t                *step_slot;
  char                   solution_name[PETSC_MAX_PATH_LEN];
  int                    solution;
  PetscBool              iscart;
  const char *const      velnames[3]          = {"VelocityX", "VelocityY", "VelocityZ"};
  const char *const      intervelnames[3]     = {"IntermediateVelocityX", "IntermediateVelocityY", "IntermediateVelocityZ"};
  const char *const      convecnames[3]       = {"ConvectionX", "ConvectionY", "ConvectionZ"};
  const char *const      prevconvecnames[3]   = {"PrevConvectionX", "PrevConvectionY", "PrevConvectionZ"};
  const char *const      facevelnames[3]      = {"FaceVelocityX", "FaceVelocityY", "FaceVelocityZ"};
  const char *const      faceintervelnames[3] = {"FaceIntermediateVelocityX", "FaceIntermediateVelocityY", "FaceIntermediateVelocityZ"};

  PetscFunctionBegin;
  if (!cgv->file_num || !cgv->base) PetscCall(MeshView(sol->mesh, viewer));

  PetscCall(MeshGetDM(sol->mesh, &dm));
  PetscCall(MeshGetFaceDM(sol->mesh, &fdm));
  PetscCall(MeshGetDim(sol->mesh, &dim));

  if (!cgv->output_times) PetscCall(PetscSegBufferCreate(sizeof(PetscReal), 20, &cgv->output_times));
  if (!cgv->output_steps) PetscCall(PetscSegBufferCreate(sizeof(size_t), 20, &cgv->output_steps));
  PetscCall(DMGetOutputSequenceNumber(dm, &step, &time));
  if (time < 0.0) {
    step = 0;
    time = 0.0;
  }
  PetscCall(PetscSegBufferGet(cgv->output_times, 1, &time_slot));
  *time_slot = time;
  PetscCall(PetscSegBufferGet(cgv->output_steps, 1, &step_slot));
  *step_slot = step;
  PetscCall(PetscSNPrintf(solution_name, sizeof(solution_name), "FlowSolution%" PetscInt_FMT, step));
  CGNSCall(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, solution_name, CGNS_ENUMV(CellCenter), &solution));

  PetscCall(PetscObjectTypeCompare((PetscObject)sol->mesh, MESHCART, &iscart));

  /* Write cell-centered solutions */
  if (iscart) {
    for (d = 0; d < dim; ++d) {
      PetscCall(WriteCellCenteredSolution_Private(dm, sol->v[d], cgv->file_num, cgv->base, cgv->zone, solution, velnames[d]));
      PetscCall(WriteCellCenteredSolution_Private(dm, fsm->v_star[d], cgv->file_num, cgv->base, cgv->zone, solution, intervelnames[d]));
      PetscCall(WriteCellCenteredSolution_Private(dm, fsm->N[d], cgv->file_num, cgv->base, cgv->zone, solution, convecnames[d]));
      PetscCall(WriteCellCenteredSolution_Private(dm, fsm->N_prev[d], cgv->file_num, cgv->base, cgv->zone, solution, prevconvecnames[d]));
    }
    PetscCall(WriteCellCenteredSolution_Private(dm, sol->p, cgv->file_num, cgv->base, cgv->zone, solution, "Pressure"));
    PetscCall(WriteCellCenteredSolution_Private(dm, fsm->p_half, cgv->file_num, cgv->base, cgv->zone, solution, "PressureHalfStep"));
    PetscCall(WriteCellCenteredSolution_Private(dm, fsm->p_half_prev, cgv->file_num, cgv->base, cgv->zone, solution, "PrevPressureHalfStep"));
    PetscCall(WriteCellCenteredSolution_Private(dm, fsm->p_prime, cgv->file_num, cgv->base, cgv->zone, solution, "PressureCorrection"));
  }

  /* Write face-centered solutions */
  if (iscart) {
    CGNSCall(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "FlowSolution_t", solution, NULL));
    CGNSCall(cg_user_data_write("IFaceCenteredSolutions"));
    CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", 1, NULL));
    CGNSCall(cg_gridlocation_write(CGNS_ENUMV(IFaceCenter)));
    CGNSCall(cg_gorel(cgv->file_num, "..", 0, NULL));
    CGNSCall(cg_user_data_write("JFaceCenteredSolutions"));
    CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", 2, NULL));
    CGNSCall(cg_gridlocation_write(CGNS_ENUMV(JFaceCenter)));
    if (dim > 2) {
      CGNSCall(cg_gorel(cgv->file_num, "..", 0, NULL));
      CGNSCall(cg_user_data_write("KFaceCenteredSolutions"));
      CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", 3, NULL));
      CGNSCall(cg_gridlocation_write(CGNS_ENUMV(KFaceCenter)));
    }

    PetscCall(WriteFaceCenteredSolution_Private(fdm, fsm->fv, cgv->file_num, cgv->base, cgv->zone, solution, facevelnames));
    PetscCall(WriteFaceCenteredSolution_Private(fdm, fsm->fv_star, cgv->file_num, cgv->base, cgv->zone, solution, faceintervelnames));
  }

  PetscCall(PetscViewerFlucaCGNSCheckBatch_Internal(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
