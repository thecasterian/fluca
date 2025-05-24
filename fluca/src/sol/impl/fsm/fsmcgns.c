#include <fluca/private/flucaviewer_cgns.h>
#include <flucameshcart.h>
#include <fluca/private/sol_fsm.h>
#include <petscdmstag.h>

static const char *const presname             = "Pressure";
static const char *const preshalfname         = "PressureHalfStep";
static const char *const prevpreshalfname     = "PrevPressureHalfStep";
static const char *const prescorrecname       = "PressureCorrection";
static const char *const velnames[3]          = {"VelocityX", "VelocityY", "VelocityZ"};
static const char *const intervelnames[3]     = {"IntermediateVelocityX", "IntermediateVelocityY", "IntermediateVelocityZ"};
static const char *const convecnames[3]       = {"ConvectionX", "ConvectionY", "ConvectionZ"};
static const char *const prevconvecnames[3]   = {"PrevConvectionX", "PrevConvectionY", "PrevConvectionZ"};
static const char *const facevelnames[3]      = {"FaceVelocityX", "FaceVelocityY", "FaceVelocityZ"};
static const char *const faceintervelnames[3] = {"FaceIntermediateVelocityX", "FaceIntermediateVelocityY", "FaceIntermediateVelocityZ"};

static const char *const                face_sol_names[3]     = {"IFaceCenteredSolution", "JFaceCenteredSolution", "KFaceCenteredSolution"};
static const CGNS_ENUMT(GridLocation_t) face_sol_grid_locs[3] = {CGNS_ENUMV(IFaceCenter), CGNS_ENUMV(JFaceCenter), CGNS_ENUMV(KFaceCenter)};

static PetscErrorCode DMStagGetLocalEntries2d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt d, PetscScalar *e)
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

static PetscErrorCode DMStagGetLocalEntries3d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt d, PetscScalar *e)
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

static PetscErrorCode DMStagWriteCellCenteredSolution_Private(DM dm, Vec v, int file_num, int base, int zone, int solution, const char *name)
{
  PetscInt               dim, x[3], m[3], d;
  cgsize_t               rmin[3], rmax[3], rsize;
  int                    field;
  PetscScalar           *e;
  CGNS_ENUMT(DataType_t) datatype;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(FlucaGetCGNSDataType_Internal(PETSC_SCALAR, &datatype));

  rsize = 1;
  for (d = 0; d < dim; ++d) {
    rmin[d] = x[d] + 1;
    rmax[d] = x[d] + m[d];
    rsize *= rmax[d] - rmin[d] + 1;
  }

  PetscCall(PetscMalloc1(rsize, &e));
  switch (dim) {
  case 2:
    PetscCall(DMStagGetLocalEntries2d_Private(dm, v, DMSTAG_ELEMENT, 0, e));
    break;
  case 3:
    PetscCall(DMStagGetLocalEntries3d_Private(dm, v, DMSTAG_ELEMENT, 0, e));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
  }
  CGNSCall(cgp_field_write(file_num, base, zone, solution, datatype, name, &field));
  CGNSCall(cgp_field_write_data(file_num, base, zone, solution, field, rmin, rmax, e));
  PetscCall(PetscFree(e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagWriteFaceCenteredSolution_Private(DM dm, Vec v, int file_num, int base, int zone, int solution, const int user_data[], const char *const names[])
{
  PetscInt                    dim, M[3], x[3], m[3], nExtra[3], d, l;
  PetscBool                   isLastRank[3];
  cgsize_t                    array_size[3], rmin[3], rmax[3], rsize;
  int                         array;
  PetscScalar                *e;
  CGNS_ENUMT(DataType_t)      datatype;
  const DMStagStencilLocation locs[3] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetGlobalSizes(dm, &M[0], &M[1], &M[2]));
  PetscCall(DMStagGetCorners(dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRank[0], &isLastRank[1], &isLastRank[2]));
  for (d = 0; d < dim; ++d) nExtra[d] = isLastRank[d] ? 1 : 0;
  PetscCall(FlucaGetCGNSDataType_Internal(PETSC_SCALAR, &datatype));

  for (l = 0; l < dim; ++l) {
    rsize = 1;
    for (d = 0; d < dim; ++d) {
      array_size[d] = M[d] + (d == l ? 1 : 0);
      rmin[d]       = x[d] + 1;
      rmax[d]       = x[d] + m[d] + (d == l ? nExtra[d] : 0);
      rsize *= rmax[d] - rmin[d] + 1;
    }

    PetscCall(PetscMalloc1(rsize, &e));
    switch (dim) {
    case 2:
      PetscCall(DMStagGetLocalEntries2d_Private(dm, v, locs[l], 0, e));
      break;
    case 3:
      PetscCall(DMStagGetLocalEntries3d_Private(dm, v, locs[l], 0, e));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, "UserDefinedData_t", user_data[l], NULL));
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

  PetscFunctionBegin;
  if (!cgv->file_num || !cgv->base) PetscCall(MeshView(sol->mesh, viewer));

  PetscCall(MeshGetDM(sol->mesh, &dm));
  PetscCall(MeshGetFaceDM(sol->mesh, &fdm));
  PetscCall(MeshGetDimension(sol->mesh, &dim));

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
      PetscCall(DMStagWriteCellCenteredSolution_Private(dm, sol->v[d], cgv->file_num, cgv->base, cgv->zone, solution, velnames[d]));
      PetscCall(DMStagWriteCellCenteredSolution_Private(dm, fsm->v_star[d], cgv->file_num, cgv->base, cgv->zone, solution, intervelnames[d]));
      PetscCall(DMStagWriteCellCenteredSolution_Private(dm, fsm->N[d], cgv->file_num, cgv->base, cgv->zone, solution, convecnames[d]));
      PetscCall(DMStagWriteCellCenteredSolution_Private(dm, fsm->N_prev[d], cgv->file_num, cgv->base, cgv->zone, solution, prevconvecnames[d]));
    }
    PetscCall(DMStagWriteCellCenteredSolution_Private(dm, sol->p, cgv->file_num, cgv->base, cgv->zone, solution, presname));
    PetscCall(DMStagWriteCellCenteredSolution_Private(dm, fsm->p_half, cgv->file_num, cgv->base, cgv->zone, solution, preshalfname));
    PetscCall(DMStagWriteCellCenteredSolution_Private(dm, fsm->p_half_prev, cgv->file_num, cgv->base, cgv->zone, solution, prevpreshalfname));
    PetscCall(DMStagWriteCellCenteredSolution_Private(dm, fsm->p_prime, cgv->file_num, cgv->base, cgv->zone, solution, prescorrecname));
  }

  /* Write face-centered solutions */
  if (iscart) {
    /* Create user data for face-centered solutions
     * Solution tree structure:
     *  - solution
     *     - grid location = cell center
     *     - fields for cell-centered solutions
     *     - user data for i-face
     *       - grid location = i-face center
     *       - arrays for i-face-centered solutions
     *     - user data for j-face
     *       - grid location = j-face center
     *       - arrays for j-face-centered solutions
     *     - user data for k-face
     *       - grid location = k-face center
     *       - arrays for k-face-centered solutions
     */
    const int face_sol_user_data[3] = {1, 2, 3};

    /* Be careful to the order of writing user data */
    CGNSCall(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "FlowSolution_t", solution, NULL));
    for (d = 0; d < dim; ++d) {
      CGNSCall(cg_user_data_write(face_sol_names[d]));
      CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", face_sol_user_data[d], NULL));
      CGNSCall(cg_gridlocation_write(face_sol_grid_locs[d]));
      CGNSCall(cg_gorel(cgv->file_num, "..", 0, NULL));
    }

    PetscCall(DMStagWriteFaceCenteredSolution_Private(fdm, fsm->fv, cgv->file_num, cgv->base, cgv->zone, solution, face_sol_user_data, facevelnames));
    PetscCall(DMStagWriteFaceCenteredSolution_Private(fdm, fsm->fv_star, cgv->file_num, cgv->base, cgv->zone, solution, face_sol_user_data, faceintervelnames));
  }

  PetscCall(PetscViewerFlucaCGNSCheckBatch_Internal(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagSetLocalEntries2d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt d, PetscScalar *e)
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
  PetscCall(DMStagVecGetArray(dm, v, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, d, &iloc));
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m + nExtrax; ++i) arr[j][i][iloc] = e[cnt++];
  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagSetLocalEntries3d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt d, PetscScalar *e)
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
  PetscCall(DMStagVecGetArray(dm, v, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, d, &iloc));
  for (k = z; k < z + p + nExtraz; ++k)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m + nExtrax; ++i) arr[k][j][i][iloc] = e[cnt++];
  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FindCellCenteredSolutionFieldInfo_Private(int file_num, int base, int zone, int solution, const char *field_name, int *field, CGNS_ENUMT(DataType_t) *data_type, PetscBool *flg)
{
  int                    num_fields, f;
  char                   field_name_read[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(DataType_t) data_type_read;
  PetscBool              flg_name;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  CGNSCall(cg_nfields(file_num, base, zone, solution, &num_fields));
  for (f = 1; f <= num_fields; ++f) {
    CGNSCall(cg_field_info(file_num, base, zone, solution, f, &data_type_read, field_name_read));
    PetscCall(PetscStrcmp(field_name_read, field_name, &flg_name));
    if (flg_name) {
      *field     = f;
      *data_type = data_type_read;
      *flg       = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FindFaceCenteredSolutionUserData_Private(int file_num, int base, int zone, int solution, const char *user_data_name, int *user_data, PetscBool *flg)
{
  int       num_user_data, u;
  char      user_data_name_read[CGIO_MAX_NAME_LENGTH + 1];
  PetscBool flg_name;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, NULL));
  CGNSCall(cg_nuser_data(&num_user_data));
  for (u = 1; u <= num_user_data; ++u) {
    CGNSCall(cg_user_data_read(u, user_data_name_read));
    PetscCall(PetscStrcmp(user_data_name_read, user_data_name, &flg_name));
    if (flg_name) {
      *user_data = u;
      *flg       = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FindFaceCenteredSolutionArrayInfo_Private(int file_num, int base, int zone, int solution, int user_data, const char *array_name, int *array, CGNS_ENUMT(DataType_t) *data_type, PetscBool *flg)
{
  int                    num_arrays, a;
  char                   array_name_read[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(DataType_t) data_type_read;
  int                    data_dim_read;
  cgsize_t               dim_vec_read[CGIO_MAX_DIMENSIONS];
  PetscBool              flg_name;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, "UserDefinedData_t", user_data, NULL));
  CGNSCall(cg_narrays(&num_arrays));
  for (a = 1; a <= num_arrays; ++a) {
    CGNSCall(cg_array_info(a, array_name_read, &data_type_read, &data_dim_read, dim_vec_read));
    PetscCall(PetscStrcmp(array_name_read, array_name, &flg_name));
    if (flg_name) {
      *array     = a;
      *data_type = data_type_read;
      *flg       = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagLoadCellCenteredSolution_Private(DM dm, Vec v, int file_num, int base, int zone, int solution, const char *name)
{
  PetscInt               dim, x[3], m[3], d, i;
  int                    field;
  CGNS_ENUMT(DataType_t) data_type;
  cgsize_t               rmin[3], rmax[3], rsize;
  float                 *e_float;
  double                *e_double;
  PetscScalar           *e;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));

  {
    PetscBool flg;

    PetscCall(FindCellCenteredSolutionFieldInfo_Private(file_num, base, zone, solution, name, &field, &data_type, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot find field %s in base %d zone %d solution %d", name, base, zone, solution);
  }

  rsize = 1;
  for (d = 0; d < dim; ++d) {
    rmin[d] = x[d] + 1;
    rmax[d] = x[d] + m[d];
    rsize *= rmax[d] - rmin[d] + 1;
  }

  PetscCall(PetscMalloc1(rsize, &e));
  switch (data_type) {
  case CGNS_ENUMV(RealSingle):
    PetscCall(PetscMalloc1(rsize, &e_float));
    CGNSCall(cgp_field_read_data(file_num, base, zone, solution, field, rmin, rmax, e_float));
    for (i = 0; i < rsize; ++i) e[i] = e_float[i];
    PetscCall(PetscFree(e_float));
    break;
  case CGNS_ENUMV(RealDouble):
    PetscCall(PetscMalloc1(rsize, &e_double));
    CGNSCall(cgp_field_read_data(file_num, base, zone, solution, field, rmin, rmax, e_double));
    for (i = 0; i < rsize; ++i) e[i] = e_double[i];
    PetscCall(PetscFree(e_double));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported data type: %s", DataTypeName[data_type]);
  }

  switch (dim) {
  case 2:
    PetscCall(DMStagSetLocalEntries2d_Private(dm, v, DMSTAG_ELEMENT, 0, e));
    break;
  case 3:
    PetscCall(DMStagSetLocalEntries3d_Private(dm, v, DMSTAG_ELEMENT, 0, e));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
  }
  PetscCall(PetscFree(e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagLoadFaceCenteredSolution_Private(DM dm, Vec v, int file_num, int base, int zone, int solution, int user_data[], const char *const names[])
{
  PetscInt                    dim, M[3], x[3], m[3], nExtra[3], d, l, i;
  PetscBool                   isLastRank[3];
  int                         array[3];
  CGNS_ENUMT(DataType_t)      data_type[3];
  cgsize_t                    rmin[3], rmax[3], rsize;
  float                      *e_float;
  double                     *e_double;
  PetscScalar                *e;
  const DMStagStencilLocation locs[3] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetGlobalSizes(dm, &M[0], &M[1], &M[2]));
  PetscCall(DMStagGetCorners(dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRank[0], &isLastRank[1], &isLastRank[2]));
  for (d = 0; d < dim; ++d) nExtra[d] = isLastRank[d] ? 1 : 0;

  for (d = 0; d < dim; ++d) {
    PetscBool flg;

    PetscCall(FindFaceCenteredSolutionArrayInfo_Private(file_num, base, zone, solution, user_data[d], names[d], &array[d], &data_type[d], &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot find array %s in base %d zone %d solution %d user data %d", names[d], base, zone, solution, user_data[d]);
  }

  for (l = 0; l < dim; ++l) {
    rsize = 1;
    for (d = 0; d < dim; ++d) {
      rmin[d] = x[d] + 1;
      rmax[d] = x[d] + m[d] + (d == l ? nExtra[d] : 0);
      rsize *= rmax[d] - rmin[d] + 1;
    }

    PetscCall(PetscMalloc1(rsize, &e));
    CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, "UserDefinedData_t", user_data[l], NULL));
    switch (data_type[l]) {
    case CGNS_ENUMV(RealSingle):
      PetscCall(PetscMalloc1(rsize, &e_float));
      CGNSCall(cgp_array_read_data(array[l], rmin, rmax, e_float));
      for (i = 0; i < rsize; ++i) e[i] = e_float[i];
      PetscCall(PetscFree(e_float));
      break;
    case CGNS_ENUMV(RealDouble):
      PetscCall(PetscMalloc1(rsize, &e_double));
      CGNSCall(cgp_array_read_data(array[l], rmin, rmax, e_double));
      for (i = 0; i < rsize; ++i) e[i] = e_double[i];
      PetscCall(PetscFree(e_double));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported data type: %s", DataTypeName[data_type[l]]);
    }

    switch (dim) {
    case 2:
      PetscCall(DMStagSetLocalEntries2d_Private(dm, v, locs[l], 0, e));
      break;
    case 3:
      PetscCall(DMStagSetLocalEntries3d_Private(dm, v, locs[l], 0, e));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    PetscCall(PetscFree(e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolLoadCGNS_FSM(Sol sol, PetscInt file_num)
{
  Sol_FSM               *fsm = (Sol_FSM *)sol->data;
  DM                     dm, fdm;
  PetscInt               dim, d;
  PetscBool              iscart;
  const int              base = 1, zone = 1;
  int                    solution;
  int                    num_bases, num_zones, num_solutions, num_fields;
  int                    cell_dim, phys_dim;
  char                   base_name[CGIO_MAX_NAME_LENGTH + 1];
  char                   zone_name[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(ZoneType_t) zone_type;
  cgsize_t               sizes[9];

  PetscFunctionBegin;
  PetscCall(MeshGetDM(sol->mesh, &dm));
  PetscCall(MeshGetFaceDM(sol->mesh, &fdm));
  PetscCall(MeshGetDimension(sol->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)sol->mesh, MESHCART, &iscart));

  /* Read CGNS file info */
  CGNSCall(cg_nbases(file_num, &num_bases));
  PetscCheck(num_bases == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Only one base is supported but found %d", num_bases);
  CGNSCall(cg_base_read(file_num, base, base_name, &cell_dim, &phys_dim));
  PetscCheck(cell_dim == dim, PETSC_COMM_SELF, PETSC_ERR_LIB, "Mesh dimension %" PetscInt_FMT " does not match CGNS cell dimension %d", dim, cell_dim);
  CGNSCall(cg_nzones(file_num, base, &num_zones));
  PetscCheck(num_zones == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Only one zone is supported but found %d", num_zones);
  CGNSCall(cg_zone_read(file_num, base, zone, zone_name, sizes));
  CGNSCall(cg_zone_type(file_num, base, zone, &zone_type));
  if (iscart) {
    PetscInt M[3];

    PetscCall(MeshCartGetGlobalSizes(sol->mesh, &M[0], &M[1], &M[2]));
    PetscCheck(zone_type == CGNS_ENUMV(Structured), PETSC_COMM_SELF, PETSC_ERR_LIB, "Only structured zone is supported for Cartesian mesh");
    for (d = 0; d < dim; ++d) PetscCheck(M[d] == sizes[dim + d], PETSC_COMM_SELF, PETSC_ERR_LIB, "Mesh size %" PetscInt_FMT " does not match CGNS zone size %ld", M[d], (long)sizes[dim + d]);
  }
  CGNSCall(cg_nsols(file_num, base, zone, &num_solutions));
  solution = num_solutions; /* Assume that the last solution is the one we want */
  CGNSCall(cg_nfields(file_num, base, zone, solution, &num_fields));

  /* Load cell-centered solutions */
  if (iscart) {
    CGNS_ENUMT(GridLocation_t) grid_loc;

    CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, NULL));
    CGNSCall(cg_gridlocation_read(&grid_loc));
    PetscCheck(grid_loc == CGNS_ENUMV(CellCenter), PETSC_COMM_SELF, PETSC_ERR_LIB, "Grid location is not cell-centered in base %d zone %d solution %d", base, zone, solution);

    for (d = 0; d < dim; ++d) {
      PetscCall(DMStagLoadCellCenteredSolution_Private(dm, sol->v[d], file_num, base, zone, solution, velnames[d]));
      PetscCall(DMStagLoadCellCenteredSolution_Private(dm, fsm->v_star[d], file_num, base, zone, solution, intervelnames[d]));
      PetscCall(DMStagLoadCellCenteredSolution_Private(dm, fsm->N[d], file_num, base, zone, solution, convecnames[d]));
      PetscCall(DMStagLoadCellCenteredSolution_Private(dm, fsm->N_prev[d], file_num, base, zone, solution, prevconvecnames[d]));
    }
    PetscCall(DMStagLoadCellCenteredSolution_Private(dm, sol->p, file_num, base, zone, solution, presname));
    PetscCall(DMStagLoadCellCenteredSolution_Private(dm, fsm->p_half, file_num, base, zone, solution, preshalfname));
    PetscCall(DMStagLoadCellCenteredSolution_Private(dm, fsm->p_half_prev, file_num, base, zone, solution, prevpreshalfname));
    PetscCall(DMStagLoadCellCenteredSolution_Private(dm, fsm->p_prime, file_num, base, zone, solution, prescorrecname));
  }

  /* Load face-centered solutions */
  if (iscart) {
    int                        face_sol_user_data[3];
    CGNS_ENUMT(GridLocation_t) grid_loc;
    PetscBool                  flg;

    for (d = 0; d < dim; ++d) {
      PetscCall(FindFaceCenteredSolutionUserData_Private(file_num, base, zone, solution, face_sol_names[d], &face_sol_user_data[d], &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot find user data %s in base %d zone %d solution %d", face_sol_names[d], base, zone, solution);

      CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, "UserDefinedData_t", face_sol_user_data[d], NULL));
      CGNSCall(cg_gridlocation_read(&grid_loc));
      PetscCheck(grid_loc == face_sol_grid_locs[d], PETSC_COMM_SELF, PETSC_ERR_LIB, "Grid location is not %c-face-centered in base %d zone %d solution %d user data %d", 'i' + d, base, zone, solution, face_sol_user_data[d]);
    }

    PetscCall(DMStagLoadFaceCenteredSolution_Private(fdm, fsm->fv, file_num, base, zone, solution, face_sol_user_data, facevelnames));
    PetscCall(DMStagLoadFaceCenteredSolution_Private(fdm, fsm->fv_star, file_num, base, zone, solution, face_sol_user_data, faceintervelnames));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
