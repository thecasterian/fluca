#include <fluca/private/flucaviewercgnsimpl.h>
#include <petscdmstag.h>

/* CGNS output of fields of a vector on a DMStag. The grid zone itself is written by the Mesh (MeshView()); these
   routines only add a FlowSolution per output step and the field arrays under it. */

static const char *const                face_sol_names[3]     = {"IFaceCenteredSolution", "JFaceCenteredSolution", "KFaceCenteredSolution"};
static const CGNS_ENUMT(GridLocation_t) face_sol_grid_locs[3] = {CGNS_ENUMV(IFaceCenter), CGNS_ENUMV(JFaceCenter), CGNS_ENUMV(KFaceCenter)};

static PetscErrorCode DMStagGetLocalEntries2d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt c, PetscScalar *e)
{
  PetscInt       x, y, m, n, nExtrax, nExtray;
  PetscBool      isLastRankx, isLastRanky;
  Vec            vlocal;
  PetscScalar ***arr;
  PetscInt       iloc, i, j, cnt = 0;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRankx, &isLastRanky, NULL));
  nExtrax = (loc == DMSTAG_LEFT && isLastRankx) ? 1 : 0;
  nExtray = (loc == DMSTAG_DOWN && isLastRanky) ? 1 : 0;
  PetscCall(DMGetLocalVector(dm, &vlocal));
  PetscCall(DMGlobalToLocal(dm, v, INSERT_VALUES, vlocal));
  PetscCall(DMStagVecGetArrayRead(dm, vlocal, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, c, &iloc));
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m + nExtrax; ++i) e[cnt++] = arr[j][i][iloc];
  PetscCall(DMStagVecRestoreArrayRead(dm, vlocal, &arr));
  PetscCall(DMRestoreLocalVector(dm, &vlocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagGetLocalEntries3d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt c, PetscScalar *e)
{
  PetscInt        x, y, z, m, n, p, nExtrax, nExtray, nExtraz;
  PetscBool       isLastRankx, isLastRanky, isLastRankz;
  Vec             vlocal;
  PetscScalar ****arr;
  PetscInt        iloc, i, j, k, cnt = 0;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &isLastRankx, &isLastRanky, &isLastRankz));
  nExtrax = (loc == DMSTAG_LEFT && isLastRankx) ? 1 : 0;
  nExtray = (loc == DMSTAG_DOWN && isLastRanky) ? 1 : 0;
  nExtraz = (loc == DMSTAG_BACK && isLastRankz) ? 1 : 0;
  PetscCall(DMGetLocalVector(dm, &vlocal));
  PetscCall(DMGlobalToLocal(dm, v, INSERT_VALUES, vlocal));
  PetscCall(DMStagVecGetArrayRead(dm, vlocal, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, c, &iloc));
  for (k = z; k < z + p + nExtraz; ++k)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m + nExtrax; ++i) e[cnt++] = arr[k][j][i][iloc];
  PetscCall(DMStagVecRestoreArrayRead(dm, vlocal, &arr));
  PetscCall(DMRestoreLocalVector(dm, &vlocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagWriteCellCenteredSolution_Private(DM dm, Vec v, PetscInt c, int file_num, int base, int zone, int sol, const char *name)
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
    PetscCall(DMStagGetLocalEntries2d_Private(dm, v, DMSTAG_ELEMENT, c, e));
    break;
  case 3:
    PetscCall(DMStagGetLocalEntries3d_Private(dm, v, DMSTAG_ELEMENT, c, e));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
  }
  CGNSCall(cgp_field_write(file_num, base, zone, sol, datatype, name, &field));
  CGNSCall(cgp_field_write_data(file_num, base, zone, sol, field, rmin, rmax, e));
  PetscCall(PetscFree(e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagWriteFaceCenteredSolution_Private(DM dm, Vec v, PetscInt c, int file_num, int base, int zone, int sol, const char *names)
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
      PetscCall(DMStagGetLocalEntries2d_Private(dm, v, locs[l], c, e));
      break;
    case 3:
      PetscCall(DMStagGetLocalEntries3d_Private(dm, v, locs[l], c, e));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", sol, "UserDefinedData_t", l + 1, NULL));
    CGNSCall(cgp_array_write(names, datatype, dim, array_size, &array));
    CGNSCall(cgp_array_write_data(array, rmin, rmax, e));
    PetscCall(PetscFree(e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagSetLocalEntries2d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt c, PetscScalar *e)
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
  PetscCall(DMStagGetLocationSlot(dm, loc, c, &iloc));
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m + nExtrax; ++i) arr[j][i][iloc] = e[cnt++];
  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagSetLocalEntries3d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt c, PetscScalar *e)
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
  PetscCall(DMStagGetLocationSlot(dm, loc, c, &iloc));
  for (k = z; k < z + p + nExtraz; ++k)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m + nExtrax; ++i) arr[k][j][i][iloc] = e[cnt++];
  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FindCellCenteredSolutionFieldInfo_Private(int file_num, int base, int zone, int sol, const char *field_name, int *field, CGNS_ENUMT(DataType_t) *data_type, PetscBool *flg)
{
  int                    num_fields, f;
  char                   field_name_read[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(DataType_t) data_type_read;
  PetscBool              flg_name;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  CGNSCall(cg_nfields(file_num, base, zone, sol, &num_fields));
  for (f = 1; f <= num_fields; ++f) {
    CGNSCall(cg_field_info(file_num, base, zone, sol, f, &data_type_read, field_name_read));
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

static PetscErrorCode FindFaceCenteredSolutionUserData_Private(int file_num, int base, int zone, int sol, const char *user_data_name, int *user_data, PetscBool *flg)
{
  int       num_user_data, u;
  char      user_data_name_read[CGIO_MAX_NAME_LENGTH + 1];
  PetscBool flg_name;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", sol, NULL));
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

static PetscErrorCode FindFaceCenteredSolutionArrayInfo_Private(int file_num, int base, int zone, int sol, int user_data, const char *array_name, int *array, CGNS_ENUMT(DataType_t) *data_type, PetscBool *flg)
{
  int                    num_arrays, a;
  char                   array_name_read[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(DataType_t) data_type_read;
  int                    data_dim_read;
  cgsize_t               dim_vec_read[CGIO_MAX_DIMENSIONS];
  PetscBool              flg_name;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", sol, "UserDefinedData_t", user_data, NULL));
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

static PetscErrorCode DMStagLoadCellCenteredSolution_Private(DM dm, Vec v, PetscInt c, int file_num, int base, int zone, int sol, const char *name)
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

    PetscCall(FindCellCenteredSolutionFieldInfo_Private(file_num, base, zone, sol, name, &field, &data_type, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_FILE_UNEXPECTED, "Cannot find field %s in base %d zone %d solution %d", name, base, zone, sol);
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
    CGNSCall(cgp_field_read_data(file_num, base, zone, sol, field, rmin, rmax, e_float));
    for (i = 0; i < rsize; ++i) e[i] = e_float[i];
    PetscCall(PetscFree(e_float));
    break;
  case CGNS_ENUMV(RealDouble):
    PetscCall(PetscMalloc1(rsize, &e_double));
    CGNSCall(cgp_field_read_data(file_num, base, zone, sol, field, rmin, rmax, e_double));
    for (i = 0; i < rsize; ++i) e[i] = e_double[i];
    PetscCall(PetscFree(e_double));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported data type: %s", DataTypeName[data_type]);
  }

  switch (dim) {
  case 2:
    PetscCall(DMStagSetLocalEntries2d_Private(dm, v, DMSTAG_ELEMENT, c, e));
    break;
  case 3:
    PetscCall(DMStagSetLocalEntries3d_Private(dm, v, DMSTAG_ELEMENT, c, e));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
  }
  PetscCall(PetscFree(e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagLoadFaceCenteredSolution_Private(DM dm, Vec v, PetscInt c, int file_num, int base, int zone, int sol, int user_data[], const char name[])
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

    PetscCall(FindFaceCenteredSolutionArrayInfo_Private(file_num, base, zone, sol, user_data[d], name, &array[d], &data_type[d], &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)dm), PETSC_ERR_FILE_UNEXPECTED, "Cannot find array %s in base %d zone %d solution %d user data %d", name, base, zone, sol, user_data[d]);
  }

  for (l = 0; l < dim; ++l) {
    rsize = 1;
    for (d = 0; d < dim; ++d) {
      rmin[d] = x[d] + 1;
      rmax[d] = x[d] + m[d] + (d == l ? nExtra[d] : 0);
      rsize *= rmax[d] - rmin[d] + 1;
    }

    PetscCall(PetscMalloc1(rsize, &e));
    CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", sol, "UserDefinedData_t", user_data[l], NULL));
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
      PetscCall(DMStagSetLocalEntries2d_Private(dm, v, locs[l], c, e));
      break;
    case 3:
      PetscCall(DMStagSetLocalEntries3d_Private(dm, v, locs[l], c, e));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    PetscCall(PetscFree(e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define FLUCACGNS_MAX_SOL_NAMES 64 /* names written to one FlowSolution; a few components per field */

/* Component names already written to the current FlowSolution of a viewer. CGNS metadata (cg_nfields()/cg_narrays())
   cannot be queried while the file is open for writing, so this is tracked here, composed onto the PetscViewer since the
   FlowSolution outlives a single call. The FlowSolution's CGNS index cannot detect a new FlowSolution, because it
   restarts at 1 in every newly opened file (e.g. a batch_size-1 filename template opens a new file at every step), so
   the list is reset whenever a FlowSolution is created. */
typedef struct {
  PetscInt n;
  char     names[FLUCACGNS_MAX_SOL_NAMES][CGIO_MAX_NAME_LENGTH + 1];
} FlucaCGNSSolNames;

#define FLUCACGNS_SOL_NAMES_COMPOSED_NAME "Fluca_CGNSSolNames"

static PetscErrorCode PetscViewerFlucaCGNSGetSolNames_Private(PetscViewer viewer, FlucaCGNSSolNames **sn)
{
  PetscContainer container;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)viewer, FLUCACGNS_SOL_NAMES_COMPOSED_NAME, (PetscObject *)&container));
  if (!container) {
    PetscCall(PetscNew(sn));
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)viewer), &container));
    PetscCall(PetscContainerSetPointer(container, *sn));
    PetscCall(PetscContainerSetCtxDestroy(container, PetscCtxDestroyDefault));
    PetscCall(PetscObjectCompose((PetscObject)viewer, FLUCACGNS_SOL_NAMES_COMPOSED_NAME, (PetscObject)container));
    PetscCall(PetscContainerDestroy(&container));
  } else {
    PetscCall(PetscContainerGetPointer(container, (void **)sn));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fail if name was already written to the current FlowSolution of viewer; otherwise record it */
static PetscErrorCode PetscViewerFlucaCGNSRecordSolName_Private(PetscViewer viewer, PetscInt step, const char name[])
{
  FlucaCGNSSolNames *sn;
  PetscInt           i;
  PetscBool          same;

  PetscFunctionBegin;
  PetscCall(PetscViewerFlucaCGNSGetSolNames_Private(viewer, &sn));
  for (i = 0; i < sn->n; ++i) {
    PetscCall(PetscStrcmp(sn->names[i], name, &same));
    PetscCheck(!same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Field %s of step %" PetscInt_FMT " is already written to this viewer", name, step);
  }
  PetscCheck(sn->n < FLUCACGNS_MAX_SOL_NAMES, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Cannot track more than %d field names written to one CGNS solution", FLUCACGNS_MAX_SOL_NAMES);
  PetscCall(PetscStrncpy(sn->names[sn->n], name, sizeof(sn->names[sn->n])));
  ++sn->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Start output step step at time time: on a new step, close the file if its batch is full, open the file for this step if
   none is open, and record the step and time. The grid zone must then be written (MeshView()) before any field. */
PetscErrorCode PetscViewerFlucaCGNSBeginStep_Internal(PetscViewer viewer, PetscInt step, PetscReal time)
{
  PetscViewer_FlucaCGNS *cgv;
  size_t                *step_slot;
  PetscReal             *time_slot;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERFLUCACGNS);
  cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  if (cgv->last_step == step) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscViewerFlucaCGNSCheckBatch_Internal(viewer));
  cgv->sol = 0;
  if (!cgv->file_num) PetscCall(PetscViewerFlucaCGNSFileOpen_Internal(viewer, step));
  if (!cgv->output_steps) PetscCall(PetscSegBufferCreate(sizeof(size_t), 20, &cgv->output_steps));
  if (!cgv->output_times) PetscCall(PetscSegBufferCreate(sizeof(PetscReal), 20, &cgv->output_times));
  PetscCall(PetscSegBufferGet(cgv->output_steps, 1, &step_slot));
  PetscCall(PetscSegBufferGet(cgv->output_times, 1, &time_slot));
  *step_slot     = step;
  *time_slot     = time;
  cgv->last_step = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFlucaCGNSWriteDMStagComponents_Internal(PetscViewer viewer, Vec v, DMStagStencilLocation loc, PetscInt c0, PetscInt ncomp, const char name[])
{
  PetscViewer_FlucaCGNS *cgv;
  DM                     dm;
  PetscInt               dim, step, c, d;
  char                   sol_name[PETSC_MAX_PATH_LEN], comp_name[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERFLUCACGNS);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscAssertPointer(name, 6);
  cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  PetscCheck(loc == DMSTAG_ELEMENT || loc == DMSTAG_LEFT, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_OUTOFRANGE, "Location must be DMSTAG_ELEMENT or DMSTAG_LEFT");
  PetscCheck(cgv->last_step >= 0 && cgv->zone, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Call PetscViewerFlucaCGNSBeginStep_Internal() and write the grid zone first");
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  step = cgv->last_step;

  if (!cgv->sol) {
    FlucaCGNSSolNames *sn;

    /* One FlowSolution per step: cell fields directly under it, face arrays under one UserDefinedData per direction */
    PetscCall(PetscSNPrintf(sol_name, sizeof(sol_name), "FlowSolution%" PetscInt_FMT, step));
    CGNSCall(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, sol_name, CGNS_ENUMV(CellCenter), &cgv->sol));
    PetscCall(PetscViewerFlucaCGNSGetSolNames_Private(viewer, &sn));
    sn->n = 0;
    CGNSCall(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "FlowSolution_t", cgv->sol, NULL));
    for (d = 0; d < dim; ++d) {
      CGNSCall(cg_user_data_write(face_sol_names[d]));
      CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", d + 1, NULL));
      CGNSCall(cg_gridlocation_write(face_sol_grid_locs[d]));
      CGNSCall(cg_gorel(cgv->file_num, "..", 0, NULL));
    }
  }

  for (c = 0; c < ncomp; ++c) {
    if (ncomp == 1) PetscCall(PetscStrncpy(comp_name, name, sizeof(comp_name)));
    else PetscCall(PetscSNPrintf(comp_name, sizeof(comp_name), "%s%c", name, (char)('X' + c)));
    PetscCall(PetscViewerFlucaCGNSRecordSolName_Private(viewer, step, comp_name));
    if (loc == DMSTAG_ELEMENT) PetscCall(DMStagWriteCellCenteredSolution_Private(dm, v, c0 + c, cgv->file_num, cgv->base, cgv->zone, cgv->sol, comp_name));
    else PetscCall(DMStagWriteFaceCenteredSolution_Private(dm, v, c0 + c, cgv->file_num, cgv->base, cgv->zone, cgv->sol, comp_name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFlucaCGNSReadDMStagComponents_Internal(PetscViewer viewer, Vec v, DMStagStencilLocation loc, PetscInt c0, PetscInt ncomp, const char name[])
{
  PetscViewer_FlucaCGNS     *cgv;
  MPI_Comm                   comm = PetscObjectComm((PetscObject)v);
  DM                         dm;
  Vec                        locv;
  PetscInt                   dim, M[3], c, d;
  PetscBool                  flg;
  const int                  base = 1, zone = 1;
  int                        num_sols, cell_dim, phys_dim, sol, face_sol_user_data[3];
  char                       base_name[CGIO_MAX_NAME_LENGTH + 1], zone_name[CGIO_MAX_NAME_LENGTH + 1], sol_name[CGIO_MAX_NAME_LENGTH + 1];
  char                       comp_name[PETSC_MAX_PATH_LEN];
  CGNS_ENUMT(GridLocation_t) grid_loc;
  cgsize_t                   sizes[9];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERFLUCACGNS);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscAssertPointer(name, 6);
  PetscCall(PetscViewerCheckReadable(viewer));
  PetscCheck(loc == DMSTAG_ELEMENT || loc == DMSTAG_LEFT, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_OUTOFRANGE, "Location must be DMSTAG_ELEMENT or DMSTAG_LEFT");
  cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetGlobalSizes(dm, &M[0], &M[1], &M[2]));

  CGNSCall(cg_base_read(cgv->file_num, base, base_name, &cell_dim, &phys_dim));
  PetscCheck(cell_dim == dim, comm, PETSC_ERR_FILE_UNEXPECTED, "DM dimension %" PetscInt_FMT " does not match CGNS cell dimension %d", dim, cell_dim);
  CGNSCall(cg_zone_read(cgv->file_num, base, zone, zone_name, sizes));
  for (d = 0; d < dim; ++d) PetscCheck(M[d] == sizes[dim + d], comm, PETSC_ERR_FILE_UNEXPECTED, "DM size %" PetscInt_FMT " does not match CGNS zone size %ld", M[d], (long)sizes[dim + d]);

  /* The last FlowSolution is the latest step; CellInfo is written first so num_sols >= 2 when a solution exists */
  CGNSCall(cg_nsols(cgv->file_num, base, zone, &num_sols));
  sol = num_sols;
  CGNSCall(cg_sol_info(cgv->file_num, base, zone, sol, sol_name, &grid_loc));
  {
    PetscInt               sol_step;
    PetscReal             *times;
    size_t                 len;
    int                    count, ret, nsteps;
    char                   biter_name[CGIO_MAX_NAME_LENGTH + 1];
    CGNS_ENUMT(DataType_t) datatype;

    PetscCall(PetscStrlen(sol_name, &len));
    ret = sscanf(sol_name, "FlowSolution%" PetscInt_FMT "%n", &sol_step, &count);
    PetscCheck(ret == 1 && (int)len == count, comm, PETSC_ERR_FILE_UNEXPECTED, "%s is not a valid solution name", sol_name);
    CGNSCall(cg_biter_read(cgv->file_num, base, biter_name, &nsteps));
    CGNSCall(cg_goto(cgv->file_num, base, "BaseIterativeData_t", 1, NULL));
    PetscCall(FlucaGetCGNSDataType_Internal(PETSC_REAL, &datatype));
    PetscCall(PetscMalloc1(nsteps, &times));
    CGNSCall(cg_array_read_as(1, datatype, times));
    PetscCall(DMSetOutputSequenceNumber(dm, sol_step, times[nsteps - 1]));
    PetscCall(PetscFree(times));
  }

  if (loc == DMSTAG_LEFT) {
    for (d = 0; d < dim; ++d) {
      PetscCall(FindFaceCenteredSolutionUserData_Private(cgv->file_num, base, zone, sol, face_sol_names[d], &face_sol_user_data[d], &flg));
      PetscCheck(flg, comm, PETSC_ERR_FILE_UNEXPECTED, "Cannot find user data %s in solution %s", face_sol_names[d], sol_name);
    }
  }

  /* Start from the current values so the components outside [c0, c0 + ncomp) are kept */
  PetscCall(DMGetLocalVector(dm, &locv));
  PetscCall(DMGlobalToLocal(dm, v, INSERT_VALUES, locv));
  for (c = 0; c < ncomp; ++c) {
    if (ncomp == 1) PetscCall(PetscStrncpy(comp_name, name, sizeof(comp_name)));
    else PetscCall(PetscSNPrintf(comp_name, sizeof(comp_name), "%s%c", name, (char)('X' + c)));
    if (loc == DMSTAG_ELEMENT) PetscCall(DMStagLoadCellCenteredSolution_Private(dm, locv, c0 + c, cgv->file_num, base, zone, sol, comp_name));
    else PetscCall(DMStagLoadFaceCenteredSolution_Private(dm, locv, c0 + c, cgv->file_num, base, zone, sol, face_sol_user_data, comp_name));
  }
  PetscCall(DMLocalToGlobal(dm, locv, INSERT_VALUES, v));
  PetscCall(DMRestoreLocalVector(dm, &locv));
  PetscFunctionReturn(PETSC_SUCCESS);
}
