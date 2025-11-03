#include <fluca/private/flucaviewercgnsimpl.h>
#include <flucameshcart.h>
#include <fluca/private/nsfsmimpl.h>
#include <petscdmstag.h>

static const char *const pressure_name           = "Pressure";
static const char *const pressure_half_step_name = "PressureHalfStep";
static const char *const velocity_names[3]       = {"VelocityX", "VelocityY", "VelocityZ"};

static PetscErrorCode DMStagSetLocalEntries2d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt c, PetscScalar *e)
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
  PetscCall(DMStagVecGetArray(dm, vlocal, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, c, &iloc));
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m + nExtrax; ++i) arr[j][i][iloc] = e[cnt++];
  PetscCall(DMStagVecRestoreArray(dm, vlocal, &arr));
  PetscCall(DMLocalToGlobal(dm, vlocal, INSERT_VALUES, v));
  PetscCall(DMRestoreLocalVector(dm, &vlocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMStagSetLocalEntries3d_Private(DM dm, Vec v, DMStagStencilLocation loc, PetscInt c, PetscScalar *e)
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
  PetscCall(DMStagVecGetArray(dm, vlocal, &arr));
  PetscCall(DMStagGetLocationSlot(dm, loc, c, &iloc));
  for (k = z; k < z + p + nExtraz; ++k)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m + nExtrax; ++i) arr[k][j][i][iloc] = e[cnt++];
  PetscCall(DMStagVecRestoreArray(dm, vlocal, &arr));
  PetscCall(DMLocalToGlobal(dm, vlocal, INSERT_VALUES, v));
  PetscCall(DMRestoreLocalVector(dm, &vlocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FindSolutionFieldInfo_Private(int file_num, int base, int zone, int solution, const char *field_name, int *field, CGNS_ENUMT(DataType_t) *data_type, PetscBool *flg)
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

static PetscErrorCode DMStagLoadSolution_Private(DM dm, Vec v, PetscInt c, int file_num, int base, int zone, int solution, const char *name)
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

    PetscCall(FindSolutionFieldInfo_Private(file_num, base, zone, solution, name, &field, &data_type, &flg));
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

PetscErrorCode NSLoadSolutionCGNS_FSM_Cart_Internal(NS ns, PetscInt file_num)
{
  NS_FSM                    *fsm = (NS_FSM *)ns->data;
  DM                         sdm, vdm;
  PetscInt                   dim, M[3], d;
  cgsize_t                   sizes[9];
  const int                  base = 1, zone = 1;
  int                        solution;
  int                        num_bases, num_zones, num_solutions, num_fields;
  int                        cell_dim, phys_dim;
  char                       base_name[CGIO_MAX_NAME_LENGTH + 1];
  char                       zone_name[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(ZoneType_t)     zone_type;
  CGNS_ENUMT(GridLocation_t) grid_loc;
  Vec                        v, p;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_SCALAR, &sdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDimension(ns->mesh, &dim));

  /* Read CGNS file info */
  CGNSCall(cg_nbases(file_num, &num_bases));
  PetscCheck(num_bases == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Only one base is supported but found %d", num_bases);
  CGNSCall(cg_base_read(file_num, base, base_name, &cell_dim, &phys_dim));
  PetscCheck(cell_dim == dim, PETSC_COMM_SELF, PETSC_ERR_LIB, "Mesh dimension %" PetscInt_FMT " does not match CGNS cell dimension %d", dim, cell_dim);
  CGNSCall(cg_nzones(file_num, base, &num_zones));
  PetscCheck(num_zones == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Only one zone is supported but found %d", num_zones);
  CGNSCall(cg_zone_read(file_num, base, zone, zone_name, sizes));
  CGNSCall(cg_zone_type(file_num, base, zone, &zone_type));
  PetscCall(MeshCartGetGlobalSizes(ns->mesh, &M[0], &M[1], &M[2]));
  PetscCheck(zone_type == CGNS_ENUMV(Structured), PETSC_COMM_SELF, PETSC_ERR_LIB, "Only structured zone is supported for Cartesian mesh");
  for (d = 0; d < dim; ++d) PetscCheck(M[d] == sizes[dim + d], PETSC_COMM_SELF, PETSC_ERR_LIB, "Mesh size %" PetscInt_FMT " does not match CGNS zone size %ld", M[d], (long)sizes[dim + d]);
  CGNSCall(cg_nsols(file_num, base, zone, &num_solutions));
  solution = num_solutions; /* Assume that the last solution is the one we want */
  CGNSCall(cg_nfields(file_num, base, zone, solution, &num_fields));

  /* Load solutions */
  CGNSCall(cg_goto(file_num, base, "Zone_t", zone, "FlowSolution_t", solution, NULL));
  CGNSCall(cg_gridlocation_read(&grid_loc));
  PetscCheck(grid_loc == CGNS_ENUMV(CellCenter), PETSC_COMM_SELF, PETSC_ERR_LIB, "Grid location is not cell-centered in base %d zone %d solution %d", base, zone, solution);

  /* Get solution subvectors */
  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_VELOCITY, &v));
  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_PRESSURE, &p));

  for (d = 0; d < dim; ++d) { PetscCall(DMStagLoadSolution_Private(vdm, v, d, file_num, base, zone, solution, velocity_names[d])); }
  PetscCall(DMStagLoadSolution_Private(sdm, p, 0, file_num, base, zone, solution, pressure_name));
  PetscCall(DMStagLoadSolution_Private(sdm, fsm->phalf, 0, file_num, base, zone, solution, pressure_half_step_name));

  /* Restore solution subvectors */
  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_VELOCITY, &v));
  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_PRESSURE, &p));

  PetscFunctionReturn(PETSC_SUCCESS);
}
