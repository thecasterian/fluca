#include <fluca/private/meshcartimpl.h>
#include <fluca/private/flucaviewercgnsimpl.h>
#include <petscdmstag.h>

PetscErrorCode MeshView_Cart_CGNS(Mesh mesh, PetscViewer viewer)
{
  Mesh_Cart             *cart = (Mesh_Cart *)mesh->data;
  PetscViewer_FlucaCGNS *cgv  = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  if (!mesh->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (cgv->file_num && cgv->base) PetscFunctionReturn(PETSC_SUCCESS);

  if (!cgv->file_num) PetscCall(PetscViewerFlucaCGNSFileOpen_Internal(viewer, mesh->outputseqnum));
  CGNSCall(cg_base_write(cgv->file_num, "Base", mesh->dim, mesh->dim, &cgv->base));

  {
    cgsize_t size[9] = {0};
    PetscInt d;

    for (d = 0; d < mesh->dim; ++d) {
      size[d]             = cart->N[d] + 1; /* Number of vertices */
      size[mesh->dim + d] = cart->N[d];     /* Number of elements */
    }
    CGNSCall(cg_zone_write(cgv->file_num, cgv->base, "Zone", size, CGNS_ENUMV(Structured), &cgv->zone));
  }

  {
    cgsize_t               rmin[3], rmax[3], rsize;
    PetscInt               x[3], m[3], d;
    PetscBool              isLastRank[3];
    const PetscScalar    **arrcf[3];
    PetscScalar           *e[3] = {0};
    CGNS_ENUMT(DataType_t) datatype;
    const char            *coordnames[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
    int                    coord[3];

    PetscCall(DMStagGetCorners(mesh->sdm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
    PetscCall(DMStagGetIsLastRank(mesh->sdm, &isLastRank[0], &isLastRank[1], &isLastRank[2]));
    PetscCall(FlucaGetCGNSDataType_Internal(PETSC_SCALAR, &datatype));

    for (d = 0; d < mesh->dim; ++d) {
      /* Vertex ownership; note that CGNS uses 1-based index */
      rmin[d] = x[d] + 1;
      rmax[d] = x[d] + m[d] + isLastRank[d];
    }

    rsize = 1;
    for (d = 0; d < mesh->dim; ++d) rsize *= rmax[d] - rmin[d] + 1;

    PetscCall(MeshCartGetCoordinateArraysRead(mesh, &arrcf[0], &arrcf[1], &arrcf[2]));

    for (d = 0; d < mesh->dim; ++d) {
      cgsize_t i[3];
      PetscInt cnt;

      PetscCall(PetscMalloc1(rsize, &e[d]));
      switch (mesh->dim) {
      case 2:
        cnt = 0;
        for (i[1] = rmin[1] - 1; i[1] < rmax[1]; ++i[1])
          for (i[0] = rmin[0] - 1; i[0] < rmax[0]; ++i[0]) {
            e[d][cnt] = arrcf[d][i[d]][0];
            ++cnt;
          }
        break;
      case 3:
        cnt = 0;
        for (i[2] = rmin[2] - 1; i[2] < rmax[2]; ++i[2])
          for (i[1] = rmin[1] - 1; i[1] < rmax[1]; ++i[1])
            for (i[0] = rmin[0] - 1; i[0] < rmax[0]; ++i[0]) {
              e[d][cnt] = arrcf[d][i[d]][0];
              ++cnt;
            }
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension");
      }
    }

    PetscCall(MeshCartRestoreCoordinateArraysRead(mesh, &arrcf[0], &arrcf[1], &arrcf[2]));

    for (d = 0; d < mesh->dim; ++d) {
      CGNSCall(cgp_coord_write(cgv->file_num, cgv->base, cgv->zone, datatype, coordnames[d], &coord[d]));
      CGNSCall(cgp_coord_write_data(cgv->file_num, cgv->base, cgv->zone, coord[d], rmin, rmax, e[d]));
      PetscCall(PetscFree(e[d]));
    }
  }

  /* Cell info */
  {
    PetscInt    x[3], m[3], d, i;
    int         sol, field;
    PetscMPIInt rank;
    cgsize_t    rmin[3], rmax[3], rsize;
    int        *e;

    PetscCall(DMStagGetCorners(mesh->sdm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mesh->sdm), &rank));

    rsize = 1;
    for (d = 0; d < mesh->dim; ++d) {
      rmin[d] = x[d] + 1;
      rmax[d] = x[d] + m[d];
      rsize *= rmax[d] - rmin[d] + 1;
    }
    PetscCall(PetscMalloc1(rsize, &e));
    for (i = 0; i < rsize; ++i) e[i] = rank;

    CGNSCall(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, "CellInfo", CGNS_ENUMV(CellCenter), &sol));
    CGNSCall(cgp_field_write(cgv->file_num, cgv->base, cgv->zone, sol, CGNS_ENUMV(Integer), "Rank", &field));
    CGNSCall(cgp_field_write_data(cgv->file_num, cgv->base, cgv->zone, sol, field, rmin, rmax, e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

PetscErrorCode VecView_Cart_Local_CGNS(Vec v, PetscViewer viewer)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  Mesh                   mesh;
  DM                     dm;
  PetscInt               dim, dof[4], step, d;
  PetscBool              cc, fc;
  const char            *vec_name;
  PetscReal              time;
  char                   sol_name[PETSC_MAX_PATH_LEN];
  char                   field_name[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)v, "Fluca_Mesh", (PetscObject *)&mesh));
  PetscCall(VecGetDM(v, &dm));
  PetscCheck(mesh && dm, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a Mesh");

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetDOF(dm, &dof[0], &dof[1], &dof[2], &dof[3]));
  switch (dim) {
  case 2:
    cc = dof[0] == 0 && dof[1] == 0 && dof[2] > 0;
    fc = dof[0] == 0 && dof[1] > 0 && dof[2] == 0;
    break;
  case 3:
    cc = dof[0] == 0 && dof[1] == 0 && dof[2] == 0 && dof[3] > 0;
    fc = dof[0] == 0 && dof[1] == 0 && dof[2] > 0 && dof[3] == 0;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension");
  }
  PetscCheck(cc || fc, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not cell-centered nor face-centered");

  if (cgv->zone) {
    // TODO: check compatibility with Mesh written
  }

  PetscCall(PetscObjectGetName((PetscObject)v, &vec_name));
  PetscCall(MeshGetOutputSequenceNumber(mesh, &step, &time));
  if (step < 0) {
    step = 0;
    time = 0.;
  }

  if (cgv->last_step != step) {
    size_t    *step_slot;
    PetscReal *time_slot;

    PetscCall(PetscViewerFlucaCGNSCheckBatch_Internal(viewer));
    cgv->sol = 0;
    if (!cgv->zone) PetscCall(MeshView(mesh, viewer));
    if (!cgv->output_steps) PetscCall(PetscSegBufferCreate(sizeof(size_t), 20, &cgv->output_steps));
    if (!cgv->output_times) PetscCall(PetscSegBufferCreate(sizeof(PetscReal), 20, &cgv->output_times));

    PetscCall(PetscSegBufferGet(cgv->output_steps, 1, &step_slot));
    PetscCall(PetscSegBufferGet(cgv->output_times, 1, &time_slot));
    *step_slot = step;
    *time_slot = time;

    cgv->last_step = step;
  }

  if (!cgv->sol) {
    const char *const                face_sol_names[3]           = {"IFaceCenteredSolution", "JFaceCenteredSolution", "KFaceCenteredSolution"};
    const CGNS_ENUMT(GridLocation_t) face_sol_grid_locationss[3] = {CGNS_ENUMV(IFaceCenter), CGNS_ENUMV(JFaceCenter), CGNS_ENUMV(KFaceCenter)};

    /* Solution tree structure:
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
     * Be careful to the order of writing user data */
    PetscCall(PetscSNPrintf(sol_name, sizeof(sol_name), "FlowSolution%" PetscInt_FMT, step));
    CGNSCall(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, sol_name, CGNS_ENUMV(CellCenter), &cgv->sol));
    CGNSCall(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "FlowSolution_t", cgv->sol, NULL));
    for (d = 0; d < dim; ++d) {
      CGNSCall(cg_user_data_write(face_sol_names[d]));
      CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", d + 1, NULL));
      CGNSCall(cg_gridlocation_write(face_sol_grid_locationss[d]));
      CGNSCall(cg_gorel(cgv->file_num, "..", 0, NULL));
    }
  }

  if (cc) {
    if (dof[dim] == 1) {
      PetscCall(DMStagWriteCellCenteredSolution_Private(dm, v, 0, cgv->file_num, cgv->base, cgv->zone, cgv->sol, vec_name));
    } else {
      for (d = 0; d < dof[dim]; ++d) {
        PetscCall(PetscSNPrintf(field_name, sizeof(field_name), "%s%c", vec_name, 'X' + d));
        PetscCall(DMStagWriteCellCenteredSolution_Private(dm, v, d, cgv->file_num, cgv->base, cgv->zone, cgv->sol, field_name));
      }
    }
  } else if (fc) {
    if (dof[dim - 1] == 1) {
      PetscCall(DMStagWriteFaceCenteredSolution_Private(dm, v, 0, cgv->file_num, cgv->base, cgv->zone, cgv->sol, vec_name));
    } else {
      for (d = 0; d < dof[dim - 1]; ++d) {
        PetscCall(PetscSNPrintf(field_name, sizeof(field_name), "%s%c", vec_name, 'X' + d));
        PetscCall(DMStagWriteFaceCenteredSolution_Private(dm, v, d, cgv->file_num, cgv->base, cgv->zone, cgv->sol, field_name));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartCreateCGNS(MPI_Comm comm, PetscInt file_num, Mesh *mesh)
{
  const int              base = 1, zone = 1;
  int                    num_bases, num_zones, num_coords;
  int                    cell_dim, phys_dim;
  int                    d;
  char                   base_name[CGIO_MAX_NAME_LENGTH + 1];
  char                   zone_name[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(ZoneType_t) zone_type;
  cgsize_t               sizes[9];
  PetscScalar           *arr_coord[3];

  PetscFunctionBegin;
  PetscAssertPointer(mesh, 3);

  PetscCall(MeshCreate(comm, mesh));
  PetscCall(MeshSetType(*mesh, MESHCART));

  /* Read CGNS file. */
  CGNSCall(cg_nbases(file_num, &num_bases));
  PetscCheck(num_bases == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Only one base is supported");
  CGNSCall(cg_base_read(file_num, base, base_name, &cell_dim, &phys_dim));
  CGNSCall(cg_nzones(file_num, base, &num_zones));
  PetscCheck(num_zones == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Only one zone is supported");
  CGNSCall(cg_zone_read(file_num, base, zone, zone_name, sizes));
  CGNSCall(cg_zone_type(file_num, base, zone, &zone_type));
  PetscCheck(zone_type == CGNS_ENUMV(Structured), PETSC_COMM_SELF, PETSC_ERR_LIB, "Only structured zone is supported");

  CGNSCall(cg_ncoords(file_num, base, zone, &num_coords));
  PetscCheck(num_coords == cell_dim, PETSC_COMM_SELF, PETSC_ERR_LIB, "Number of coordinates does not match cell dimension");
  for (d = 0; d < cell_dim; ++d) {
    cgsize_t rmin[3] = {1, 1, 1}, rmax[3] = {1, 1, 1};

    rmax[d] = sizes[d];
    PetscCall(PetscMalloc1(sizes[d], &arr_coord[d]));
    CGNSCall(cgp_coord_read_data(file_num, base, zone, d + 1, rmin, rmax, arr_coord[d]));
  }

  /* Build mesh. */
  PetscCall(MeshSetDimension(*mesh, cell_dim));
  PetscCall(MeshCartSetBoundaryTypes(*mesh, MESHCART_BOUNDARY_NONE, MESHCART_BOUNDARY_NONE, MESHCART_BOUNDARY_NONE));
  PetscCall(MeshCartSetGlobalSizes(*mesh, sizes[cell_dim], sizes[cell_dim + 1], cell_dim == 3 ? sizes[cell_dim + 2] : 1));
  PetscCall(MeshCartSetNumRanks(*mesh, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MeshCartSetOwnershipRanges(*mesh, NULL, NULL, NULL));

  PetscCall(MeshSetUp(*mesh));

  {
    PetscInt      x[3], m[3];
    PetscScalar **a[3];
    PetscInt      i, iprev;

    PetscCall(MeshCartGetCorners(*mesh, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2]));
    PetscCall(MeshCartGetCoordinateArrays(*mesh, &a[0], &a[1], &a[2]));
    PetscCall(MeshCartGetCoordinateLocationSlot(*mesh, MESHCART_PREV, &iprev));
    for (d = 0; d < cell_dim; ++d) {
      for (i = x[d]; i <= x[d] + m[d]; ++i) a[d][i][iprev] = arr_coord[d][i];
      PetscCall(PetscFree(arr_coord[d]));
    }
    PetscCall(MeshCartRestoreCoordinateArrays(*mesh, &a[0], &a[1], &a[2]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartCreateCGNSFromFile(MPI_Comm comm, const char filename[], Mesh *mesh)
{
  int file_num = -1;

  PetscFunctionBegin;
  PetscAssertPointer(filename, 2);
  PetscAssertPointer(mesh, 3);

  CGNSCall(cgp_mpi_comm(comm));
  CGNSCall(cgp_open(filename, CG_MODE_READ, &file_num));
  PetscCheck(file_num > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cgp_open(\"%s\", ...) did not return a valid file number", filename);
  PetscCall(MeshCartCreateCGNS(comm, file_num, mesh));
  CGNSCall(cgp_close(file_num));
  PetscFunctionReturn(PETSC_SUCCESS);
}
