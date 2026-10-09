#include <fluca/private/meshimpl.h>
#include <fluca/private/flucaviewercgnsimpl.h>
#include <petscdmstag.h>

#define MESH_CGNS_INFO_NAME "FlucaMesh"       /* UserDefinedData_t under the zone */
#define MESH_CGNS_BT_NAME   "DMBoundaryTypes" /* int array, one entry per direction */

/* Write the base, the zone (vertex coordinates and DM boundary types) and the CellInfo solution, once per file */
static PetscErrorCode MeshWriteZone_Cartesian_Private(Mesh mesh, PetscViewer viewer, PetscInt step)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  DM                     dm  = mesh->dm;
  PetscInt               dim = mesh->dim, M[3], x[3], m[3], d, i;
  PetscBool              last[3];
  DMBoundaryType         bt[3];

  PetscFunctionBegin;
  PetscCheck(dim == 2 || dim == 3, PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "CGNS output requires a 2D or 3D mesh");
  if (cgv->file_num && cgv->base) PetscFunctionReturn(PETSC_SUCCESS);
  if (!cgv->file_num) PetscCall(PetscViewerFlucaCGNSFileOpen_Internal(viewer, step));
  CGNSCall(cg_base_write(cgv->file_num, "Base", (int)dim, (int)dim, &cgv->base));

  PetscCall(DMStagGetGlobalSizes(dm, &M[0], &M[1], &M[2]));
  PetscCall(DMStagGetCorners(dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &last[0], &last[1], &last[2]));
  PetscCall(DMStagGetBoundaryTypes(dm, &bt[0], &bt[1], &bt[2]));

  {
    cgsize_t size[9] = {0};

    for (d = 0; d < dim; ++d) {
      size[d]       = M[d] + 1; /* vertices */
      size[dim + d] = M[d];     /* elements */
    }
    CGNSCall(cg_zone_write(cgv->file_num, cgv->base, "Zone", size, CGNS_ENUMV(Structured), &cgv->zone));
  }

  {
    int      bt_int[3];
    cgsize_t n = dim;

    for (d = 0; d < dim; ++d) bt_int[d] = (int)bt[d];
    CGNSCall(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, NULL));
    CGNSCall(cg_user_data_write(MESH_CGNS_INFO_NAME));
    CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", 1, NULL));
    CGNSCall(cg_array_write(MESH_CGNS_BT_NAME, CGNS_ENUMV(Integer), 1, &n, bt_int));
  }

  {
    cgsize_t               rmin[3], rmax[3], rsize = 1, idx[3];
    const PetscScalar    **arr[3] = {NULL, NULL, NULL};
    PetscScalar           *e;
    PetscInt               iprev, cnt;
    CGNS_ENUMT(DataType_t) datatype;
    const char            *coordnames[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
    int                    coord;

    PetscCall(FlucaGetCGNSDataType_Internal(PETSC_SCALAR, &datatype));
    for (d = 0; d < dim; ++d) {
      rmin[d] = x[d] + 1; /* CGNS indices are 1-based; the last rank also owns the last vertex */
      rmax[d] = x[d] + m[d] + (last[d] ? 1 : 0);
      rsize *= rmax[d] - rmin[d] + 1;
    }
    PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
    PetscCall(PetscMalloc1(rsize, &e));
    for (d = 0; d < dim; ++d) {
      cnt = 0;
      if (dim == 2) {
        for (idx[1] = rmin[1] - 1; idx[1] < rmax[1]; ++idx[1])
          for (idx[0] = rmin[0] - 1; idx[0] < rmax[0]; ++idx[0]) e[cnt++] = arr[d][idx[d]][iprev];
      } else {
        for (idx[2] = rmin[2] - 1; idx[2] < rmax[2]; ++idx[2])
          for (idx[1] = rmin[1] - 1; idx[1] < rmax[1]; ++idx[1])
            for (idx[0] = rmin[0] - 1; idx[0] < rmax[0]; ++idx[0]) e[cnt++] = arr[d][idx[d]][iprev];
      }
      CGNSCall(cgp_coord_write(cgv->file_num, cgv->base, cgv->zone, datatype, coordnames[d], &coord));
      CGNSCall(cgp_coord_write_data(cgv->file_num, cgv->base, cgv->zone, coord, rmin, rmax, e));
    }
    PetscCall(PetscFree(e));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
  }

  {
    cgsize_t    rmin[3], rmax[3], rsize = 1;
    int        *e, sol, field;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
    for (d = 0; d < dim; ++d) {
      rmin[d] = x[d] + 1;
      rmax[d] = x[d] + m[d];
      rsize *= rmax[d] - rmin[d] + 1;
    }
    PetscCall(PetscMalloc1(rsize, &e));
    for (i = 0; i < rsize; ++i) e[i] = rank;
    CGNSCall(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, "CellInfo", CGNS_ENUMV(CellCenter), &sol));
    CGNSCall(cgp_field_write(cgv->file_num, cgv->base, cgv->zone, sol, CGNS_ENUMV(Integer), "Rank", &field));
    CGNSCall(cgp_field_write_data(cgv->file_num, cgv->base, cgv->zone, sol, field, rmin, rmax, e));
    PetscCall(PetscFree(e));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView_Cartesian_CGNS(Mesh mesh, PetscViewer viewer)
{
  PetscInt step;

  PetscFunctionBegin;
  PetscCheck(mesh->dm, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "DM not set. Call MeshSetDM() or MeshLoad() first");
  PetscCall(DMGetOutputSequenceNumber(mesh->dm, &step, NULL));
  PetscCall(MeshWriteZone_Cartesian_Private(mesh, viewer, step < 0 ? 0 : step));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Vertex i of a direction with loaded vertices xv[0..N], extended periodically or by reflection about the ends */
static PetscScalar MeshCartesianVertex_Private(const PetscScalar xv[], PetscInt N, DMBoundaryType bt, PetscInt i)
{
  PetscInt q;

  if (i >= 0 && i <= N) return xv[i];
  if (bt == DM_BOUNDARY_PERIODIC) {
    q = i >= 0 ? i / N : -((-i + N - 1) / N);
    return xv[i - q * N] + (PetscReal)q * (xv[N] - xv[0]);
  }
  return i < 0 ? 2. * xv[0] - xv[-i] : 2. * xv[N] - xv[2 * N - i];
}

PetscErrorCode MeshLoad_Cartesian_CGNS(Mesh mesh, PetscViewer viewer)
{
  PetscViewer_FlucaCGNS *cgv  = (PetscViewer_FlucaCGNS *)viewer->data;
  MPI_Comm               comm = PetscObjectComm((PetscObject)mesh);
  const int              base = 1, zone = 1;
  int                    num_bases, num_zones, num_coords, num_user_data, cell_dim, phys_dim, u, d;
  char                   name[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(ZoneType_t) zone_type;
  CGNS_ENUMT(DataType_t) datatype, file_datatype;
  cgsize_t               sizes[9];
  PetscInt               N[3]   = {1, 1, 1}, gx[3], gm[3], iprev, ielem, i;
  DMBoundaryType         bt[3]  = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscScalar           *xv[3]  = {NULL, NULL, NULL};
  PetscScalar          **arr[3] = {NULL, NULL, NULL};
  DM                     dm;
  PetscBool              found = PETSC_FALSE;

  PetscFunctionBegin;
  CGNSCall(cg_nbases(cgv->file_num, &num_bases));
  PetscCheck(num_bases == 1, comm, PETSC_ERR_FILE_UNEXPECTED, "Only one base is supported");
  CGNSCall(cg_base_read(cgv->file_num, base, name, &cell_dim, &phys_dim));
  PetscCheck(cell_dim == 2 || cell_dim == 3, comm, PETSC_ERR_SUP, "CGNS input requires a 2D or 3D mesh");
  CGNSCall(cg_nzones(cgv->file_num, base, &num_zones));
  PetscCheck(num_zones == 1, comm, PETSC_ERR_FILE_UNEXPECTED, "Only one zone is supported");
  CGNSCall(cg_zone_read(cgv->file_num, base, zone, name, sizes));
  CGNSCall(cg_zone_type(cgv->file_num, base, zone, &zone_type));
  PetscCheck(zone_type == CGNS_ENUMV(Structured), comm, PETSC_ERR_FILE_UNEXPECTED, "Only structured zones are supported");
  CGNSCall(cg_ncoords(cgv->file_num, base, zone, &num_coords));
  PetscCheck(num_coords == cell_dim, comm, PETSC_ERR_FILE_UNEXPECTED, "Number of coordinates does not match the cell dimension");
  PetscCall(FlucaGetCGNSDataType_Internal(PETSC_SCALAR, &datatype));

  for (d = 0; d < cell_dim; ++d) {
    cgsize_t rmin[3] = {1, 1, 1}, rmax[3] = {1, 1, 1};

    N[d] = (PetscInt)sizes[cell_dim + d];
    PetscCheck(N[d] >= 2, comm, PETSC_ERR_FILE_UNEXPECTED, "MeshLoad needs at least 2 elements per direction");
    CGNSCall(cg_coord_info(cgv->file_num, base, zone, d + 1, &file_datatype, name));
    PetscCheck(file_datatype == datatype, comm, PETSC_ERR_FILE_UNEXPECTED, "Coordinate %s is not stored in the PetscScalar precision", name);
    rmax[d] = sizes[d];
    PetscCall(PetscMalloc1(sizes[d], &xv[d]));
    CGNSCall(cgp_coord_read_data(cgv->file_num, base, zone, d + 1, rmin, rmax, xv[d]));
  }

  CGNSCall(cg_goto(cgv->file_num, base, "Zone_t", zone, NULL));
  CGNSCall(cg_nuser_data(&num_user_data));
  for (u = 1; u <= num_user_data && !found; ++u) {
    CGNSCall(cg_user_data_read(u, name));
    PetscCall(PetscStrcmp(name, MESH_CGNS_INFO_NAME, &found));
    if (found) {
      int bt_int[3];

      CGNSCall(cg_gorel(cgv->file_num, "UserDefinedData_t", u, NULL));
      CGNSCall(cg_array_read_as(1, CGNS_ENUMV(Integer), bt_int));
      for (d = 0; d < cell_dim; ++d) {
        PetscCheck(bt_int[d] >= (int)DM_BOUNDARY_NONE && bt_int[d] <= (int)DM_BOUNDARY_TWIST, comm, PETSC_ERR_FILE_UNEXPECTED, "Invalid DM boundary type %d for direction %" PetscInt_FMT, bt_int[d], d);
        bt[d] = (DMBoundaryType)bt_int[d];
      }
    }
  }
  /* Files without the FlucaMesh node (written before boundary types were stored) load as non-periodic */

  if (cell_dim == 2) PetscCall(DMStagCreate2d(comm, bt[0], bt[1], N[0], N[1], PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  else PetscCall(DMStagCreate3d(comm, bt[0], bt[1], bt[2], N[0], N[1], N[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));
  PetscCall(DMStagGetGhostCorners(dm, &gx[0], &gx[1], &gx[2], &gm[0], &gm[1], &gm[2]));
  PetscCall(DMStagGetProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielem));
  for (d = 0; d < cell_dim; ++d)
    for (i = gx[d]; i < gx[d] + gm[d]; ++i) {
      arr[d][i][iprev] = MeshCartesianVertex_Private(xv[d], N[d], bt[d], i);
      arr[d][i][ielem] = 0.5 * (MeshCartesianVertex_Private(xv[d], N[d], bt[d], i) + MeshCartesianVertex_Private(xv[d], N[d], bt[d], i + 1));
    }
  PetscCall(DMStagRestoreProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  for (d = 0; d < cell_dim; ++d) PetscCall(PetscFree(xv[d]));

  PetscCall(DMDestroy(&mesh->dm));
  mesh->dm  = dm;
  mesh->dim = cell_dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}
