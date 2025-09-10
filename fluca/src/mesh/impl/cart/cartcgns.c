#include <fluca/private/meshcartimpl.h>
#include <fluca/private/flucaviewercgnsimpl.h>
#include <petscdmstag.h>

PetscErrorCode MeshView_CartCGNS(Mesh mesh, PetscViewer viewer)
{
  Mesh_Cart             *cart = (Mesh_Cart *)mesh->data;
  PetscViewer_FlucaCGNS *cgv  = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  if (!mesh->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (cgv->file_num && cgv->base) PetscFunctionReturn(PETSC_SUCCESS);

  if (!cgv->file_num) {
    PetscInt timestep;

    PetscCall(DMGetOutputSequenceNumber(mesh->sdm, &timestep, NULL));
    PetscCall(PetscViewerFileOpen_FlucaCGNS_Internal(viewer, timestep));
  }
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
    int         solution, field;
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

    CGNSCall(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, "CellInfo", CGNS_ENUMV(CellCenter), &solution));
    CGNSCall(cgp_field_write(cgv->file_num, cgv->base, cgv->zone, solution, CGNS_ENUMV(Integer), "Rank", &field));
    CGNSCall(cgp_field_write_data(cgv->file_num, cgv->base, cgv->zone, solution, field, rmin, rmax, e));
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
