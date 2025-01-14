#include <fluca/private/mesh_cart.h>
#include <fluca/private/flucaviewer_cgns.h>
#include <pcgnslib.h>
#include <petscdmstag.h>

PetscErrorCode MeshView_CartCGNS(Mesh mesh, PetscViewer viewer)
{
  Mesh_Cart             *cart = (Mesh_Cart *)mesh->data;
  PetscViewer_FlucaCGNS *cgv  = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  if (mesh->state < MESH_STATE_SETUP) PetscFunctionReturn(PETSC_SUCCESS);
  if (cgv->file_num && cgv->base) PetscFunctionReturn(PETSC_SUCCESS);

  if (!cgv->file_num) {
    DM       dm;
    PetscInt timestep;

    PetscCall(MeshGetDM(mesh, &dm));
    PetscCall(DMGetOutputSequenceNumber(dm, &timestep, NULL));
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
    cgsize_t            rmin[3], rmax[3], rsize;
    PetscInt            s[3], m[3], d;
    PetscBool           isLastRank[3];
    const PetscScalar **arrcf[3];
    PetscScalar        *x[3]          = {0};
    const char         *coordnames[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
    int                 coord[3];

    PetscCall(DMStagGetCorners(cart->dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
    PetscCall(DMStagGetIsLastRank(cart->dm, &isLastRank[0], &isLastRank[1], &isLastRank[2]));

    for (d = 0; d < mesh->dim; ++d) {
      /* Vertex ownership; note that CGNS uses 1-based index */
      rmin[d] = s[d] + 1;
      rmax[d] = s[d] + m[d] + isLastRank[d];
    }

    rsize = 1;
    for (d = 0; d < mesh->dim; ++d) rsize *= rmax[d] - rmin[d] + 1;

    PetscCall(MeshCartGetCoordinateArraysRead(mesh, &arrcf[0], &arrcf[1], &arrcf[2]));

    for (d = 0; d < mesh->dim; ++d) {
      cgsize_t i[3];
      PetscInt cnt;

      PetscCall(PetscMalloc1(rsize, &x[d]));
      switch (mesh->dim) {
      case 2:
        cnt = 0;
        for (i[1] = rmin[1] - 1; i[1] < rmax[1]; ++i[1])
          for (i[0] = rmin[0] - 1; i[0] < rmax[0]; ++i[0]) {
            x[d][cnt] = arrcf[d][i[d]][0];
            ++cnt;
          }
        break;
      case 3:
        cnt = 0;
        for (i[2] = rmin[2] - 1; i[2] < rmax[2]; ++i[2])
          for (i[1] = rmin[1] - 1; i[1] < rmax[1]; ++i[1])
            for (i[0] = rmin[0] - 1; i[0] < rmax[0]; ++i[0]) {
              x[d][cnt] = arrcf[d][i[d]][0];
              ++cnt;
            }
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension");
      }
    }

    PetscCall(MeshCartRestoreCoordinateArraysRead(mesh, &arrcf[0], &arrcf[1], &arrcf[2]));

    for (d = 0; d < mesh->dim; ++d) {
      CGNSCall(cgp_coord_write(cgv->file_num, cgv->base, cgv->zone, CGNS_ENUMV(RealDouble), coordnames[d], &coord[d]));
      CGNSCall(cgp_coord_write_data(cgv->file_num, cgv->base, cgv->zone, coord[d], rmin, rmax, x[d]));
      PetscCall(PetscFree(x[d]));
    }
  }

  /* Cell info */
  {
    DM          dm;
    PetscInt    xs[3], xm[3], d, i;
    int         solution, field;
    PetscMPIInt rank;
    cgsize_t    rmin[3], rmax[3], rsize;
    int        *e;

    PetscCall(MeshGetDM(mesh, &dm));
    PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], NULL, NULL, NULL));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

    rsize = 1;
    for (d = 0; d < mesh->dim; ++d) {
      rmin[d] = xs[d] + 1;
      rmax[d] = xs[d] + xm[d];
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
