#include <fluca/private/mesh_cart.h>
#include <pcgnslib.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/viewercgnsimpl.h>
#include <petscdmstag.h>

PetscErrorCode MeshView_CartCGNS(Mesh mesh, PetscViewer viewer)
{
  Mesh_Cart        *cart = (Mesh_Cart *)mesh->data;
  PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBegin;

  if (mesh->state < MESH_STATE_SETUP) PetscFunctionReturn(PETSC_SUCCESS);
  if (cgns->file_num && cgns->base) PetscFunctionReturn(PETSC_SUCCESS);

  if (!cgns->file_num) {
    DM       dm;
    PetscInt timestep;

    PetscCall(MeshGetDM(mesh, &dm));
    PetscCall(DMGetOutputSequenceNumber(dm, &timestep, NULL));
    PetscCall(PetscViewerCGNSFileOpen_Internal(viewer, timestep));
  }
  PetscCallCGNS(cg_base_write(cgns->file_num, "Base", mesh->dim, mesh->dim, &cgns->base));

  {
    cgsize_t size[9] = {0};
    PetscInt d;

    for (d = 0; d < mesh->dim; ++d) {
      size[d]             = cart->N[d] + 1; /* Number of vertices */
      size[mesh->dim + d] = cart->N[d];     /* Number of elements */
    }
    PetscCallCGNS(cg_zone_write(cgns->file_num, cgns->base, "Zone", size, CGNS_ENUMV(Structured), &cgns->zone));
  }

  {
    cgsize_t          rmin[3], rmax[3], rsize;
    PetscInt          s[3], m[3], d;
    PetscBool         isLastRank[3];
    const PetscReal **arrcf[3];
    PetscReal        *x[3]          = {0};
    const char       *coordnames[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};
    int               coord[3];

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
      PetscCallCGNS(cgp_coord_write(cgns->file_num, cgns->base, cgns->zone, CGNS_ENUMV(RealDouble), coordnames[d], &coord[d]));
      PetscCallCGNS(cgp_coord_write_data(cgns->file_num, cgns->base, cgns->zone, coord[d], rmin, rmax, x[d]));
      PetscCall(PetscFree(x[d]));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
