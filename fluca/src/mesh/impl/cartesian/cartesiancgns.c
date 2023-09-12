#include <fluca/private/mesh_cartesian.h>
#include <fluca/private/meshimpl.h>
#include <fluca/private/viewercgnsutils.h>
#include <flucamap.h>
#include <pcgnslib.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/viewercgnsimpl.h>
#include <petscdmstag.h>

PetscErrorCode MeshView_CartesianCGNS(Mesh mesh, PetscViewer v) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)v->data;
    FlucaMap map;
    PetscContainer viewerinfo_container;
    ViewerCGNSInfo *viewerinfo;

    PetscFunctionBegin;

    PetscCall(PetscObjectQuery((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject *)&map));
    if (!map) {
        PetscCall(FlucaMapCreate(PetscObjectComm((PetscObject)mesh), &map, NULL, NULL));
        PetscCall(PetscObjectCompose((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject)map));
        PetscCall(FlucaMapDestroy(&map));
        PetscCall(PetscObjectQuery((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject *)&map));
    }

    if (!cgns->file_num)
        PetscCall(PetscViewerCGNSFileOpen_Internal(v, mesh->seqnum));
    if (cgns->base == 0)
        PetscCallCGNS(cg_base_write(cgns->file_num, "Base", mesh->dim, mesh->dim, &cgns->base));

    PetscCall(FlucaMapGetValue(map, (PetscObject)mesh, (PetscObject *)&viewerinfo_container));
    if (viewerinfo_container)
        /* mesh is already written */
        PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)mesh), &viewerinfo_container));
    PetscCall(PetscMalloc1(1, &viewerinfo));
    PetscCall(PetscContainerSetPointer(viewerinfo_container, viewerinfo));
    PetscCall(PetscContainerSetUserDestroy(viewerinfo_container, PetscContainerUserDestroyDefault));
    PetscCall(FlucaMapInsert(map, (PetscObject)mesh, (PetscObject)viewerinfo_container));
    PetscCall(PetscContainerDestroy(&viewerinfo_container));

    {
        cgsize_t size[9] = {0};
        PetscInt d;

        for (d = 0; d < mesh->dim; d++) {
            size[d] = cart->N[d] + 1;         /* Number of vertices */
            size[mesh->dim + d] = cart->N[d]; /* Number of elements */
        }
        PetscCallCGNS(
            cg_zone_write(cgns->file_num, cgns->base, "Zone", size, CGNS_ENUMV(Structured), &viewerinfo->zone));
    }

    {
        cgsize_t rmin[3], rmax[3], rsize;
        PetscInt d;
        PetscReal *x[3] = {0};
        const char *coordnames[3] = {"CoordinateX", "CoordinateY", "CoordinateZ"};

        for (d = 0; d < mesh->dim; d++) {
            DM subfdm;
            PetscInt xs, xm;

            PetscCall(DMProductGetDM(cart->cfdm, d, &subfdm));
            PetscCall(DMStagGetCorners(subfdm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));

            /* Vertex ownership; note that CGNS uses 1-based index */
            rmin[d] = xs + 1;
            rmax[d] = xs + xm + (cart->rank[d] == cart->nRanks[d] - 1);
        }

        rsize = 1;
        for (d = 0; d < mesh->dim; d++)
            rsize *= rmax[d] - rmin[d] + 1;

        for (d = 0; d < mesh->dim; d++) {
            DM subfdm;
            const PetscReal **arrcf;
            cgsize_t i[3];
            PetscInt cnt, ileft;

            PetscCall(DMProductGetDM(cart->cfdm, d, &subfdm));
            PetscCall(DMStagVecGetArrayRead(subfdm, cart->cf[d], &arrcf));
            PetscCall(DMStagGetLocationSlot(subfdm, DMSTAG_LEFT, 0, &ileft));
            PetscCall(PetscMalloc1(rsize, &x[d]));
            switch (mesh->dim) {
                case 2:
                    cnt = 0;
                    for (i[1] = rmin[1] - 1; i[1] < rmax[1]; i[1]++)
                        for (i[0] = rmin[0] - 1; i[0] < rmax[0]; i[0]++)
                            x[d][cnt++] = arrcf[i[d]][ileft];
                    break;
                case 3:
                    cnt = 0;
                    for (i[2] = rmin[2] - 1; i[2] < rmax[2]; i[2]++)
                        for (i[1] = rmin[1] - 1; i[1] < rmax[1]; i[1]++)
                            for (i[0] = rmin[0] - 1; i[0] < rmax[0]; i[0]++)
                                x[d][cnt++] = arrcf[i[d]][ileft];
                    break;
                default:
                    SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension");
            }
            PetscCall(DMStagVecRestoreArrayRead(subfdm, cart->cf[d], &arrcf));
        }

        for (d = 0; d < mesh->dim; d++) {
            PetscCallCGNS(cgp_coord_write(cgns->file_num, cgns->base, viewerinfo->zone, CGNS_ENUMV(RealDouble),
                                          coordnames[d], &viewerinfo->coord[d]));
            PetscCallCGNS(cgp_coord_write_data(cgns->file_num, cgns->base, viewerinfo->zone, viewerinfo->coord[d], rmin,
                                               rmax, x[d]));
            PetscCall(PetscFree(x[d]));
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
