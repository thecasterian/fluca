#include <flucamap.h>
#include <impl/cgnsviewerinfo.h>
#include <impl/solimpl.h>
#include <pcgnslib.h>
#include <petsc/private/viewercgnsimpl.h>
#include <petscdmda.h>

static PetscErrorCode SolView_CGNSCartesian_WriteVec(DM dm, Vec v, int file_num, int base, int zone, int sol,
                                                     const char *name) {
    PetscInt dim, xs[3], xm[3], d, cnt, i, j, k;
    const PetscReal **arr2d, ***arr3d;
    cgsize_t rmin[3], rmax[3], rsize;
    int field;
    double *arrraw;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMDAGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2]));

    rsize = 1;
    for (d = 0; d < dim; d++) {
        rmin[d] = xs[d] + 1;
        rmax[d] = xs[d] + xm[d];
        rsize *= xm[d];
    }

    PetscCall(PetscMalloc1(rsize, &arrraw));
    switch (dim) {
        case 2:
            PetscCall(DMDAVecGetArrayRead(dm, v, &arr2d));
            cnt = 0;
            for (j = rmin[1] - 1; j <= rmax[1] - 1; j++)
                for (i = rmin[0] - 1; i <= rmax[0] - 1; i++)
                    arrraw[cnt++] = arr2d[j][i];
            PetscCall(DMDAVecRestoreArrayRead(dm, v, &arr2d));
            break;
        case 3:
            PetscCall(DMDAVecGetArrayRead(dm, v, &arr3d));
            cnt = 0;
            for (k = rmin[2] - 1; k <= rmax[2] - 1; k++)
                for (j = rmin[1] - 1; j <= rmax[1] - 1; j++)
                    for (i = rmin[0] - 1; i <= rmax[0] - 1; i++)
                        arrraw[cnt++] = arr3d[k][j][i];
            PetscCall(DMDAVecRestoreArrayRead(dm, v, &arr3d));
            break;
        default:
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    PetscCallCGNS(cgp_field_write(file_num, base, zone, sol, CGNS_ENUMV(RealDouble), name, &field));
    PetscCallCGNS(cgp_field_write_data(file_num, base, zone, sol, field, rmin, rmax, arrraw));
    PetscCall(PetscFree(arrraw));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolView_CGNSCartesian(Sol sol, PetscViewer v) {
    PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)v->data;
    PetscInt dim;
    FlucaMap map;
    PetscContainer viewerinfo_container;
    CGNSViewerInfo *viewerinfo;
    DM dm;

    PetscFunctionBegin;

    PetscCall(MeshView(sol->mesh, v));

    PetscCall(MeshGetDim(sol->mesh, &dim));

    PetscCall(PetscObjectQuery((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject *)&map));
    PetscCall(FlucaMapGetValue(map, (PetscObject)sol->mesh, (PetscObject *)&viewerinfo_container));
    PetscCall(PetscContainerGetPointer(viewerinfo_container, (void **)&viewerinfo));

    PetscCallCGNS(cg_sol_write(cgns->file_num, cgns->base, viewerinfo->zone, "Solution", CGNS_ENUMV(CellCenter),
                               &viewerinfo->sol));

    PetscCall(MeshGetDM(sol->mesh, &dm));
    PetscCall(SolView_CGNSCartesian_WriteVec(dm, sol->u, cgns->file_num, cgns->base, viewerinfo->zone, viewerinfo->sol,
                                             "VelocityX"));
    PetscCall(SolView_CGNSCartesian_WriteVec(dm, sol->v, cgns->file_num, cgns->base, viewerinfo->zone, viewerinfo->sol,
                                             "VelocityY"));
    if (dim > 2)
        PetscCall(SolView_CGNSCartesian_WriteVec(dm, sol->w, cgns->file_num, cgns->base, viewerinfo->zone,
                                                 viewerinfo->sol, "VelocityZ"));
    PetscCall(SolView_CGNSCartesian_WriteVec(dm, sol->p, cgns->file_num, cgns->base, viewerinfo->zone, viewerinfo->sol,
                                             "Pressure"));

    PetscFunctionReturn(PETSC_SUCCESS);
}