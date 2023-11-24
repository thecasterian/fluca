#include <fluca/private/viewercgnsutils.h>
#include <petsc/private/viewercgnsimpl.h>

PetscErrorCode FlucaViewerCGNSFileOpen_Private(PetscViewer v, int sequence_number) {
    PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)v->data;

    PetscFunctionBegin;

    PetscCheck((!cgns->filename) ^ (sequence_number < 0), PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_INCOMP,
               "Expect either a template filename or non-negative sequence number");
    if (!cgns->filename) {
        char filename_numbered[PETSC_MAX_PATH_LEN];
        // Cast sequence_number so %d can be used also when PetscInt is 64-bit. We could upgrade the format string if
        // users run more than 2B time steps.
        PetscCall(
            PetscSNPrintf(filename_numbered, sizeof(filename_numbered), cgns->filename_template, (int)sequence_number));
        PetscCall(PetscStrallocpy(filename_numbered, &cgns->filename));
    }

    switch (cgns->btype) {
        case FILE_MODE_READ:
            SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "FILE_MODE_READ not yet implemented");
            break;
        case FILE_MODE_WRITE:
#if defined(PETSC_HDF5_HAVE_PARALLEL)
            PetscCallCGNS(cgp_mpi_comm(PetscObjectComm((PetscObject)v)));
            PetscCallCGNS(cgp_open(cgns->filename, CG_MODE_WRITE, &cgns->file_num));
#else
            PetscCallCGNS(cg_open(filename, CG_MODE_WRITE, &cgns->file_num));
#endif
            break;
        case FILE_MODE_UNDEFINED:
            SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ORDER,
                    "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
        default:
            SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Unsupported file mode %s",
                    PetscFileModes[cgns->btype]);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaViewerCGNSWriteStructuredSolution_Private(DM da, Vec v, int file_num, int base, int zone, int sol,
                                                              const char *name) {
    PetscInt dim, xs[3], xm[3], d, cnt, i, j, k;
    const PetscReal **arr2d, ***arr3d;
    cgsize_t rmin[3], rmax[3], rsize;
    int field;
    double *arrraw;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(da, &dim));
    PetscCall(DMDAGetCorners(da, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2]));

    rsize = 1;
    for (d = 0; d < dim; d++) {
        rmin[d] = xs[d] + 1;
        rmax[d] = xs[d] + xm[d];
        rsize *= xm[d];
    }

    PetscCall(PetscMalloc1(rsize, &arrraw));
    switch (dim) {
        case 2:
            PetscCall(DMDAVecGetArrayRead(da, v, &arr2d));
            cnt = 0;
            for (j = rmin[1] - 1; j <= rmax[1] - 1; j++)
                for (i = rmin[0] - 1; i <= rmax[0] - 1; i++)
                    arrraw[cnt++] = arr2d[j][i];
            PetscCall(DMDAVecRestoreArrayRead(da, v, &arr2d));
            break;
        case 3:
            PetscCall(DMDAVecGetArrayRead(da, v, &arr3d));
            cnt = 0;
            for (k = rmin[2] - 1; k <= rmax[2] - 1; k++)
                for (j = rmin[1] - 1; j <= rmax[1] - 1; j++)
                    for (i = rmin[0] - 1; i <= rmax[0] - 1; i++)
                        arrraw[cnt++] = arr3d[k][j][i];
            PetscCall(DMDAVecRestoreArrayRead(da, v, &arr3d));
            break;
        default:
            SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    PetscCallCGNS(cgp_field_write(file_num, base, zone, sol, CGNS_ENUMV(RealDouble), name, &field));
    PetscCallCGNS(cgp_field_write_data(file_num, base, zone, sol, field, rmin, rmax, arrraw));
    PetscCall(PetscFree(arrraw));

    PetscFunctionReturn(PETSC_SUCCESS);
}
