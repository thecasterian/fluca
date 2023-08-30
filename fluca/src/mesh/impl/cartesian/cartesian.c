#include <impl/meshimpl.h>
#include <mesh/impl/cartesian/cartesian.h>
#include <petsc/private/petscimpl.h>
#include <petscdmda.h>
#include <petscdmstag.h>

extern PetscErrorCode MeshView_CartesianCGNS(Mesh mesh, PetscViewer v);

static PetscErrorCode MeshCartesianCreateCoordinate(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    MPI_Comm comm;
    DMBoundaryType bnd = DM_BOUNDARY_GHOSTED;
    PetscInt d, s = 1, daDof = 1, stagDof0 = 1, stagDof1 = 0;

    PetscFunctionBegin;

    PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

    PetscCall(DMCreate(comm, &cart->cdm));
    PetscCall(DMSetType(cart->cdm, DMPRODUCT));
    PetscCall(DMSetDimension(cart->cdm, mesh->dim));
    PetscCall(DMSetUp(cart->cdm));
    PetscCall(DMCreate(comm, &cart->cfdm));
    PetscCall(DMSetType(cart->cfdm, DMPRODUCT));
    PetscCall(DMSetDimension(cart->cfdm, mesh->dim));
    PetscCall(DMSetUp(cart->cfdm));

    for (d = 0; d < mesh->dim; d++) {
        DM subdm, subfdm;
        MPI_Comm subcomm;
        PetscMPIInt color, key = 0;

        switch (d) {
            case 0:
                color = cart->rank[1] + (mesh->dim > 2 ? cart->nRanks[1] * cart->rank[2] : 0);
                break;
            case 1:
                color = cart->rank[0] + (mesh->dim > 2 ? cart->nRanks[0] * cart->rank[2] : 0);
                break;
            case 2:
                color = cart->rank[0] + cart->nRanks[0] * cart->rank[1];
                break;
            default:
                SETERRQ(comm, PETSC_ERR_SUP, "Unsupported dimension index %" PetscInt_FMT, d);
        }
        PetscCallMPI(MPI_Comm_split(comm, color, key, &subcomm));

        PetscCall(DMDACreate1d(subcomm, bnd, cart->N[d], daDof, s, cart->l[d], &subdm));
        PetscCall(DMSetUp(subdm));
        PetscCall(
            DMStagCreate1d(subcomm, bnd, cart->N[d], stagDof0, stagDof1, DMSTAG_STENCIL_BOX, s, cart->l[d], &subfdm));
        PetscCall(DMSetUp(subfdm));

        PetscCall(DMProductSetDM(cart->cdm, d, subdm));
        PetscCall(DMProductSetDimensionIndex(cart->cdm, d, 0));
        PetscCall(DMProductSetDM(cart->cfdm, d, subfdm));
        PetscCall(DMProductSetDimensionIndex(cart->cfdm, d, 0));

        PetscCall(DMCreateLocalVector(subdm, &cart->c[d]));
        PetscCall(DMCreateLocalVector(subfdm, &cart->cf[d]));

        PetscCall(DMDestroy(&subdm));
        PetscCall(DMDestroy(&subfdm));
        PetscCallMPI(MPI_Comm_free(&subcomm));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetUp_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    MPI_Comm comm;
    PetscMPIInt rank;
    DMBoundaryType bndx, bndy, bndz;
    PetscInt s = 1, daDof = 1, stagDof0, stagDof1, stagDof2, stagDof3;
    const PetscInt *lx, *ly, *lz;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndTypes[0], &bndx));
    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndTypes[1], &bndy));
    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndTypes[2], &bndz));

    /* Allocate DMs */
    switch (mesh->dim) {
        case 2:
            PetscCall(DMDACreate2d(comm, bndx, bndy, DMDA_STENCIL_STAR, cart->N[0], cart->N[1], cart->nRanks[0],
                                   cart->nRanks[1], daDof, s, cart->l[0], cart->l[1], &cart->dm));
            break;
        case 3:
            PetscCall(DMDACreate3d(comm, bndx, bndy, bndz, DMDA_STENCIL_STAR, cart->N[0], cart->N[1], cart->N[2],
                                   cart->nRanks[0], cart->nRanks[1], cart->nRanks[2], daDof, s, cart->l[0], cart->l[1],
                                   cart->l[2], &cart->dm));
            break;
        default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
    }
    PetscCall(DMSetUp(cart->dm));

    PetscCall(DMDAGetInfo(cart->dm, NULL, NULL, NULL, NULL, &cart->nRanks[0], &cart->nRanks[1], &cart->nRanks[2], NULL,
                          NULL, NULL, NULL, NULL, NULL));
    if (!cart->l[0])
        PetscCall(PetscMalloc1(cart->nRanks[0], &cart->l[0]));
    if (!cart->l[1])
        PetscCall(PetscMalloc1(cart->nRanks[1], &cart->l[1]));
    if (mesh->dim > 2 && !cart->l[2])
        PetscCall(PetscMalloc1(cart->nRanks[2], &cart->l[2]));
    PetscCall(DMDAGetOwnershipRanges(cart->dm, &lx, &ly, &lz));
    PetscCall(PetscArraycpy(cart->l[0], lx, cart->nRanks[0]));
    PetscCall(PetscArraycpy(cart->l[1], ly, cart->nRanks[1]));
    if (mesh->dim > 2)
        PetscCall(PetscArraycpy(cart->l[2], lz, cart->nRanks[2]));

    switch (mesh->dim) {
        case 2:
            stagDof0 = 0;
            stagDof1 = 1;
            stagDof2 = 0;
            PetscCall(DMStagCreate2d(comm, bndx, bndy, cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1],
                                     stagDof0, stagDof1, stagDof2, DMSTAG_STENCIL_STAR, s, cart->l[0], cart->l[1],
                                     &cart->fdm));
            break;
        case 3:
            stagDof0 = 0;
            stagDof1 = 0;
            stagDof2 = 1;
            stagDof3 = 0;
            PetscCall(DMStagCreate3d(comm, bndx, bndy, bndz, cart->N[0], cart->N[1], cart->N[2], cart->nRanks[0],
                                     cart->nRanks[1], cart->nRanks[2], stagDof0, stagDof1, stagDof2, stagDof3,
                                     DMSTAG_STENCIL_STAR, s, cart->l[0], cart->l[1], cart->l[2], &cart->fdm));
            break;
        default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
    }
    PetscCall(DMSetUp(cart->fdm));

    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    switch (mesh->dim) {
        case 2:
            cart->rank[0] = rank % cart->nRanks[0];
            cart->rank[1] = rank / cart->nRanks[0];
            cart->rank[2] = 0;
            break;
        case 3:
            cart->rank[0] = rank % cart->nRanks[0];
            cart->rank[1] = (rank % (cart->nRanks[0] * cart->nRanks[1])) / cart->nRanks[0];
            cart->rank[2] = rank / (cart->nRanks[0] * cart->nRanks[1]);
            break;
        default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
    }

    PetscCall(MeshCartesianCreateCoordinate(mesh));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscInt d;

    PetscFunctionBegin;

    for (d = 0; d < MESH_MAX_DIM; d++)
        PetscCall(PetscFree(cart->l[d]));
    PetscCall(DMDestroy(&cart->dm));
    PetscCall(DMDestroy(&cart->fdm));
    PetscCall(DMDestroy(&cart->cdm));
    PetscCall(DMDestroy(&cart->cfdm));
    for (d = 0; d < MESH_MAX_DIM; d++) {
        PetscCall(VecDestroy(&cart->c[d]));
        PetscCall(VecDestroy(&cart->cf[d]));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView_Cartesian(Mesh mesh, PetscViewer v) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscMPIInt rank;
    PetscBool isascii, iscgns;

    // TODO: support other viewers

    PetscFunctionBegin;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mesh), &rank));
    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));
    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERCGNS, &iscgns));

    if (isascii) {
        PetscViewerFormat format;

        PetscCall(PetscViewerGetFormat(v, &format));
        if (format == PETSC_VIEWER_LOAD_BALANCE) {
            PetscMPIInt size;
            DMDALocalInfo info;
            PetscInt i, nmax = 0, nmin = PETSC_MAX_INT, navg = 0, *nz, nzlocal;

            PetscCall(MPI_Comm_size(PetscObjectComm((PetscObject)mesh), &size));
            PetscCall(DMDAGetLocalInfo(cart->dm, &info));
            if (mesh->dim == 2)
                nzlocal = info.mx * info.my;
            else
                nzlocal = info.mx * info.my * info.mz;
            PetscCall(PetscMalloc1(size, &nz));
            PetscCallMPI(MPI_Allgather(&nzlocal, 1, MPIU_INT, nz, 1, MPIU_INT, PetscObjectComm((PetscObject)mesh)));
            for (i = 0; i < (PetscInt)size; i++) {
                nmax = PetscMax(nmax, nz[i]);
                nmin = PetscMin(nmin, nz[i]);
                navg += nz[i];
            }
            navg /= size;
            PetscCall(PetscFree(nz));
            PetscCall(PetscViewerASCIIPrintf(v,
                                             "  Load Balance - Grid Points: Min %" PetscInt_FMT "  avg %" PetscInt_FMT
                                             "  max %" PetscInt_FMT "\n",
                                             nmin, navg, nmax));
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        if (format != PETSC_VIEWER_ASCII_VTK_DEPRECATED && format != PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED &&
            format != PETSC_VIEWER_ASCII_GLVIS) {
            DMDALocalInfo info;

            PetscCall(DMDAGetLocalInfo(cart->dm, &info));
            PetscCall(PetscViewerASCIIPushSynchronized(v));
            switch (mesh->dim) {
                case 2:
                    PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                                 "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT
                                                                 " m %" PetscInt_FMT " n %" PetscInt_FMT "\n",
                                                                 rank, cart->N[0], cart->N[1], cart->nRanks[0],
                                                                 cart->nRanks[1]));
                    PetscCall(PetscViewerASCIISynchronizedPrintf(
                        v,
                        "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Y range of indices: %" PetscInt_FMT
                        " %" PetscInt_FMT "\n",
                        info.xs, info.xs + info.xm, info.ys, info.ys + info.ym));
                    break;
                case 3:
                    PetscCall(PetscViewerASCIISynchronizedPrintf(
                        v,
                        "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT " P %" PetscInt_FMT " m %" PetscInt_FMT
                        " n %" PetscInt_FMT " p %" PetscInt_FMT "\n",
                        rank, cart->N[0], cart->N[1], cart->N[1], cart->nRanks[0], cart->nRanks[1], cart->nRanks[2]));
                    PetscCall(PetscViewerASCIISynchronizedPrintf(
                        v,
                        "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Y range of indices: %" PetscInt_FMT
                        " %" PetscInt_FMT ", Z range of indices: %" PetscInt_FMT " %" PetscInt_FMT "\n",
                        info.xs, info.xs + info.xm, info.ys, info.ys + info.ym, info.zs, info.zs + info.zm));
                    break;
                default:
                    SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension");
            }
            PetscCall(PetscViewerFlush(v));
            PetscCall(PetscViewerASCIIPopSynchronized(v));
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    } else if (iscgns) {
        PetscCall(MeshView_CartesianCGNS(mesh, v));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCreate_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart;
    PetscInt d;

    PetscFunctionBegin;

    PetscCall(PetscNew(&cart));
    mesh->data = (void *)cart;

    for (d = 0; d < MESH_MAX_DIM; d++) {
        cart->N[d] = -1;
        cart->nRanks[d] = PETSC_DECIDE;
        cart->l[d] = NULL;
        cart->bndTypes[d] = MESH_BOUNDARY_NOT_PERIODIC;
    }

    cart->dm = NULL;
    cart->fdm = NULL;
    cart->cdm = NULL;
    cart->cfdm = NULL;
    for (d = 0; d < MESH_MAX_DIM; d++) {
        cart->c[d] = NULL;
        cart->cf[d] = NULL;
    }

    mesh->ops->setup = MeshSetUp_Cartesian;
    mesh->ops->destroy = MeshDestroy_Cartesian;
    mesh->ops->view = MeshView_Cartesian;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetSizes(Mesh mesh, PetscInt M, PetscInt N, PetscInt P) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    cart->N[0] = M;
    cart->N[1] = N;
    cart->N[2] = P;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetNumProcs(Mesh mesh, PetscInt m, PetscInt n, PetscInt p) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    PetscCheck(cart->nRanks[0] != m && cart->l[0], PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "Cannot set number of procs after setting ownership ranges, or reset ownership ranges");
    PetscCheck(cart->nRanks[1] != n && cart->l[1], PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "Cannot set number of procs after setting ownership ranges, or reset ownership ranges");
    PetscCheck(cart->nRanks[2] != p && cart->l[2], PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "Cannot set number of procs after setting ownership ranges, or reset ownership ranges");
    cart->nRanks[0] = m;
    cart->nRanks[1] = n;
    cart->nRanks[2] = p;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetBoundaryType(Mesh mesh, MeshBoundaryType bndx, MeshBoundaryType bndy,
                                            MeshBoundaryType bndz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    cart->bndTypes[0] = bndx;
    cart->bndTypes[1] = bndy;
    cart->bndTypes[2] = bndz;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetOwnershipRanges(Mesh mesh, const PetscInt *lx, const PetscInt *ly, const PetscInt *lz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    const PetscInt *lin[3] = {lx, ly, lz};
    PetscInt d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");

    for (d = 0; d < mesh->dim; d++) {
        if (lin[d]) {
            PetscCheck(cart->nRanks[d] >= 0, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
                       "Cannot set ownership ranges before setting number of procs");
            if (!cart->l[d])
                PetscCall(PetscMalloc1(cart->nRanks[d], &cart->l[d]));
            PetscCall(PetscArraycpy(cart->l[d], lin[d], cart->nRanks[d]));
        } else {
            PetscCall(PetscFree(cart->l[d]));
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetUniformCoordinates(Mesh mesh, PetscReal xmin, PetscReal xmax, PetscReal ymin,
                                                  PetscReal ymax, PetscReal zmin, PetscReal zmax) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscReal cmin[3] = {xmin, ymin, zmin}, cmax[3] = {xmax, ymax, zmax};
    PetscInt d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called after MeshSetUp()");

    for (d = 0; d < mesh->dim; d++) {
        DM subdm, subfdm;
        PetscReal *arrc;
        PetscReal **arrcf;
        PetscInt xs, xm, i, ileft;

        PetscCall(DMProductGetDM(cart->cdm, d, &subdm));
        PetscCall(DMDAVecGetArray(subdm, cart->c[d], &arrc));
        PetscCall(DMDAGetCorners(subdm, &xs, NULL, NULL, &xm, NULL, NULL));
        for (i = xs; i < xs + xm; i++)
            arrc[i] = cmin[d] + (cmax[d] - cmin[d]) * (i + 0.5) / cart->N[d];
        PetscCall(DMDAVecRestoreArray(subdm, cart->c[d], &arrc));

        PetscCall(DMProductGetDM(cart->cfdm, d, &subfdm));
        PetscCall(DMStagVecGetArray(subfdm, cart->cf[d], &arrcf));
        PetscCall(DMStagGetCorners(subfdm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));
        PetscCall(DMStagGetLocationSlot(subfdm, DMSTAG_LEFT, 0, &ileft));
        for (i = xs; i <= xs + xm; i++)
            arrcf[i][ileft] = cmin[d] + (cmax[d] - cmin[d]) * i / cart->N[d];
        PetscCall(DMStagVecRestoreArray(subfdm, cart->cf[d], &arrcf));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetInfo(Mesh mesh, PetscInt *dim, PetscInt *M, PetscInt *N, PetscInt *P, PetscInt *m,
                                    PetscInt *n, PetscInt *p, MeshBoundaryType *bndx, MeshBoundaryType *bndy,
                                    MeshBoundaryType *bndz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    if (dim)
        *dim = mesh->dim;
    if (M)
        *M = cart->N[0];
    if (N)
        *N = cart->N[1];
    if (P)
        *P = cart->N[2];
    if (m)
        *m = cart->nRanks[0];
    if (n)
        *n = cart->nRanks[1];
    if (p)
        *p = cart->nRanks[2];
    if (bndx)
        *bndx = cart->bndTypes[0];
    if (bndy)
        *bndy = cart->bndTypes[1];
    if (bndz)
        *bndz = cart->bndTypes[2];

    PetscFunctionReturn(PETSC_SUCCESS);
}
