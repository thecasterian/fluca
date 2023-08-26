#include <impl/meshimpl.h>
#include <mesh/impl/cartesian/cartesian.h>
#include <petsc/private/petscimpl.h>
#include <petscdmda.h>
#include <petscdmstag.h>

PetscErrorCode MeshSetUp_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    MPI_Comm comm;
    DMBoundaryType bndx, bndy, bndz;
    PetscInt s = 1, daDof = 1, stagDof0, stagDof1, stagDof2, stagDof3;
    PetscInt m, n, p;
    const PetscInt *lx, *ly, *lz;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndx, &bndx));
    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndy, &bndy));
    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndz, &bndz));

    /* Allocate DMs */
    if (mesh->dim == 2)
        PetscCall(DMDACreate2d(comm, bndx, bndy, DMDA_STENCIL_STAR, cart->M, cart->N, cart->m, cart->n, daDof, s,
                               cart->lx, cart->ly, &cart->dm));
    else
        PetscCall(DMDACreate3d(comm, bndx, bndy, bndz, DMDA_STENCIL_STAR, cart->M, cart->N, cart->P, cart->m, cart->n,
                               cart->p, daDof, s, cart->lx, cart->ly, cart->lz, &cart->dm));
    PetscCall(DMSetUp(cart->dm));

    PetscCall(DMDAGetInfo(cart->dm, NULL, NULL, NULL, NULL, &m, &n, &p, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetOwnershipRanges(cart->dm, &lx, &ly, &lz));

    if (mesh->dim == 2) {
        stagDof0 = 0;
        stagDof1 = 1;
        stagDof2 = 0;
        PetscCall(DMStagCreate2d(comm, bndx, bndy, cart->M, cart->N, m, n, stagDof0, stagDof1, stagDof2,
                                 DMSTAG_STENCIL_STAR, s, lx, ly, &cart->vdm));
    } else {
        stagDof0 = 0;
        stagDof1 = 0;
        stagDof2 = 1;
        stagDof3 = 0;
        PetscCall(DMStagCreate3d(comm, bndx, bndy, bndz, cart->M, cart->N, cart->P, m, n, p, stagDof0, stagDof1,
                                 stagDof2, stagDof3, DMSTAG_STENCIL_STAR, s, lx, ly, lz, &cart->vdm));
    }
    PetscCall(DMSetUp(cart->vdm));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    PetscCall(PetscFree(cart->lx));
    PetscCall(PetscFree(cart->ly));
    PetscCall(PetscFree(cart->lz));
    PetscCall(DMDestroy(&cart->dm));
    PetscCall(DMDestroy(&cart->vdm));
    PetscCall(DMDestroy(&cart->cxdm));
    PetscCall(DMDestroy(&cart->cydm));
    PetscCall(DMDestroy(&cart->czdm));
    PetscCall(VecDestroy(&cart->cx));
    PetscCall(VecDestroy(&cart->cy));
    PetscCall(VecDestroy(&cart->cz));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView_Cartesian(Mesh mesh, PetscViewer v) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscMPIInt rank;
    PetscBool isascii;

    // TODO: support other viewers

    PetscFunctionBegin;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mesh), &rank));
    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));

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
            PetscInt m, n, p;

            PetscCall(DMDAGetLocalInfo(cart->dm, &info));
            PetscCall(DMDAGetInfo(cart->dm, NULL, NULL, NULL, NULL, &m, &n, &p, NULL, NULL, NULL, NULL, NULL, NULL));
            PetscCall(PetscViewerASCIIPushSynchronized(v));
            if (mesh->dim == 2) {
                PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                             "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT
                                                             " m %" PetscInt_FMT " n %" PetscInt_FMT "\n",
                                                             rank, cart->M, cart->N, m, n));
                PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                             "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             "\n",
                                                             info.xs, info.xs + info.xm, info.ys, info.ys + info.ym));
            } else {
                PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                             "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT
                                                             " P %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT
                                                             " p %" PetscInt_FMT "\n",
                                                             rank, cart->M, cart->N, cart->P, m, n, p));
                PetscCall(PetscViewerASCIISynchronizedPrintf(
                    v,
                    "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Y range of indices: %" PetscInt_FMT
                    " %" PetscInt_FMT ", Z range of indices: %" PetscInt_FMT " %" PetscInt_FMT "\n",
                    info.xs, info.xs + info.xm, info.ys, info.ys + info.ym, info.zs, info.zs + info.zm));
            }
            PetscCall(PetscViewerFlush(v));
            PetscCall(PetscViewerASCIIPopSynchronized(v));
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCreate_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart;

    PetscFunctionBegin;

    PetscCall(PetscNew(&cart));
    mesh->data = (void *)cart;

    cart->bndx = MESH_BOUNDARY_NOT_PERIODIC;
    cart->bndy = MESH_BOUNDARY_NOT_PERIODIC;
    cart->bndz = MESH_BOUNDARY_NOT_PERIODIC;
    cart->M = -1;
    cart->N = -1;
    cart->P = -1;
    cart->m = PETSC_DECIDE;
    cart->n = PETSC_DECIDE;
    cart->p = PETSC_DECIDE;
    cart->lx = NULL;
    cart->ly = NULL;
    cart->lz = NULL;
    cart->dm = NULL;
    cart->vdm = NULL;
    cart->cxdm = NULL;
    cart->cydm = NULL;
    cart->czdm = NULL;
    cart->cx = NULL;
    cart->cy = NULL;
    cart->cz = NULL;

    mesh->ops->setup = MeshSetUp_Cartesian;
    mesh->ops->destroy = MeshDestroy_Cartesian;
    mesh->ops->view = MeshView_Cartesian;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetSize(Mesh mesh, PetscInt M, PetscInt N, PetscInt P) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    cart->M = M;
    cart->N = N;
    cart->P = P;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetNumProcs(Mesh mesh, PetscInt m, PetscInt n, PetscInt p) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    PetscCheck(cart->m != m && cart->lx, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "Cannot set number of procs after setting ownership ranges, or reset ownership ranges");
    PetscCheck(cart->n != n && cart->ly, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "Cannot set number of procs after setting ownership ranges, or reset ownership ranges");
    PetscCheck(cart->p != p && cart->lz, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "Cannot set number of procs after setting ownership ranges, or reset ownership ranges");
    cart->m = m;
    cart->n = n;
    cart->p = p;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetBoundaryType(Mesh mesh, MeshBoundaryType bndx, MeshBoundaryType bndy,
                                            MeshBoundaryType bndz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    cart->bndx = bndx;
    cart->bndy = bndy;
    cart->bndz = bndz;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetOwnershipRanges(Mesh mesh, const PetscInt *lx, const PetscInt *ly, const PetscInt *lz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");

    if (lx) {
        PetscCheck(cart->m >= 0, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
                   "Cannot set ownership ranges before setting number of procs");
        if (!cart->lx)
            PetscCall(PetscMalloc1(cart->m, &cart->lx));
        PetscCall(PetscArraycpy(cart->lx, lx, cart->m));
    } else {
        PetscCall(PetscFree(cart->lx));
    }

    if (ly) {
        PetscCheck(cart->n >= 0, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
                   "Cannot set ownership ranges before setting number of procs");
        if (!cart->ly)
            PetscCall(PetscMalloc1(cart->n, &cart->ly));
        PetscCall(PetscArraycpy(cart->ly, ly, cart->n));
    } else {
        PetscCall(PetscFree(cart->ly));
    }

    if (lz) {
        PetscCheck(cart->p >= 0, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
                   "Cannot set ownership ranges before setting number of procs");
        if (!cart->lz)
            PetscCall(PetscMalloc1(cart->p, &cart->lz));
        PetscCall(PetscArraycpy(cart->lz, lz, cart->p));
    } else {
        PetscCall(PetscFree(cart->lz));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetUniformCoordinates(Mesh mesh, PetscReal xmin, PetscReal xmax, PetscReal ymin,
                                                  PetscReal ymax, PetscReal zmin, PetscReal zmax) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    MPI_Comm comm, subcomm;
    PetscInt m, n, p;
    DMBoundaryType bnd = DM_BOUNDARY_GHOSTED;
    PetscMPIInt rank, rankx, ranky, rankz, color, key = 0;
    const PetscInt *lx, *ly, *lz;
    PetscInt s = 1, dof0 = 1, dof1 = 1;
    PetscReal **arrc;
    PetscInt xs, xm;
    PetscInt ielem, ileft, i;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called after MeshSetUp()");

    PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));
    PetscCall(DMDAGetInfo(cart->dm, NULL, NULL, NULL, NULL, &m, &n, &p, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (mesh->dim == 2) {
        rankx = rank % m;
        ranky = rank / m;
        rankz = 0;
    } else {
        rankx = rank % m;
        ranky = (rank % (m * n)) / m;
        rankz = rank / (m * n);
    }
    PetscCall(DMDAGetOwnershipRanges(cart->dm, &lx, &ly, &lz));

    /* X coordinate */
    if (!cart->cxdm) {
        color = ranky + (mesh->dim > 2 ? n * rankz : 0);
        PetscCallMPI(MPI_Comm_split(comm, color, key, &subcomm));
        PetscCall(DMStagCreate1d(subcomm, bnd, cart->M, dof0, dof1, DMSTAG_STENCIL_BOX, s, lx, &cart->cxdm));
        PetscCall(DMSetUp(cart->cxdm));
        PetscCallMPI(MPI_Comm_free(&subcomm));
    }
    if (!cart->cx)
        PetscCall(DMCreateLocalVector(cart->cxdm, &cart->cx));
    PetscCall(DMStagVecGetArray(cart->cxdm, cart->cx, &arrc));
    PetscCall(DMStagGetCorners(cart->cxdm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMStagGetLocationSlot(cart->cxdm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->cxdm, DMSTAG_LEFT, 0, &ileft));
    for (i = xs - 1; i < xs + xm + 1; i++)
        arrc[i][ielem] = (xmax - xmin) * (i + 0.5) / cart->M + xmin;
    for (i = xs; i < xs + xm + 1; i++)
        arrc[i][ileft] = (xmax - xmin) * i / cart->M + xmin;
    PetscCall(DMStagVecRestoreArray(cart->cxdm, cart->cx, &arrc));
    PetscCall(DMLocalToLocalBegin(cart->cxdm, cart->cx, INSERT_VALUES, cart->cx));
    PetscCall(DMLocalToLocalEnd(cart->cxdm, cart->cx, INSERT_VALUES, cart->cx));

    /* Y coordinate */
    if (!cart->cydm) {
        color = rankx + (mesh->dim > 2 ? m * rankz : 0);
        PetscCallMPI(MPI_Comm_split(comm, color, key, &subcomm));
        PetscCall(DMStagCreate1d(subcomm, bnd, cart->N, dof0, dof1, DMSTAG_STENCIL_BOX, s, ly, &cart->cydm));
        PetscCall(DMSetUp(cart->cydm));
        PetscCallMPI(MPI_Comm_free(&subcomm));
    }
    if (!cart->cy)
        PetscCall(DMCreateLocalVector(cart->cydm, &cart->cy));
    PetscCall(DMStagVecGetArray(cart->cydm, cart->cy, &arrc));
    PetscCall(DMStagGetCorners(cart->cydm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMStagGetLocationSlot(cart->cydm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->cydm, DMSTAG_LEFT, 0, &ileft));
    for (i = xs - 1; i < xs + xm + 1; i++)
        arrc[i][ielem] = (ymax - ymin) * (i + 0.5) / cart->N + ymin;
    for (i = xs; i <= xs + xm; i++)
        arrc[i][ileft] = (ymax - ymin) * i / cart->N + ymin;
    PetscCall(DMStagVecRestoreArray(cart->cydm, cart->cy, &arrc));
    PetscCall(DMLocalToLocalBegin(cart->cydm, cart->cy, INSERT_VALUES, cart->cy));
    PetscCall(DMLocalToLocalEnd(cart->cydm, cart->cy, INSERT_VALUES, cart->cy));

    /* Z coordinate */
    if (mesh->dim > 2) {
        if (!cart->czdm) {
            color = rankx + m * ranky;
            PetscCallMPI(MPI_Comm_split(comm, color, key, &subcomm));
            PetscCall(DMStagCreate1d(subcomm, bnd, cart->P, dof0, dof1, DMSTAG_STENCIL_BOX, s, lz, &cart->czdm));
            PetscCall(DMSetUp(cart->czdm));
            PetscCallMPI(MPI_Comm_free(&subcomm));
        }
        if (!cart->cz)
            PetscCall(DMCreateLocalVector(cart->czdm, &cart->cz));
        PetscCall(DMStagVecGetArray(cart->czdm, cart->cz, &arrc));
        PetscCall(DMStagGetCorners(cart->czdm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));
        PetscCall(DMStagGetLocationSlot(cart->czdm, DMSTAG_ELEMENT, 0, &ielem));
        PetscCall(DMStagGetLocationSlot(cart->czdm, DMSTAG_LEFT, 0, &ileft));
        for (i = xs - 1; i < xs + xm + 1; i++)
            arrc[i][ielem] = (zmax - zmin) * (i + 0.5) / cart->P + zmin;
        for (i = xs; i < xs + xm + 1; i++)
            arrc[i][ileft] = (zmax - zmin) * i / cart->P + zmin;
        PetscCall(DMStagVecRestoreArray(cart->czdm, cart->cz, &arrc));
        PetscCall(DMLocalToLocalBegin(cart->czdm, cart->cz, INSERT_VALUES, cart->cz));
        PetscCall(DMLocalToLocalEnd(cart->czdm, cart->cz, INSERT_VALUES, cart->cz));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianTransformCoordinates(Mesh mesh, PetscReal (*tx)(PetscReal, void *),
                                                 PetscReal (*ty)(PetscReal, void *), PetscReal (*tz)(PetscReal, void *),
                                                 void *ctx) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscReal **arrc;
    PetscInt ielem, ileft;
    PetscInt xs, xm;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMStagVecGetArray(cart->cxdm, cart->cx, &arrc));
    PetscCall(DMStagGetCorners(cart->cxdm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMStagGetLocationSlot(cart->cxdm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->cxdm, DMSTAG_LEFT, 0, &ileft));
    for (PetscInt i = xs; i < xs + xm; i++)
        arrc[i][ielem] = tx(arrc[i][ielem], ctx);
    for (PetscInt i = xs; i <= xs + xm; i++)
        arrc[i][ileft] = tx(arrc[i][ileft], ctx);
    PetscCall(DMStagVecRestoreArray(cart->cxdm, cart->cx, &arrc));
    PetscCall(DMLocalToLocalBegin(cart->cxdm, cart->cx, INSERT_VALUES, cart->cx));
    PetscCall(DMLocalToLocalEnd(cart->cxdm, cart->cx, INSERT_VALUES, cart->cx));

    PetscCall(DMStagVecGetArray(cart->cydm, cart->cy, &arrc));
    PetscCall(DMStagGetCorners(cart->cydm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMStagGetLocationSlot(cart->cydm, DMSTAG_ELEMENT, 0, &ielem));
    PetscCall(DMStagGetLocationSlot(cart->cydm, DMSTAG_LEFT, 0, &ileft));
    for (PetscInt i = xs; i < xs + xm; i++)
        arrc[i][ielem] = ty(arrc[i][ielem], ctx);
    for (PetscInt i = xs; i <= xs + xm; i++)
        arrc[i][ileft] = ty(arrc[i][ileft], ctx);
    PetscCall(DMStagVecRestoreArray(cart->cydm, cart->cy, &arrc));
    PetscCall(DMLocalToLocalBegin(cart->cydm, cart->cy, INSERT_VALUES, cart->cy));
    PetscCall(DMLocalToLocalEnd(cart->cydm, cart->cy, INSERT_VALUES, cart->cy));

    if (mesh->dim > 2) {
        PetscCall(DMStagVecGetArray(cart->czdm, cart->cz, &arrc));
        PetscCall(DMStagGetCorners(cart->czdm, &xs, NULL, NULL, &xm, NULL, NULL, NULL, NULL, NULL));
        PetscCall(DMStagGetLocationSlot(cart->czdm, DMSTAG_ELEMENT, 0, &ielem));
        PetscCall(DMStagGetLocationSlot(cart->czdm, DMSTAG_LEFT, 0, &ileft));
        for (PetscInt i = xs; i < xs + xm; i++)
            arrc[i][ielem] = tz(arrc[i][ielem], ctx);
        for (PetscInt i = xs; i <= xs + xm; i++)
            arrc[i][ileft] = tz(arrc[i][ileft], ctx);
        PetscCall(DMStagVecRestoreArray(cart->czdm, cart->cz, &arrc));
        PetscCall(DMLocalToLocalBegin(cart->czdm, cart->cz, INSERT_VALUES, cart->cz));
        PetscCall(DMLocalToLocalEnd(cart->czdm, cart->cz, INSERT_VALUES, cart->cz));
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
        *M = cart->M;
    if (N)
        *N = cart->N;
    if (P)
        *P = cart->P;
    if (m)
        *m = cart->m;
    if (n)
        *n = cart->n;
    if (p)
        *p = cart->p;
    if (bndx)
        *bndx = cart->bndx;
    if (bndy)
        *bndy = cart->bndy;
    if (bndz)
        *bndz = cart->bndz;

    PetscFunctionReturn(PETSC_SUCCESS);
}
