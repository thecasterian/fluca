#include <fluca/private/mesh_cartesian.h>
#include <fluca/private/meshimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscdmda.h>
#include <petscdmstag.h>

extern PetscErrorCode MeshView_CartesianCGNS(Mesh mesh, PetscViewer v);

const char *MeshCartesianCoordinateStencilLocations[] = {
    "LEFT", "RIGHT", "BOTTOM", "TOP", "BACK", "FRONT", "MeshCartesianCoordinateStencilLocation", "", NULL};

PetscErrorCode MeshSetFromOptions_Cartesian(Mesh mesh, PetscOptionItems *PetscOptionsObject) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    PetscOptionsHeadBegin(PetscOptionsObject, "MeshCartesian Options");

    PetscCall(PetscOptionsBoundedInt("-cartesian_grid_x", "Number of grid elements in x direction",
                                     "MeshCartesianSetSizes", cart->N[0], &cart->N[0], NULL, 1));
    PetscCall(PetscOptionsBoundedInt("-cartesian_grid_y", "Number of grid elements in y direction",
                                     "MeshCartesianSetSizes", cart->N[1], &cart->N[1], NULL, 1));
    if (mesh->dim > 2)
        PetscCall(PetscOptionsBoundedInt("-cartesian_grid_z", "Number of grid elements in z direction",
                                         "MeshCartesianSetSizes", cart->N[2], &cart->N[2], NULL, 1));

    PetscCall(PetscOptionsBoundedInt("-cartesian_processors_x", "Number of processors in x direction",
                                     "MeshCartesianSetNumProcs", cart->nRanks[0], &cart->nRanks[0], NULL,
                                     PETSC_DECIDE));
    PetscCall(PetscOptionsBoundedInt("-cartesian_processors_y", "Number of processors in y direction",
                                     "MeshCartesianSetNumProcs", cart->nRanks[1], &cart->nRanks[1], NULL,
                                     PETSC_DECIDE));
    if (mesh->dim > 2)
        PetscCall(PetscOptionsBoundedInt("-cartesian_processors_z", "Number of processors in z direction",
                                         "MeshCartesianSetNumProcs", cart->nRanks[2], &cart->nRanks[2], NULL,
                                         PETSC_DECIDE));

    PetscCall(PetscOptionsEnum("-cartesian_boundary_x", "Boundary type in x direction", "MeshCartesianSetBoundaryType",
                               MeshBoundaryTypes, (PetscEnum)cart->bndTypes[0], (PetscEnum *)&cart->bndTypes[0], NULL));
    PetscCall(PetscOptionsEnum("-cartesian_boundary_y", "Boundary type in y direction", "MeshCartesianSetBoundaryType",
                               MeshBoundaryTypes, (PetscEnum)cart->bndTypes[1], (PetscEnum *)&cart->bndTypes[1], NULL));
    if (mesh->dim > 2)
        PetscCall(PetscOptionsEnum("-cartesian_boundary_z", "Boundary type in z direction",
                                   "MeshCartesianSetBoundaryType", MeshBoundaryTypes, (PetscEnum)cart->bndTypes[2],
                                   (PetscEnum *)&cart->bndTypes[2], NULL));

    PetscOptionsHeadEnd();

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetUp_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    MPI_Comm comm;
    PetscMPIInt rank;
    DMBoundaryType bndx, bndy, bndz;
    PetscInt daS = 1, daDof = 1, stagS = 1, stagDof0, stagDof1, stagDof2, stagDof3;
    const PetscInt *lx, *ly, *lz;
    PetscInt d;

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
                                   cart->nRanks[1], daDof, daS, cart->l[0], cart->l[1], &cart->dm));
            break;
        case 3:
            PetscCall(DMDACreate3d(comm, bndx, bndy, bndz, DMDA_STENCIL_STAR, cart->N[0], cart->N[1], cart->N[2],
                                   cart->nRanks[0], cart->nRanks[1], cart->nRanks[2], daDof, daS, cart->l[0],
                                   cart->l[1], cart->l[2], &cart->dm));
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
                                     stagDof0, stagDof1, stagDof2, DMSTAG_STENCIL_STAR, stagS, cart->l[0], cart->l[1],
                                     &cart->fdm));
            break;
        case 3:
            stagDof0 = 0;
            stagDof1 = 0;
            stagDof2 = 1;
            stagDof3 = 0;
            PetscCall(DMStagCreate3d(comm, bndx, bndy, bndz, cart->N[0], cart->N[1], cart->N[2], cart->nRanks[0],
                                     cart->nRanks[1], cart->nRanks[2], stagDof0, stagDof1, stagDof2, stagDof3,
                                     DMSTAG_STENCIL_STAR, stagS, cart->l[0], cart->l[1], cart->l[2], &cart->fdm));
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

    for (d = 0; d < mesh->dim; d++) {
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
        PetscCallMPI(MPI_Comm_split(comm, color, key, &cart->subcomm[d]));
    }

    for (d = 0; d < mesh->dim; d++) {
        PetscReal *c, *w;

        PetscMalloc1(cart->N[d] + 2, &c);
        cart->c[d] = c + 1;
        PetscMalloc1(cart->N[d] + 1, &cart->cf[d]);
        PetscMalloc1(cart->N[d] + 2, &w);
        cart->w[d] = w + 1;
        PetscMalloc1(cart->N[d] + 1, &cart->rf[d]);
        PetscMalloc1(cart->N[d], &cart->a[d]);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscInt d;

    PetscFunctionBegin;

    for (d = 0; d < mesh->dim; d++) {
        PetscCall(PetscFree(cart->l[d]));
        PetscCallMPI(MPI_Comm_free(&cart->subcomm[d]));
    }
    PetscCall(DMDestroy(&cart->dm));
    PetscCall(DMDestroy(&cart->fdm));
    for (d = 0; d < mesh->dim; d++) {
        PetscReal *dummy;

        dummy = cart->c[d] - 1;
        PetscCall(PetscFree(dummy));
        cart->c[d] = NULL;
        PetscCall(PetscFree(cart->cf[d]));
        dummy = cart->w[d] - 1;
        PetscCall(PetscFree(dummy));
        cart->w[d] = NULL;
        PetscCall(PetscFree(cart->rf[d]));
        PetscCall(PetscFree(cart->a[d]));
    }

    PetscCall(PetscFree(mesh->data));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView_Cartesian(Mesh mesh, PetscViewer v) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscMPIInt rank;
    PetscBool isascii, iscgns, isdraw;

    PetscFunctionBegin;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mesh), &rank));
    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));
    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERCGNS, &iscgns));
    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERDRAW, &isdraw));

    if (isascii) {
        DMDALocalInfo info;

        PetscCall(DMDAGetLocalInfo(cart->dm, &info));
        PetscCall(PetscViewerASCIIPushSynchronized(v));
        switch (mesh->dim) {
            case 2:
                PetscCall(PetscViewerASCIISynchronizedPrintf(
                    v,
                    "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT "\n",
                    rank, cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1]));
                PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                             "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             "\n",
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
    } else if (iscgns) {
        PetscCall(MeshView_CartesianCGNS(mesh, v));
    } else if (isdraw) {
        PetscCall(DMView(cart->dm, v));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetDM_Cartesian(Mesh mesh, DM *dm) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    *dm = cart->dm;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetFaceDM_Cartesian(Mesh mesh, DM *dm) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    *dm = cart->fdm;
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
        cart->rank[d] = -1;
        cart->subcomm[d] = NULL;
    }

    cart->dm = NULL;
    cart->fdm = NULL;
    for (d = 0; d < MESH_MAX_DIM; d++) {
        cart->c[d] = NULL;
        cart->cf[d] = NULL;
        cart->w[d] = NULL;
        cart->rf[d] = NULL;
        cart->a[d] = NULL;
    }

    mesh->ops->setfromoptions = MeshSetFromOptions_Cartesian;
    mesh->ops->setup = MeshSetUp_Cartesian;
    mesh->ops->destroy = MeshDestroy_Cartesian;
    mesh->ops->getdm = MeshGetDM_Cartesian;
    mesh->ops->getfacedm = MeshGetFaceDM_Cartesian;
    mesh->ops->view = MeshView_Cartesian;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianSetSizes(Mesh mesh, PetscInt M, PetscInt N, PetscInt P) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(mesh->state < MESH_STATE_SETUP, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
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
    PetscCheck(mesh->state < MESH_STATE_SETUP, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
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
    PetscCheck(mesh->state < MESH_STATE_SETUP, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
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
    PetscCheck(mesh->state < MESH_STATE_SETUP, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
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
    PetscInt d, i;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(mesh->state >= MESH_STATE_SETUP, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called after MeshSetUp()");

    for (d = 0; d < mesh->dim; d++) {
        for (i = -1; i < cart->N[d] + 1; i++)
            cart->c[d][i] = cmin[d] + (cmax[d] - cmin[d]) * (i + 0.5) / cart->N[d];
        for (i = 0; i <= cart->N[d]; i++)
            cart->cf[d][i] = cmin[d] + (cmax[d] - cmin[d]) * i / cart->N[d];
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianFaceCoordinateGetArray(Mesh mesh, PetscReal ***ax, PetscReal ***ay, PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscReal ***arr[3] = {ax, ay, az};
    PetscInt s[3], m[3], d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMDAGetCorners(cart->dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2]));

    for (d = 0; d < mesh->dim; d++)
        if (arr[d]) {
            PetscReal **a;
            PetscInt i;

            PetscCall(PetscMalloc1(m[d] + 1, &a));
            for (i = 0; i <= m[d]; i++)
                a[i] = cart->cf[d] + s[d] + i;
            *arr[d] = a - s[d];
        }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianFaceCoordinateGetArrayRead(Mesh mesh, const PetscReal ***ax, const PetscReal ***ay,
                                                       const PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    const PetscReal ***arr[3] = {ax, ay, az};
    PetscInt s[3], m[3], d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    *ax = NULL;
    *ay = NULL;
    *az = NULL;

    PetscCall(DMDAGetCorners(cart->dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2]));

    for (d = 0; d < mesh->dim; d++)
        if (arr[d]) {
            const PetscReal **a;
            PetscInt i;

            PetscCall(PetscMalloc1(m[d] + 1, &a));
            for (i = 0; i <= m[d]; i++)
                a[i] = cart->cf[d] + s[d] + i;
            *arr[d] = a - s[d];
        }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianFaceCoordinateRestoreArray(Mesh mesh, PetscReal ***ax, PetscReal ***ay, PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscReal ***arr[3] = {ax, ay, az};
    PetscInt s[3], m[3], d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMDAGetCorners(cart->dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2]));

    for (d = 0; d < mesh->dim; d++)
        if (arr[d]) {
            void *dummy = (void *)(*arr[d] + s[d]);
            PetscFree(dummy);
            *arr[d] = NULL;
        }

    for (d = 0; d < mesh->dim; d++) {
        /* All gather face coordinates */
        PetscReal *sendbuf = cart->cf[d] + s[d];
        PetscMPIInt sendcount = m[d] + (s[d] + m[d] >= cart->N[d]);
        PetscMPIInt recvcounts[cart->nRanks[d]], displs[cart->nRanks[d]];
        PetscInt i;

        for (i = 0; i < cart->nRanks[d]; i++) {
            recvcounts[i] = cart->l[d][i] + (i == cart->nRanks[d] - 1);
            displs[i] = i > 0 ? displs[i - 1] + recvcounts[i - 1] : 0;
        }

        PetscCallMPI(MPI_Allgatherv(sendbuf, sendcount, MPIU_REAL, cart->cf[d], recvcounts, displs, MPIU_REAL,
                                    cart->subcomm[d]));

        for (i = 0; i < cart->N[d]; i++) {
            cart->c[d][i] = 0.5 * (cart->cf[d][i] + cart->cf[d][i + 1]);
            cart->w[d][i] = cart->cf[d][i + 1] - cart->cf[d][i];
        }
        /* Ghots elements */
        switch (cart->bndTypes[d]) {
            case MESH_BOUNDARY_NOT_PERIODIC:
                cart->c[d][-1] = cart->cf[d][0] - 0.5 * cart->w[d][0];
                cart->c[d][cart->N[d]] = cart->cf[d][cart->N[d]] + 0.5 * cart->w[d][cart->N[d] - 1];
                cart->w[d][-1] = cart->w[d][0];
                cart->w[d][cart->N[d]] = cart->w[d][cart->N[d] - 1];
                break;
            case MESH_BOUNDARY_PERIODIC:
                cart->c[d][-1] = cart->cf[d][0] - 0.5 * cart->w[d][cart->N[d] - 1];
                cart->c[d][cart->N[d]] = cart->cf[d][cart->N[d]] + 0.5 * cart->w[d][0];
                cart->w[d][-1] = cart->w[d][cart->N[d] - 1];
                cart->w[d][cart->N[d]] = cart->w[d][0];
                break;
            default:
                SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported boundary type");
        }
        for (i = 0; i <= cart->N[d]; i++)
            cart->rf[d][i] = cart->w[d][i] / (cart->w[d][i - 1] + cart->w[d][i]);
        for (i = 0; i < cart->N[d]; i++) {
            cart->a[d][i][0] = 1.0 / (cart->w[d][i] * (cart->c[d][i] - cart->c[d][i - 1]));
            cart->a[d][i][1] = 1.0 / (cart->w[d][i] * (cart->c[d][i + 1] - cart->c[d][i]));
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianFaceCoordinateRestoreArrayRead(Mesh mesh, const PetscReal ***arrx, const PetscReal ***arry,
                                                           const PetscReal ***arrz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    const PetscReal ***arr[3] = {arrx, arry, arrz};
    PetscInt s[3], d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMDAGetCorners(cart->dm, &s[0], &s[1], &s[2], NULL, NULL, NULL));

    for (d = 0; d < mesh->dim; d++)
        if (arr[d]) {
            void *dummy = (void *)(*arr[d] + s[d]);
            PetscFree(dummy);
            *arr[d] = NULL;
        }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianCoordinateGetArrayRead(Mesh mesh, const PetscReal **arrx, const PetscReal **arry,
                                                   const PetscReal **arrz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    const PetscReal **arr[3] = {arrx, arry, arrz};
    PetscInt d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    for (d = 0; d < mesh->dim; d++)
        if (arr[d])
            *arr[d] = cart->c[d];

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianCoordinateRestoreArrayRead(Mesh mesh, const PetscReal **arrx, const PetscReal **arry,
                                                       const PetscReal **arrz) {
    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    if (arrx)
        *arrx = NULL;
    if (arry)
        *arry = NULL;
    if (arrz)
        *arrz = NULL;

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

PetscErrorCode MeshCartesianGetCorners(Mesh mesh, PetscInt *xs, PetscInt *ys, PetscInt *zs, PetscInt *xm, PetscInt *ym,
                                       PetscInt *zm) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMDAGetCorners(cart->dm, xs, ys, zs, xm, ym, zm));

    PetscFunctionReturn(PETSC_SUCCESS);
}
