#include <fluca/private/mesh_cartesian.h>
#include <fluca/private/meshimpl.h>
#include <petsc/private/petscimpl.h>
#include <petscdmstag.h>

extern PetscErrorCode MeshView_CartesianCGNS(Mesh mesh, PetscViewer v);

const char *MeshCartesianCoordinateStencilLocations[] = {
    "LEFT", "RIGHT", "BOTTOM", "TOP", "BACK", "FRONT", "MeshCartesianCoordinateStencilLocation", "", NULL};

static PetscErrorCode ConvertStencilLocation_Private(MeshCartesianStencilLocation meshloc,
                                                     DMStagStencilLocation *dmstagloc) {
    PetscFunctionBegin;
    switch (meshloc) {
        case MESHCARTESIAN_NULL_LOCATION:
            *dmstagloc = DMSTAG_NULL_LOCATION;
            break;
        case MESHCARTESIAN_LEFT:
            *dmstagloc = DMSTAG_LEFT;
            break;
        case MESHCARTESIAN_RIGHT:
            *dmstagloc = DMSTAG_RIGHT;
            break;
        case MESHCARTESIAN_DOWN:
            *dmstagloc = DMSTAG_DOWN;
            break;
        case MESHCARTESIAN_UP:
            *dmstagloc = DMSTAG_UP;
            break;
        case MESHCARTESIAN_BACK:
            *dmstagloc = DMSTAG_BACK;
            break;
        case MESHCARTESIAN_FRONT:
            *dmstagloc = DMSTAG_FRONT;
            break;
        default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid mesh stencil location");
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetFromOptions_Cartesian(Mesh mesh, PetscOptionItems *PetscOptionsObject) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscInt dim;

    PetscFunctionBegin;

    PetscCall(MeshGetDim(mesh, &dim));

    PetscOptionsHeadBegin(PetscOptionsObject, "MeshCartesian Options");

    PetscCall(PetscOptionsBoundedInt("-cartesian_grid_x", "Number of grid elements in x direction",
                                     "MeshCartesianSetSizes", cart->N[0], &cart->N[0], NULL, 1));
    PetscCall(PetscOptionsBoundedInt("-cartesian_grid_y", "Number of grid elements in y direction",
                                     "MeshCartesianSetSizes", cart->N[1], &cart->N[1], NULL, 1));
    if (dim > 2)
        PetscCall(PetscOptionsBoundedInt("-cartesian_grid_z", "Number of grid elements in z direction",
                                         "MeshCartesianSetSizes", cart->N[2], &cart->N[2], NULL, 1));

    PetscCall(PetscOptionsBoundedInt("-cartesian_processors_x", "Number of processors in x direction",
                                     "MeshCartesianSetNumProcs", cart->nRanks[0], &cart->nRanks[0], NULL,
                                     PETSC_DECIDE));
    PetscCall(PetscOptionsBoundedInt("-cartesian_processors_y", "Number of processors in y direction",
                                     "MeshCartesianSetNumProcs", cart->nRanks[1], &cart->nRanks[1], NULL,
                                     PETSC_DECIDE));
    if (dim > 2)
        PetscCall(PetscOptionsBoundedInt("-cartesian_processors_z", "Number of processors in z direction",
                                         "MeshCartesianSetNumProcs", cart->nRanks[2], &cart->nRanks[2], NULL,
                                         PETSC_DECIDE));

    PetscCall(PetscOptionsEnum("-cartesian_boundary_x", "Boundary type in x direction", "MeshCartesianSetBoundaryType",
                               MeshBoundaryTypes, (PetscEnum)cart->bndTypes[0], (PetscEnum *)&cart->bndTypes[0], NULL));
    PetscCall(PetscOptionsEnum("-cartesian_boundary_y", "Boundary type in y direction", "MeshCartesianSetBoundaryType",
                               MeshBoundaryTypes, (PetscEnum)cart->bndTypes[1], (PetscEnum *)&cart->bndTypes[1], NULL));
    if (dim > 2)
        PetscCall(PetscOptionsEnum("-cartesian_boundary_z", "Boundary type in z direction",
                                   "MeshCartesianSetBoundaryType", MeshBoundaryTypes, (PetscEnum)cart->bndTypes[2],
                                   (PetscEnum *)&cart->bndTypes[2], NULL));

    PetscOptionsHeadEnd();

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetUp_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    MPI_Comm comm;
    DMBoundaryType bnd[3];
    const PetscInt *l[3];
    PetscMPIInt rank, subrank[3];
    PetscInt dim, d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(MeshGetDim(mesh, &dim));
    PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

    for (d = 0; d < dim; d++)
        PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndTypes[d], &bnd[d]));

    switch (dim) {
        case 2:
            PetscCall(DMStagCreate2d(comm, bnd[0], bnd[1], cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1], 0,
                                     0, 1, DMSTAG_STENCIL_STAR, 1, cart->l[0], cart->l[1], &cart->dm));
            break;
        case 3:
            PetscCall(DMStagCreate3d(comm, bnd[0], bnd[1], bnd[2], cart->N[0], cart->N[1], cart->N[2], cart->nRanks[0],
                                     cart->nRanks[1], cart->nRanks[2], 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, cart->l[0],
                                     cart->l[1], cart->l[2], &cart->dm));
            break;
        default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
    }
    PetscCall(DMSetUp(cart->dm));
    switch (dim) {
        case 2:
            PetscCall(DMStagCreateCompatibleDMStag(cart->dm, 0, 1, 0, 0, &cart->fdm));
            break;
        case 3:
            PetscCall(DMStagCreateCompatibleDMStag(cart->dm, 0, 0, 1, 0, &cart->fdm));
            break;
    }

    /* Copy informations */
    PetscCall(DMStagGetNumRanks(cart->dm, &cart->nRanks[0], &cart->nRanks[1], &cart->nRanks[2]));
    PetscCall(DMStagGetOwnershipRanges(cart->dm, &l[0], &l[1], &l[2]));
    for (d = 0; d < dim; d++) {
        if (!cart->l[d])
            PetscCall(PetscMalloc1(cart->nRanks[d], &cart->l[d]));
        PetscCall(PetscArraycpy(cart->l[d], l[d], cart->nRanks[d]));
    }

    /* Set coordinate DM */
    PetscCall(DMStagSetUniformCoordinatesProduct(cart->dm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

    /* Allocate boundary DM */
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    switch (dim) {
        case 2:
            subrank[0] = rank % cart->nRanks[0];
            subrank[1] = rank / cart->nRanks[0];
            subrank[2] = 0;
            break;
        case 3:
            subrank[0] = rank % cart->nRanks[0];
            subrank[1] = (rank % (cart->nRanks[0] * cart->nRanks[1])) / cart->nRanks[0];
            subrank[2] = rank / (cart->nRanks[0] * cart->nRanks[1]);
            break;
        default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
    }
    for (d = 0; d < dim; d++) {
        MPI_Comm subcomm;
        PetscInt d1 = (d + 1) % dim, d2 = (d + 2) % dim;

        PetscCallMPI(MPI_Comm_split(comm, subrank[d], 0, &subcomm));
        switch (dim) {
            case 2:
                PetscCall(DMStagCreate1d(subcomm, DM_BOUNDARY_NONE, cart->N[d1], 0, 1, DMSTAG_STENCIL_NONE, 0, l[d1],
                                         &cart->bdm[d]));
                break;
            case 3:
                PetscCall(DMStagCreate2d(subcomm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, cart->N[d1], cart->N[d2],
                                         cart->nRanks[d1], cart->nRanks[d2], 0, 0, 1, DMSTAG_STENCIL_NONE, 0, l[d1],
                                         l[d2], &cart->bdm[d]));
                break;
            default:
                SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscInt dim, d;

    PetscFunctionBegin;

    PetscCall(MeshGetDim(mesh, &dim));

    for (d = 0; d < dim; d++)
        PetscCall(PetscFree(cart->l[d]));
    PetscCall(DMDestroy(&cart->dm));
    PetscCall(DMDestroy(&cart->fdm));
    for (d = 0; d < dim; d++)
        PetscCall(DMDestroy(&cart->bdm[d]));

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
        PetscInt dim, xs, xm, ys, ym, zs, zm;

        PetscCall(MeshGetDim(mesh, &dim));
        PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));

        PetscCall(PetscViewerASCIIPushSynchronized(v));
        switch (dim) {
            case 2:
                PetscCall(PetscViewerASCIISynchronizedPrintf(
                    v,
                    "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT "\n",
                    rank, cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1]));
                PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                             "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             "\n",
                                                             xs, xs + xm, ys, ys + ym));
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
                    xs, xs + xm, ys, ys + ym, zs, zs + zm));
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
    }

    cart->dm = NULL;
    cart->fdm = NULL;
    for (d = 0; d < MESH_MAX_DIM; d++)
        cart->bdm[d] = NULL;

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
    PetscInt dim, d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(mesh->state < MESH_STATE_SETUP, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");

    PetscCall(MeshGetDim(mesh, &dim));
    for (d = 0; d < dim; d++) {
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

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(mesh->state >= MESH_STATE_SETUP, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called after MeshSetUp()");
    PetscCall(DMStagSetUniformCoordinatesProduct(cart->dm, xmin, xmax, ymin, ymax, zmin, zmax));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetCoordinateArrays(Mesh mesh, PetscReal ***ax, PetscReal ***ay, PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(DMStagGetProductCoordinateArrays(cart->dm, ax, ay, az));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianRestoreCoordinateArrays(Mesh mesh, PetscReal ***ax, PetscReal ***ay, PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(DMStagRestoreProductCoordinateArrays(cart->dm, ax, ay, az));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetCoordinateArraysRead(Mesh mesh, const PetscReal ***ax, const PetscReal ***ay,
                                                    const PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, ax, ay, az));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianRestoreCoordinateArraysRead(Mesh mesh, const PetscReal ***ax, const PetscReal ***ay,
                                                        const PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, ax, ay, az));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetCoordinateLocationSlot(Mesh mesh, MeshCartesianStencilLocation loc, PetscInt *slot) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    DMStagStencilLocation dmstagloc;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(ConvertStencilLocation_Private(loc, &dmstagloc));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, dmstagloc, slot));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetGlobalSizes(Mesh mesh, PetscInt *M, PetscInt *N, PetscInt *P) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(DMStagGetGlobalSizes(cart->dm, M, N, P));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetCorners(Mesh mesh, PetscInt *xs, PetscInt *ys, PetscInt *zs, PetscInt *xm, PetscInt *ym,
                                       PetscInt *zm) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMStagGetCorners(cart->dm, xs, ys, zs, xm, ym, zm, NULL, NULL, NULL));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetIsFirstRank(Mesh mesh, PetscBool *isFirstRank0, PetscBool *isFirstRank1,
                                           PetscBool *isFirstRank2) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(DMStagGetIsFirstRank(cart->dm, isFirstRank0, isFirstRank1, isFirstRank2));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetIsLastRank(Mesh mesh, PetscBool *isLastRank0, PetscBool *isLastRank1,
                                          PetscBool *isLastRank2) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCall(DMStagGetIsLastRank(cart->dm, isLastRank0, isLastRank1, isLastRank2));
    PetscFunctionReturn(PETSC_SUCCESS);
}
