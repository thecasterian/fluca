#include <fluca/private/mesh_cartesian.h>
#include <fluca/private/meshimpl.h>
#include <petsc/private/petscimpl.h>
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
    DMBoundaryType dmBndTypes[3];
    PetscInt dofElem = 1, dofFace = 1, stencilWidth = 1;
    const PetscInt *l[3];
    DM cdm;
    PetscInt d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

    for (d = 0; d < mesh->dim; d++)
        PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndTypes[d], &dmBndTypes[d]));

    /* Allocate DMs */
    switch (mesh->dim) {
        case 2:
            PetscCall(DMStagCreate2d(comm, dmBndTypes[0], dmBndTypes[1], cart->N[0], cart->N[1], cart->nRanks[0],
                                     cart->nRanks[1], 0, 0, dofElem, DMSTAG_STENCIL_STAR, stencilWidth, cart->l[0],
                                     cart->l[1], &cart->dm));
            break;
        case 3:
            PetscCall(DMStagCreate3d(comm, dmBndTypes[0], dmBndTypes[1], dmBndTypes[2], cart->N[0], cart->N[1],
                                     cart->N[2], cart->nRanks[0], cart->nRanks[1], cart->nRanks[2], 0, 0, 0, dofElem,
                                     DMSTAG_STENCIL_STAR, stencilWidth, cart->l[0], cart->l[1], cart->l[2], &cart->dm));
            break;
        default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
    }
    PetscCall(DMSetUp(cart->dm));
    switch (mesh->dim) {
        case 2:
            PetscCall(DMStagCreateCompatibleDMStag(cart->dm, 0, dofFace, 0, 0, &cart->fdm));
            break;
        case 3:
            PetscCall(DMStagCreateCompatibleDMStag(cart->dm, 0, 0, dofFace, 0, &cart->fdm));
            break;
        default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
    }

    PetscCall(DMStagGetNumRanks(cart->dm, &cart->nRanks[0], &cart->nRanks[1], &cart->nRanks[2]));
    for (d = 0; d < mesh->dim; d++)
        if (!cart->l[d])
            PetscCall(PetscMalloc1(cart->nRanks[d], &cart->l[d]));
    PetscCall(DMStagGetOwnershipRanges(cart->dm, &l[0], &l[1], &l[2]));
    for (d = 0; d < mesh->dim; d++)
        PetscCall(PetscArraycpy(cart->l[d], l[d], cart->nRanks[d]));

    PetscCall(DMStagSetUniformCoordinatesProduct(cart->dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
    PetscCall(DMGetCoordinateDM(cart->dm, &cdm));
    for (d = 0; d < mesh->dim; d++) {
        DM subdm;
        MPI_Comm subcomm;

        PetscCall(DMProductGetDM(cdm, d, &subdm));
        PetscCall(PetscObjectGetComm((PetscObject)subdm, &subcomm));

        PetscCall(DMStagCreate1d(subcomm, dmBndTypes[d], cart->N[d], 0, dofElem, DMSTAG_STENCIL_BOX, stencilWidth,
                                 cart->l[d], &cart->subdm[d]));
        PetscCall(DMSetUp(cart->subdm[d]));
        PetscCall(DMCreateLocalVector(cart->subdm[d], &cart->width[d]));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscInt d;

    PetscFunctionBegin;

    for (d = 0; d < mesh->dim; d++)
        PetscCall(PetscFree(cart->l[d]));
    PetscCall(DMDestroy(&cart->dm));
    PetscCall(DMDestroy(&cart->fdm));
    for (d = 0; d < mesh->dim; d++) {
        PetscCall(DMDestroy(&cart->subdm[d]));
        PetscCall(VecDestroy(&cart->width[d]));
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
        PetscInt xs, ys, zs, xm, ym, zm;

        PetscCall(DMStagGetCorners(cart->dm, &xs, &ys, &zs, &xm, &ym, &zm, NULL, NULL, NULL));
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
    for (d = 0; d < MESH_MAX_DIM; d++) {
        cart->subdm[d] = NULL;
        cart->width[d] = NULL;
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

PetscErrorCode MeshCartesianGetCoordinateArraysRead(Mesh mesh, const PetscReal ***ax, const PetscReal ***ay,
                                                    const PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMStagGetProductCoordinateArraysRead(cart->dm, ax, ay, az));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianRestoreCoordinateArrays(Mesh mesh, PetscReal ***ax, PetscReal ***ay, PetscReal ***az) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    PetscReal ***a[3] = {ax, ay, az};
    PetscInt i, d, iprev, inext, icenter;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_LEFT, &iprev));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_RIGHT, &inext));
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, DMSTAG_ELEMENT, &icenter));
    for (d = 0; d < mesh->dim; d++) {
        PetscInt s, m;
        PetscReal **arrw;
        PetscInt iw;

        PetscCall(DMStagGetCorners(cart->subdm[d], &s, NULL, NULL, &m, NULL, NULL, NULL, NULL, NULL));
        for (i = s; i < s + m; i++)
            (*a[d])[i][icenter] = ((*a[d])[i][iprev] + (*a[d])[i][inext]) / 2.0;

        PetscCall(DMStagVecGetArray(cart->subdm[d], cart->width[d], &arrw));
        PetscCall(DMStagGetLocationSlot(cart->subdm[d], DMSTAG_ELEMENT, 0, &iw));
        for (i = s; i < s + m; i++)
            arrw[i][iw] = (*a[d])[i][inext] - (*a[d])[i][iprev];
        PetscCall(DMStagVecRestoreArray(cart->subdm[d], cart->width[d], &arrw));
        PetscCall(DMLocalToLocalBegin(cart->subdm[d], cart->width[d], INSERT_VALUES, cart->width[d]));
        PetscCall(DMLocalToLocalEnd(cart->subdm[d], cart->width[d], INSERT_VALUES, cart->width[d]));
    }

    // set widths and coordinates of ghosts
    for (d = 0; d < mesh->dim; d++) {
        PetscInt s, m;
        PetscReal **arrw;
        PetscInt iw;

        PetscCall(DMStagGetCorners(cart->subdm[d], &s, NULL, NULL, &m, NULL, NULL, NULL, NULL, NULL));
        PetscCall(DMStagVecGetArray(cart->subdm[d], cart->width[d], &arrw));
        PetscCall(DMStagGetLocationSlot(cart->subdm[d], DMSTAG_ELEMENT, 0, &iw));
        if (s == 0 && cart->bndTypes[d] != MESH_BOUNDARY_PERIODIC)
            arrw[s - 1][iw] = arrw[s][iw];
        if (s + m == cart->N[d] && cart->bndTypes[d] != MESH_BOUNDARY_PERIODIC)
            arrw[s + m][iw] = arrw[s + m - 1][iw];
        (*a[d])[s - 1][icenter] = (*a[d])[s][iprev] - arrw[s - 1][iw] / 2.0;
        (*a[d])[s + m][icenter] = (*a[d])[s + m - 1][inext] + arrw[s + m][iw] / 2.0;
        PetscCall(DMStagVecRestoreArray(cart->subdm[d], cart->width[d], &arrw));
    }

    PetscCall(DMStagRestoreProductCoordinateArrays(cart->dm, ax, ay, az));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianRestoreCoordinateArraysRead(Mesh mesh, const PetscReal ***arrx, const PetscReal ***arry,
                                                        const PetscReal ***arrz) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(DMStagRestoreProductCoordinateArraysRead(cart->dm, arrx, arry, arrz));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetCoordinateLocationSlot(Mesh mesh, MeshCartesianCoordinateStencilLocation loc,
                                                      PetscInt *slot) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    DMStagStencilLocation stagLoc;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    switch (loc) {
        case MESHCARTESIAN_PREV:
            stagLoc = DMSTAG_LEFT;
            break;
        case MESHCARTESIAN_NEXT:
            stagLoc = DMSTAG_RIGHT;
            break;
        default:
            SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONG, "Invalid coordinate stencil location");
    }
    PetscCall(DMStagGetProductCoordinateLocationSlot(cart->dm, stagLoc, slot));

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

    PetscCall(DMStagGetCorners(cart->dm, xs, ys, zs, xm, ym, zm, NULL, NULL, NULL));

    PetscFunctionReturn(PETSC_SUCCESS);
}
