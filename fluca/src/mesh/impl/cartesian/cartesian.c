#include <impl/meshimpl.h>
#include <mesh/impl/cartesian/cartesian.h>
#include <petsc/private/petscimpl.h>
#include <petscdmda.h>
#include <petscdmstag.h>

static PetscErrorCode MeshBoundaryTypeToDMBoundaryType(MeshBoundaryType type, DMBoundaryType *dmtype) {
    PetscFunctionBegin;

    switch (type) {
        case MESH_BOUNDARY_NOT_PERIODIC:
            *dmtype = DM_BOUNDARY_GHOSTED;
            break;
        case MESH_BOUNDARY_PERIODIC:
            *dmtype = DM_BOUNDARY_PERIODIC;
            break;
        default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid boundary type %d", type);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetUp_Cartesian(Mesh mesh) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;
    MPI_Comm comm;
    DMBoundaryType bndx, bndy, bndz;
    PetscInt daDof, stagDof0, stagDof1, stagDof2, stagDof3;
    PetscInt m, n, p;
    const PetscInt *lx, *ly, *lz;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

    PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndx, &bndx));
    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndy, &bndy));
    PetscCall(MeshBoundaryTypeToDMBoundaryType(cart->bndz, &bndz));

    /* Allocate DMs */
    daDof = 1;
    if (mesh->dim == 2)
        PetscCall(DMDACreate2d(comm, bndx, bndy, DMDA_STENCIL_STAR, cart->M, cart->N, cart->m, cart->n, daDof, cart->s,
                               cart->lx, cart->ly, &cart->dm));
    else
        PetscCall(DMDACreate3d(comm, bndx, bndy, bndz, DMDA_STENCIL_STAR, cart->M, cart->N, cart->P, cart->m, cart->n,
                               cart->p, daDof, cart->s, cart->lx, cart->ly, cart->lz, &cart->dm));
    PetscCall(DMSetUp(cart->dm));

    PetscCall(DMDAGetInfo(cart->dm, NULL, NULL, NULL, NULL, &m, &n, &p, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetOwnershipRanges(cart->dm, &lx, &ly, &lz));

    if (mesh->dim == 2) {
        stagDof0 = 0;
        stagDof1 = 1;
        stagDof2 = 0;
        PetscCall(DMStagCreate2d(comm, bndx, bndy, cart->M, cart->N, m, n, stagDof0, stagDof1, stagDof2,
                                 DMSTAG_STENCIL_STAR, cart->s, lx, ly, &cart->vdm));
    } else {
        stagDof0 = 0;
        stagDof1 = 0;
        stagDof2 = 1;
        stagDof3 = 0;
        PetscCall(DMStagCreate3d(comm, bndx, bndy, bndz, cart->M, cart->N, cart->P, m, n, p, stagDof0, stagDof1,
                                 stagDof2, stagDof3, DMSTAG_STENCIL_STAR, cart->s, lx, ly, lz, &cart->vdm));
    }
    PetscCall(DMSetUp(cart->vdm));

    // TODO: allocate coordinates

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
                                                             " m %" PetscInt_FMT " n %" PetscInt_FMT " s %" PetscInt_FMT
                                                             "\n",
                                                             rank, cart->M, cart->N, m, n, cart->s));
                PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                             "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT
                                                             "\n",
                                                             info.xs, info.xs + info.xm, info.ys, info.ys + info.ym));
            } else {
                PetscCall(PetscViewerASCIISynchronizedPrintf(v,
                                                             "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT
                                                             " P %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT
                                                             " p %" PetscInt_FMT " s %" PetscInt_FMT "\n",
                                                             rank, cart->M, cart->N, cart->P, m, n, p, cart->s));
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
    cart->s = -1;
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

PetscErrorCode MeshCartesianSetStencilWidth(Mesh mesh, PetscInt s) {
    Mesh_Cartesian *cart = (Mesh_Cartesian *)mesh->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
    PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE,
               "This function must be called before MeshSetUp()");
    cart->s = s;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianGetInfo(Mesh mesh, PetscInt *dim, PetscInt *M, PetscInt *N, PetscInt *P, PetscInt *m,
                                    PetscInt *n, PetscInt *p, PetscInt *s, MeshBoundaryType *bndx,
                                    MeshBoundaryType *bndy, MeshBoundaryType *bndz) {
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
    if (s)
        *s = cart->s;
    if (bndx)
        *bndx = cart->bndx;
    if (bndy)
        *bndy = cart->bndy;
    if (bndz)
        *bndz = cart->bndz;

    PetscFunctionReturn(PETSC_SUCCESS);
}
