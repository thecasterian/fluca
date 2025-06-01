#include <fluca/private/meshcartimpl.h>
#include <fluca/private/meshimpl.h>
#include <petsc/private/petscimpl.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

extern PetscErrorCode MeshView_CartCGNS(Mesh mesh, PetscViewer v);

const char *MeshCartCoordinateStencilLocations[] = {"PREV", "NEXT", "MeshCartCoordinateStencilLocation", "", NULL};

PetscLogEvent MESHCART_CreateFromFile;

PetscErrorCode MeshSetFromOptions_Cart(Mesh mesh, PetscOptionItems PetscOptionsObject)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;
  char       opt[PETSC_MAX_OPTION_NAME];
  char       text[PETSC_MAX_PATH_LEN];
  PetscInt   nRefine = 0, i, d;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MeshCart Options");
  for (d = 0; d < mesh->dim; ++d) {
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-cart_grid_%c", 'x' + d));
    PetscCall(PetscSNPrintf(text, PETSC_MAX_PATH_LEN, "Number of elements in the %c direction", 'x' + d));
    PetscCall(PetscOptionsBoundedInt(opt, text, "MeshCartSetGlobalSizes", cart->N[d], &cart->N[d], NULL, 1));
  }
  for (d = 0; d < mesh->dim; ++d) {
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-cart_ranks_%c", 'x' + d));
    PetscCall(PetscSNPrintf(text, PETSC_MAX_PATH_LEN, "Number of ranks in the %c direction", 'x' + d));
    PetscCall(PetscOptionsBoundedInt(opt, text, "MeshCartSetNumRanks", cart->nRanks[d], &cart->nRanks[d], NULL, PETSC_DECIDE));
  }
  for (d = 0; d < mesh->dim; ++d) {
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-cart_boundary_type_%c", 'x' + d));
    PetscCall(PetscSNPrintf(text, PETSC_MAX_PATH_LEN, "Boundary type in the %c direction", 'x' + d));
    PetscCall(PetscOptionsEnum(opt, text, "MeshCartSetBoundaryTypes", MeshBoundaryTypes, (PetscEnum)cart->bndTypes[d], (PetscEnum *)&cart->bndTypes[d], NULL));
  }
  for (d = 0; d < mesh->dim; ++d) {
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-cart_refine_%c", 'x' + d));
    PetscCall(PetscSNPrintf(text, PETSC_MAX_PATH_LEN, "Refinement factor in the %c direction", 'x' + d));
    PetscCall(PetscOptionsBoundedInt(opt, text, "MeshCartSetRefinementFactor", cart->refineFactor[d], &cart->refineFactor[d], NULL, 1));
  }
  PetscCall(PetscOptionsBoundedInt("-cart_refine", "Refine grid one or more times", "None", nRefine, &nRefine, NULL, 0));
  PetscOptionsHeadEnd();

  for (d = 0; d < mesh->dim; ++d) {
    PetscInt refineFactorTotal = 1;

    for (i = 0; i < nRefine; ++i) refineFactorTotal *= cart->refineFactor[d];
    cart->N[d] *= refineFactorTotal;
    if (cart->l[d])
      for (i = 0; i < cart->nRanks[d]; ++i) cart->l[d][i] *= refineFactorTotal;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetUp_Cart(Mesh mesh)
{
  Mesh_Cart      *cart = (Mesh_Cart *)mesh->data;
  MPI_Comm        comm;
  DMBoundaryType  dmBndTypes[3];
  const PetscInt *l[3];
  PetscInt        d;
  const PetscInt  dofElem = 1, dofFace = 1, stencilWidth = 1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

  for (d = 0; d < mesh->dim; ++d) {
    switch (cart->bndTypes[d]) {
    case MESH_BOUNDARY_NONE:
      dmBndTypes[d] = DM_BOUNDARY_GHOSTED;
      break;
    case MESH_BOUNDARY_PERIODIC:
      dmBndTypes[d] = DM_BOUNDARY_PERIODIC;
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid boundary type %d for dimension %" PetscInt_FMT, cart->bndTypes[d], d);
    }
  }

  /* Allocate DMs */
  switch (mesh->dim) {
  case 2:
    PetscCall(DMStagCreate2d(comm, dmBndTypes[0], dmBndTypes[1], cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1], 0, 0, dofElem, DMSTAG_STENCIL_STAR, stencilWidth, cart->l[0], cart->l[1], &mesh->dm));
    break;
  case 3:
    PetscCall(DMStagCreate3d(comm, dmBndTypes[0], dmBndTypes[1], dmBndTypes[2], cart->N[0], cart->N[1], cart->N[2], cart->nRanks[0], cart->nRanks[1], cart->nRanks[2], 0, 0, 0, dofElem, DMSTAG_STENCIL_STAR, stencilWidth, cart->l[0], cart->l[1], cart->l[2],
                             &mesh->dm));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
  }
  PetscCall(DMStagSetRefinementFactor(mesh->dm, cart->refineFactor[0], cart->refineFactor[1], cart->refineFactor[2]));
  PetscCall(DMSetUp(mesh->dm));
  switch (mesh->dim) {
  case 2:
    PetscCall(DMStagCreateCompatibleDMStag(mesh->dm, 0, dofFace, 0, 0, &mesh->fdm));
    break;
  case 3:
    PetscCall(DMStagCreateCompatibleDMStag(mesh->dm, 0, 0, dofFace, 0, &mesh->fdm));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
  }

  PetscCall(DMStagGetNumRanks(mesh->dm, &cart->nRanks[0], &cart->nRanks[1], &cart->nRanks[2]));
  for (d = 0; d < mesh->dim; ++d)
    if (!cart->l[d]) PetscCall(PetscMalloc1(cart->nRanks[d], &cart->l[d]));
  PetscCall(DMStagGetOwnershipRanges(mesh->dm, &l[0], &l[1], &l[2]));
  for (d = 0; d < mesh->dim; ++d) PetscCall(PetscArraycpy(cart->l[d], l[d], cart->nRanks[d]));

  PetscCall(DMStagSetUniformCoordinatesProduct(mesh->dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy_Cart(Mesh mesh)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;
  PetscInt   d;

  PetscFunctionBegin;
  for (d = 0; d < mesh->dim; ++d) PetscCall(PetscFree(cart->l[d]));
  PetscCall(DMDestroy(&mesh->dm));
  PetscCall(DMDestroy(&mesh->fdm));
  PetscCall(PetscFree(mesh->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView_Cart(Mesh mesh, PetscViewer v)
{
  Mesh_Cart  *cart = (Mesh_Cart *)mesh->data;
  PetscMPIInt rank;
  PetscBool   isascii, iscgns;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mesh), &rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERFLUCACGNS, &iscgns));

  if (isascii) {
    PetscInt x, y, z, m, n, p;

    PetscCall(DMStagGetCorners(mesh->dm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    switch (mesh->dim) {
    case 2:
      PetscCall(PetscViewerASCIISynchronizedPrintf(v, "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT "\n", rank, cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(v, "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT "\n", x, x + m, y, y + n));
      break;
    case 3:
      PetscCall(PetscViewerASCIISynchronizedPrintf(v, "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT " P %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT " p %" PetscInt_FMT "\n", rank, cart->N[0], cart->N[1], cart->N[1], cart->nRanks[0],
                                                   cart->nRanks[1], cart->nRanks[2]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(v, "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Z range of indices: %" PetscInt_FMT " %" PetscInt_FMT "\n", x, x + m, y, y + n, z, z + p));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  } else if (iscgns) {
    PetscCall(MeshView_CartCGNS(mesh, v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetNumberBoundaries_Cart(Mesh mesh, PetscInt *nb)
{
  PetscFunctionBegin;
  if (nb) *nb = 2 * mesh->dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCreate_Cart(Mesh mesh)
{
  Mesh_Cart *cart;
  PetscInt   d;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cart));
  mesh->data = (void *)cart;

  for (d = 0; d < MESH_MAX_DIM; ++d) {
    cart->N[d]            = -1;
    cart->nRanks[d]       = PETSC_DECIDE;
    cart->l[d]            = NULL;
    cart->bndTypes[d]     = MESH_BOUNDARY_NONE;
    cart->refineFactor[d] = 2;
  }
  mesh->ops->setfromoptions      = MeshSetFromOptions_Cart;
  mesh->ops->setup               = MeshSetUp_Cart;
  mesh->ops->destroy             = MeshDestroy_Cart;
  mesh->ops->view                = MeshView_Cart;
  mesh->ops->getnumberboundaries = MeshGetNumberBoundaries_Cart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartCreate2d(MPI_Comm comm, MeshBoundaryType bndx, MeshBoundaryType bndy, PetscInt M, PetscInt N, PetscInt m, PetscInt n, const PetscInt *lx, const PetscInt *ly, Mesh *mesh)
{
  PetscFunctionBegin;
  PetscCall(MeshCreate(comm, mesh));
  PetscCall(MeshSetType(*mesh, MESHCART));
  PetscCall(MeshSetDimension(*mesh, 2));
  PetscCall(MeshCartSetBoundaryTypes(*mesh, bndx, bndy, MESH_BOUNDARY_NONE));
  PetscCall(MeshCartSetGlobalSizes(*mesh, M, N, 1));
  PetscCall(MeshCartSetNumRanks(*mesh, m, n, PETSC_DECIDE));
  PetscCall(MeshCartSetOwnershipRanges(*mesh, lx, ly, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartCreate3d(MPI_Comm comm, MeshBoundaryType bndx, MeshBoundaryType bndy, MeshBoundaryType bndz, PetscInt M, PetscInt N, PetscInt P, PetscInt m, PetscInt n, PetscInt p, const PetscInt *lx, const PetscInt *ly, const PetscInt *lz, Mesh *mesh)
{
  PetscFunctionBegin;
  PetscCall(MeshCreate(comm, mesh));
  PetscCall(MeshSetType(*mesh, MESHCART));
  PetscCall(MeshSetDimension(*mesh, 3));
  PetscCall(MeshCartSetBoundaryTypes(*mesh, bndx, bndy, bndz));
  PetscCall(MeshCartSetGlobalSizes(*mesh, M, N, P));
  PetscCall(MeshCartSetNumRanks(*mesh, m, n, p));
  PetscCall(MeshCartSetOwnershipRanges(*mesh, lx, ly, lz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartSetGlobalSizes(Mesh mesh, PetscInt M, PetscInt N, PetscInt P)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called before MeshSetUp()");
  cart->N[0] = M;
  cart->N[1] = N;
  cart->N[2] = P;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetGlobalSizes(Mesh mesh, PetscInt *M, PetscInt *N, PetscInt *P)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (M) *M = cart->N[0];
  if (N) *N = cart->N[1];
  if (P) *P = cart->N[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartSetNumRanks(Mesh mesh, PetscInt m, PetscInt n, PetscInt p)
{
  Mesh_Cart *cart      = (Mesh_Cart *)mesh->data;
  PetscInt   nRanks[3] = {m, n, p};
  PetscInt   d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called before MeshSetUp()");
  for (d = 0; d < mesh->dim; ++d)
    if (cart->l[d]) PetscCheck(cart->nRanks[d] != nRanks[d], PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Cannot set number of procs after setting ownership ranges, or reset ownership ranges");
  cart->nRanks[0] = m;
  cart->nRanks[1] = n;
  cart->nRanks[2] = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetNumRanks(Mesh mesh, PetscInt *m, PetscInt *n, PetscInt *p)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (m) *m = cart->nRanks[0];
  if (n) *n = cart->nRanks[1];
  if (p) *p = cart->nRanks[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartSetBoundaryTypes(Mesh mesh, MeshBoundaryType bndx, MeshBoundaryType bndy, MeshBoundaryType bndz)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called before MeshSetUp()");
  cart->bndTypes[0] = bndx;
  cart->bndTypes[1] = bndy;
  cart->bndTypes[2] = bndz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetBoundaryTypes(Mesh mesh, MeshBoundaryType *bndx, MeshBoundaryType *bndy, MeshBoundaryType *bndz)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (bndx) *bndx = cart->bndTypes[0];
  if (bndy) *bndy = cart->bndTypes[1];
  if (bndz) *bndz = cart->bndTypes[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartSetOwnershipRanges(Mesh mesh, const PetscInt lx[], const PetscInt ly[], const PetscInt lz[])
{
  Mesh_Cart      *cart   = (Mesh_Cart *)mesh->data;
  const PetscInt *lin[3] = {lx, ly, lz};
  PetscInt        d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called before MeshSetUp()");

  for (d = 0; d < mesh->dim; ++d) {
    if (lin[d]) {
      PetscCheck(cart->nRanks[d] >= 0, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Cannot set ownership ranges before setting number of procs");
      if (!cart->l[d]) PetscCall(PetscMalloc1(cart->nRanks[d], &cart->l[d]));
      PetscCall(PetscArraycpy(cart->l[d], lin[d], cart->nRanks[d]));
    } else {
      PetscCall(PetscFree(cart->l[d]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetOwnershipRanges(Mesh mesh, const PetscInt *lx[], const PetscInt *ly[], const PetscInt *lz[])
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (lx) *lx = cart->l[0];
  if (ly) *ly = cart->l[1];
  if (lz) *lz = cart->l[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartSetRefinementFactor(Mesh mesh, PetscInt refine_x, PetscInt refine_y, PetscInt refine_z)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called before MeshSetUp()");
  if (refine_x > 0) cart->refineFactor[0] = refine_x;
  if (refine_y > 0) cart->refineFactor[1] = refine_y;
  if (refine_z > 0) cart->refineFactor[2] = refine_z;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetRefinementFactor(Mesh mesh, PetscInt *refine_x, PetscInt *refine_y, PetscInt *refine_z)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  if (refine_x) *refine_x = cart->refineFactor[0];
  if (refine_y) *refine_y = cart->refineFactor[1];
  if (refine_z) *refine_z = cart->refineFactor[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartSetUniformCoordinates(Mesh mesh, PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax, PetscReal zmin, PetscReal zmax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called after MeshSetUp()");

  PetscCall(DMStagSetUniformCoordinatesProduct(mesh->dm, xmin, xmax, ymin, ymax, zmin, zmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetCoordinateArrays(Mesh mesh, PetscScalar ***ax, PetscScalar ***ay, PetscScalar ***az)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetProductCoordinateArrays(mesh->dm, ax, ay, az));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetCoordinateArraysRead(Mesh mesh, const PetscScalar ***ax, const PetscScalar ***ay, const PetscScalar ***az)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, ax, ay, az));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartRestoreCoordinateArrays(Mesh mesh, PetscScalar ***ax, PetscScalar ***ay, PetscScalar ***az)
{
  PetscScalar ***a[3] = {ax, ay, az};
  PetscInt       x[3], m[3];
  PetscInt       i, d, iprev, inext, icenter;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

  PetscCall(DMStagGetCorners(mesh->dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_RIGHT, &inext));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &icenter));
  for (d = 0; d < mesh->dim; ++d)
    for (i = x[d]; i < x[d] + m[d]; ++i) (*a[d])[i][icenter] = ((*a[d])[i][iprev] + (*a[d])[i][inext]) / 2.0;

  PetscCall(DMStagRestoreProductCoordinateArrays(mesh->dm, ax, ay, az));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartRestoreCoordinateArraysRead(Mesh mesh, const PetscScalar ***arrx, const PetscScalar ***arry, const PetscScalar ***arrz)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, arrx, arry, arrz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetCoordinateLocationSlot(Mesh mesh, MeshCartCoordinateStencilLocation loc, PetscInt *slot)
{
  DMStagStencilLocation stagLoc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  switch (loc) {
  case MESHCART_PREV:
    stagLoc = DMSTAG_LEFT;
    break;
  case MESHCART_NEXT:
    stagLoc = DMSTAG_RIGHT;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONG, "Invalid coordinate stencil location");
  }
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, stagLoc, slot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetLocalSizes(Mesh mesh, PetscInt *m, PetscInt *n, PetscInt *p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetLocalSizes(mesh->dm, m, n, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetCorners(Mesh mesh, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetCorners(mesh->dm, x, y, z, m, n, p, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetIsFirstRank(Mesh mesh, PetscBool *isFirstRankX, PetscBool *isFirstRankY, PetscBool *isFirstRankZ)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetIsFirstRank(mesh->dm, isFirstRankX, isFirstRankY, isFirstRankZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetIsLastRank(Mesh mesh, PetscBool *isLastRankX, PetscBool *isLastRankY, PetscBool *isLastRankZ)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetIsLastRank(mesh->dm, isLastRankX, isLastRankY, isLastRankZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckExtension_Private(const char filename[], const char extension[], PetscBool *is_extension)
{
  size_t len_filename, len_extension;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(filename, &len_filename));
  PetscCall(PetscStrlen(extension, &len_extension));
  if (len_filename < len_extension) {
    *is_extension = PETSC_FALSE;
  } else {
    PetscCall(PetscStrcmp(filename + len_filename - len_extension, extension, is_extension));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartCreateFromFile(MPI_Comm comm, const char filename[], const char meshname[], Mesh *mesh)
{
  const char *ext_cgns = ".cgns";
  PetscBool   is_cgns;
  size_t      len = 0;

  PetscFunctionBegin;
  PetscAssertPointer(filename, 2);
  if (meshname) PetscAssertPointer(meshname, 3);
  PetscAssertPointer(mesh, 4);

  PetscCall(MeshInitializePackage());
  PetscCall(PetscLogEventBegin(MESHCART_CreateFromFile, 0, 0, 0, 0));

  PetscCall(CheckExtension_Private(filename, ext_cgns, &is_cgns));
  if (is_cgns) PetscCall(MeshCartCreateCGNSFromFile(comm, filename, mesh));
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot load file %s: unrecognized extension", filename);

  PetscCall(PetscStrlen(meshname, &len));
  if (len) PetscCall(PetscObjectSetName((PetscObject)*mesh, meshname));

  PetscCall(PetscLogEventEnd(MESHCART_CreateFromFile, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
