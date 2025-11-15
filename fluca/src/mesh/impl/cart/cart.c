#include <fluca/private/meshcartimpl.h>
#include <fluca/private/meshimpl.h>
#include <petsc/private/petscimpl.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

const char *MeshCartBoundaryTypes[]              = {"NONE", "PERIODIC", "MeshCartBoundaryType", "", NULL};
const char *MeshCartBoundaryLocations[]          = {"LEFT", "RIGHT", "DOWN", "UP", "BACK", "FRONT", "MeshCartBoundaryLocation", "", NULL};
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
    PetscCall(PetscOptionsEnum(opt, text, "MeshCartSetBoundaryTypes", MeshCartBoundaryTypes, (PetscEnum)cart->bndTypes[d], (PetscEnum *)&cart->bndTypes[d], NULL));
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
  PetscInt        x[3], m[3];
  PetscScalar   **arrc[3];
  PetscInt        d, i, iprev, ielem;
  DM              cdm;
  const PetscInt  dofScalar = 1, dofVector = mesh->dim, stencilWidth = 1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));

  for (d = 0; d < mesh->dim; ++d) {
    switch (cart->bndTypes[d]) {
    case MESHCART_BOUNDARY_NONE:
      dmBndTypes[d] = DM_BOUNDARY_NONE;
      break;
    case MESHCART_BOUNDARY_PERIODIC:
      dmBndTypes[d] = DM_BOUNDARY_PERIODIC;
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid boundary type %d for dimension %" PetscInt_FMT, cart->bndTypes[d], d);
    }
  }

  /* Create scalar DM */
  switch (mesh->dim) {
  case 2:
    PetscCall(DMStagCreate2d(comm, dmBndTypes[0], dmBndTypes[1], cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1], 0, 0, dofScalar, DMSTAG_STENCIL_STAR, stencilWidth, cart->l[0], cart->l[1], &mesh->sdm));
    break;
  case 3:
    PetscCall(DMStagCreate3d(comm, dmBndTypes[0], dmBndTypes[1], dmBndTypes[2], cart->N[0], cart->N[1], cart->N[2], cart->nRanks[0], cart->nRanks[1], cart->nRanks[2], 0, 0, 0, dofScalar, DMSTAG_STENCIL_STAR, stencilWidth, cart->l[0], cart->l[1], cart->l[2],
                             &mesh->sdm));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
  }
  PetscCall(DMStagSetRefinementFactor(mesh->sdm, cart->refineFactor[0], cart->refineFactor[1], cart->refineFactor[2]));
  PetscCall(DMSetUp(mesh->sdm));

  PetscCall(DMStagGetNumRanks(mesh->sdm, &cart->nRanks[0], &cart->nRanks[1], &cart->nRanks[2]));
  for (d = 0; d < mesh->dim; ++d)
    if (!cart->l[d]) PetscCall(PetscMalloc1(cart->nRanks[d], &cart->l[d]));
  PetscCall(DMStagGetOwnershipRanges(mesh->sdm, &l[0], &l[1], &l[2]));
  for (d = 0; d < mesh->dim; ++d) PetscCall(PetscArraycpy(cart->l[d], l[d], cart->nRanks[d]));

  /* Create vector DM and face DMs */
  switch (mesh->dim) {
  case 2:
    PetscCall(DMStagCreateCompatibleDMStag(mesh->sdm, 0, 0, dofVector, 0, &mesh->vdm));
    PetscCall(DMStagCreateCompatibleDMStag(mesh->sdm, 0, dofScalar, 0, 0, &mesh->Sdm));
    PetscCall(DMStagCreateCompatibleDMStag(mesh->sdm, 0, dofVector, 0, 0, &mesh->Vdm));
    break;
  case 3:
    PetscCall(DMStagCreateCompatibleDMStag(mesh->sdm, 0, 0, 0, dofVector, &mesh->vdm));
    PetscCall(DMStagCreateCompatibleDMStag(mesh->sdm, 0, 0, dofScalar, 0, &mesh->Sdm));
    PetscCall(DMStagCreateCompatibleDMStag(mesh->sdm, 0, 0, dofVector, 0, &mesh->Vdm));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, mesh->dim);
  }

  PetscCall(DMSetMatrixPreallocateOnly(mesh->sdm, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(mesh->vdm, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(mesh->Sdm, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(mesh->Vdm, PETSC_TRUE));

  /* Set common coordinate DM */
  PetscCall(DMStagSetUniformCoordinatesProduct(mesh->sdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMStagGetCorners(mesh->sdm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArrays(mesh->sdm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->sdm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->sdm, DMSTAG_ELEMENT, &ielem));
  for (d = 0; d < mesh->dim; ++d)
    if (cart->coordLoaded[d]) {
      for (i = x[d]; i <= x[d] + m[d]; ++i) {
        arrc[d][i][iprev] = cart->coordLoaded[d][i];
        if (i < x[d] + m[d]) arrc[d][i][ielem] = (cart->coordLoaded[d][i] + cart->coordLoaded[d][i + 1]) / 2.;
      }
      // TODO: set ghost coordinates
    }
  PetscCall(DMStagRestoreProductCoordinateArrays(mesh->sdm, &arrc[0], &arrc[1], &arrc[2]));

  PetscCall(DMGetCoordinateDM(mesh->sdm, &cdm));
  PetscCall(DMStagSetCoordinateDMType(mesh->vdm, DMPRODUCT));
  PetscCall(DMSetCoordinateDM(mesh->vdm, cdm));
  PetscCall(DMStagSetCoordinateDMType(mesh->Sdm, DMPRODUCT));
  PetscCall(DMSetCoordinateDM(mesh->Sdm, cdm));
  PetscCall(DMStagSetCoordinateDMType(mesh->Vdm, DMPRODUCT));
  PetscCall(DMSetCoordinateDM(mesh->Vdm, cdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshDestroy_Cart(Mesh mesh)
{
  Mesh_Cart *cart = (Mesh_Cart *)mesh->data;
  PetscInt   d;

  PetscFunctionBegin;
  for (d = 0; d < mesh->dim; ++d) {
    PetscCall(PetscFree(cart->l[d]));
    PetscCall(PetscFree(cart->coordLoaded[d]));
  }
  PetscCall(DMDestroy(&mesh->sdm));
  PetscCall(DMDestroy(&mesh->vdm));
  PetscCall(DMDestroy(&mesh->Sdm));
  PetscCall(DMDestroy(&mesh->Vdm));
  PetscCall(PetscFree(mesh->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshView_Cart(Mesh mesh, PetscViewer viewer)
{
  Mesh_Cart  *cart = (Mesh_Cart *)mesh->data;
  PetscMPIInt rank;
  PetscBool   isascii, iscgns;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mesh), &rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));

  if (isascii) {
    PetscInt x, y, z, m, n, p;

    PetscCall(DMStagGetCorners(mesh->sdm, &x, &y, &z, &m, &n, &p, NULL, NULL, NULL));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    switch (mesh->dim) {
    case 2:
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT "\n", rank, cart->N[0], cart->N[1], cart->nRanks[0], cart->nRanks[1]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT "\n", x, x + m, y, y + n));
      break;
    case 3:
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Processor [%d] M %" PetscInt_FMT " N %" PetscInt_FMT " P %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT " p %" PetscInt_FMT "\n", rank, cart->N[0], cart->N[1], cart->N[1], cart->nRanks[0],
                                                   cart->nRanks[1], cart->nRanks[2]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Y range of indices: %" PetscInt_FMT " %" PetscInt_FMT ", Z range of indices: %" PetscInt_FMT " %" PetscInt_FMT "\n", x, x + m, y, y + n, z, z + p));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported mesh dimension");
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  } else if (iscgns) {
    PetscCall(MeshView_Cart_CGNS(mesh, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshLoad_Cart(Mesh mesh, PetscViewer viewer)
{
  PetscBool iscgns;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) PetscCall(MeshLoad_Cart_CGNS(mesh, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCreateGlobalVector_Cart(Mesh mesh, MeshDMType dmtype, Vec *vec)
{
  DM dm;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(mesh, dmtype, &dm));
  PetscCall(DMCreateGlobalVector(dm, vec));
  PetscCall(PetscObjectCompose((PetscObject)(*vec), "Fluca_Mesh", (PetscObject)mesh));
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void))VecView_Cart));
  PetscCall(VecSetOperation(*vec, VECOP_LOAD, (void (*)(void))VecLoad_Cart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCreateMatrix_Cart(Mesh mesh, MeshDMType rdmtype, MeshDMType cdmtype, Mat *mat)
{
  DM                     rdm, cdm;
  PetscInt               rentries, centries;
  ISLocalToGlobalMapping rltog, cltog;
  MatType                mattype;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(mesh, rdmtype, &rdm));
  PetscCall(MeshGetDM(mesh, cdmtype, &cdm));
  PetscCall(DMStagGetEntries(rdm, &rentries));
  PetscCall(DMStagGetEntries(cdm, &centries));
  PetscCall(DMGetLocalToGlobalMapping(rdm, &rltog));
  PetscCall(DMGetLocalToGlobalMapping(cdm, &cltog));
  PetscCall(DMGetMatType(rdm, &mattype));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)mesh), mat));
  PetscCall(MatSetSizes(*mat, rentries, centries, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(*mat, mattype));
  PetscCall(MatSetLocalToGlobalMapping(*mat, rltog, cltog));
  PetscCall(MatSetUp(*mat));
  PetscCall(MatSetOption(*mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
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
    cart->bndTypes[d]     = MESHCART_BOUNDARY_NONE;
    cart->refineFactor[d] = 2;
    cart->coordLoaded[d]  = NULL;
  }
  mesh->ops->setfromoptions      = MeshSetFromOptions_Cart;
  mesh->ops->setup               = MeshSetUp_Cart;
  mesh->ops->destroy             = MeshDestroy_Cart;
  mesh->ops->view                = MeshView_Cart;
  mesh->ops->load                = MeshLoad_Cart;
  mesh->ops->createglobalvector  = MeshCreateGlobalVector_Cart;
  mesh->ops->creatematrix        = MeshCreateMatrix_Cart;
  mesh->ops->getnumberboundaries = MeshGetNumberBoundaries_Cart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartCreate2d(MPI_Comm comm, MeshCartBoundaryType bndx, MeshCartBoundaryType bndy, PetscInt M, PetscInt N, PetscInt m, PetscInt n, const PetscInt *lx, const PetscInt *ly, Mesh *mesh)
{
  PetscFunctionBegin;
  PetscCall(MeshCreate(comm, mesh));
  PetscCall(MeshSetType(*mesh, MESHCART));
  PetscCall(MeshSetDimension(*mesh, 2));
  PetscCall(MeshCartSetBoundaryTypes(*mesh, bndx, bndy, MESHCART_BOUNDARY_NONE));
  PetscCall(MeshCartSetGlobalSizes(*mesh, M, N, 1));
  PetscCall(MeshCartSetNumRanks(*mesh, m, n, PETSC_DECIDE));
  PetscCall(MeshCartSetOwnershipRanges(*mesh, lx, ly, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartCreate3d(MPI_Comm comm, MeshCartBoundaryType bndx, MeshCartBoundaryType bndy, MeshCartBoundaryType bndz, PetscInt M, PetscInt N, PetscInt P, PetscInt m, PetscInt n, PetscInt p, const PetscInt *lx, const PetscInt *ly, const PetscInt *lz, Mesh *mesh)
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
  Mesh_Cart *cart     = (Mesh_Cart *)mesh->data;
  PetscInt   sizes[3] = {M, N, P};
  PetscInt   d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "This function must be called before MeshSetUp()");
  for (d = 0; d < mesh->dim; ++d)
    if (cart->coordLoaded[d]) PetscCheck(cart->N[d] != sizes[d], PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Cannot set global size after loading coordinates");
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

PetscErrorCode MeshCartSetBoundaryTypes(Mesh mesh, MeshCartBoundaryType bndx, MeshCartBoundaryType bndy, MeshCartBoundaryType bndz)
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

PetscErrorCode MeshCartGetBoundaryTypes(Mesh mesh, MeshCartBoundaryType *bndx, MeshCartBoundaryType *bndy, MeshCartBoundaryType *bndz)
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
  PetscCall(DMStagSetUniformCoordinatesProduct(mesh->sdm, xmin, xmax, ymin, ymax, zmin, zmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetCoordinateArrays(Mesh mesh, PetscScalar ***ax, PetscScalar ***ay, PetscScalar ***az)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetProductCoordinateArrays(mesh->sdm, ax, ay, az));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetCoordinateArraysRead(Mesh mesh, const PetscScalar ***ax, const PetscScalar ***ay, const PetscScalar ***az)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->sdm, ax, ay, az));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartRestoreCoordinateArrays(Mesh mesh, PetscScalar ***ax, PetscScalar ***ay, PetscScalar ***az)
{
  PetscScalar ***a[3] = {ax, ay, az};
  PetscInt       x[3], m[3];
  PetscInt       i, d, iprevc, inextc, ielemc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);

  PetscCall(DMStagGetCorners(mesh->sdm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->sdm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->sdm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->sdm, DMSTAG_ELEMENT, &ielemc));
  for (d = 0; d < mesh->dim; ++d)
    for (i = x[d]; i < x[d] + m[d]; ++i) (*a[d])[i][ielemc] = ((*a[d])[i][iprevc] + (*a[d])[i][inextc]) / 2.0;
  // TODO: set ghost coordinates

  PetscCall(DMStagRestoreProductCoordinateArrays(mesh->sdm, ax, ay, az));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartRestoreCoordinateArraysRead(Mesh mesh, const PetscScalar ***ax, const PetscScalar ***ay, const PetscScalar ***az)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->sdm, ax, ay, az));
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
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->sdm, stagLoc, slot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetLocalSizes(Mesh mesh, PetscInt *m, PetscInt *n, PetscInt *p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetLocalSizes(mesh->sdm, m, n, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetCorners(Mesh mesh, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetCorners(mesh->sdm, x, y, z, m, n, p, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetIsFirstRank(Mesh mesh, PetscBool *isFirstRankX, PetscBool *isFirstRankY, PetscBool *isFirstRankZ)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetIsFirstRank(mesh->sdm, isFirstRankX, isFirstRankY, isFirstRankZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetIsLastRank(Mesh mesh, PetscBool *isLastRankX, PetscBool *isLastRankY, PetscBool *isLastRankZ)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscCall(DMStagGetIsLastRank(mesh->sdm, isLastRankX, isLastRankY, isLastRankZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartGetBoundaryIndex(Mesh mesh, MeshCartBoundaryLocation loc, PetscInt *index)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  switch (loc) {
  case MESHCART_LEFT:
    *index = 0;
    break;
  case MESHCART_RIGHT:
    *index = 1;
    break;
  case MESHCART_DOWN:
    *index = 2;
    break;
  case MESHCART_UP:
    *index = 3;
    break;
  case MESHCART_BACK:
    *index = 4;
    break;
  case MESHCART_FRONT:
    *index = 5;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONG, "Invalid boundary location");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
