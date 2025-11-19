#include <fluca/private/nslinearcnimpl.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

PetscErrorCode NSSetFromOptions_CNLinear(NS ns, PetscOptionItems PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "NSCNLinear Options");
  // TODO: Add options
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSCNLinearFormInitialGuess_Private(SNES snes, Vec x, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NSCNLinearPicardComputeFunction_Private(SNES snes, Vec x, Vec f, void *ctx)
{
  NS ns = (NS)ctx;

  PetscFunctionBegin;
  PetscCall(SNESPicardComputeFunction(snes, x, f, ctx));

  /* Remove null space */
  PetscAssert(ns->nullspace, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Null space must be set");
  PetscCall(MatNullSpaceRemove(ns->nullspace, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperatorFromDMToDM_Private(DM dmfrom, DM dmto, Mat *A)
{
  PetscInt               entriesfrom, entriesto;
  ISLocalToGlobalMapping ltogfrom, ltogto;
  MatType                mattype;

  PetscFunctionBegin;
  PetscCall(DMStagGetEntries(dmfrom, &entriesfrom));
  PetscCall(DMStagGetEntries(dmto, &entriesto));
  PetscCall(DMGetLocalToGlobalMapping(dmfrom, &ltogfrom));
  PetscCall(DMGetLocalToGlobalMapping(dmto, &ltogto));
  PetscCall(DMGetMatType(dmfrom, &mattype));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dmfrom), A));
  PetscCall(MatSetSizes(*A, entriesto, entriesfrom, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(*A, mattype));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltogto, ltogfrom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetup_CNLinear(NS ns)
{
  NS_CNLinear *cnl = (NS_CNLinear *)ns->data;
  MPI_Comm     comm;
  DM           vdm, Vdm;
  PetscInt     dim, nb, i;
  PetscBool    neednullspace, iscart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));

  /* Create intermediate solution vectors and spatial operators */
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_STAG_VECTOR, &Vdm));
  PetscCall(MeshGetDimension(ns->mesh, &dim));

  PetscCall(MeshCreateGlobalVector(ns->mesh, MESH_DM_STAG_VECTOR, &cnl->v0interp));
  PetscCall(MeshCreateGlobalVector(ns->mesh, MESH_DM_SCALAR, &cnl->phalf));
  PetscCall(CreateOperatorFromDMToDM_Private(vdm, Vdm, &cnl->B));

  PetscCall(PetscObjectSetName((PetscObject)cnl->phalf, "PressureHalfStep"));

  /* Preallocate Jacobian */
  if (iscart) PetscCall(NSCNLinearFormJacobian_Cart_Internal(ns->snes, ns->x, ns->J, ns->J, ns));
  else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");

  /* Create null space */
  neednullspace = PETSC_TRUE;
  PetscCall(MeshGetNumberBoundaries(ns->mesh, &nb));
  for (i = 0; i < nb; ++i) switch (ns->bcs[i].type) {
    case NS_BC_VELOCITY:
    case NS_BC_PERIODIC:
      /* Need null space */
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported boundary condition type");
    }
  if (neednullspace) {
    IS       is;
    Vec      vecs[1], subvec;
    PetscInt subvecsize;

    PetscCall(NSGetField(ns, NS_FIELD_PRESSURE, NULL, NULL, &is));
    PetscCall(MatCreateVecs(ns->J, NULL, &vecs[0]));
    PetscCall(VecGetSubVector(vecs[0], is, &subvec));
    PetscCall(VecGetSize(subvec, &subvecsize));
    PetscCall(VecSet(subvec, 1. / PetscSqrtReal((PetscReal)subvecsize)));
    PetscCall(VecRestoreSubVector(vecs[0], is, &subvec));
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, 1, vecs, &ns->nullspace));
    PetscCall(VecDestroy(&vecs[0]));
  }

  /* Set solver functions */
  if (iscart) PetscCall(SNESSetPicard(ns->snes, ns->r, NSCNLinearFormFunction_Cart_Internal, ns->J, ns->J, NSCNLinearFormJacobian_Cart_Internal, ns));
  else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
  if (neednullspace) PetscCall(SNESSetFunction(ns->snes, ns->r, NSCNLinearPicardComputeFunction_Private, ns));
  /* Need zero initial guess to ensure least-square solution of pressure poisson equation */
  PetscCall(SNESSetComputeInitialGuess(ns->snes, NSCNLinearFormInitialGuess_Private, NULL));

  /* Set KSP options */
  {
    KSP ksp;
    PetscCall(SNESGetKSP(ns->snes, &ksp));
    PetscCall(KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, 100));
    PetscCall(KSPSetFromOptions(ksp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSIterate_CNLinear(NS ns)
{
  PetscInt  dim;
  PetscBool iscart;

  PetscFunctionBegin;
  PetscCall(MeshGetDimension(ns->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));
  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSCNLinearIterate2d_Cart_Internal(ns));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSDestroy_CNLinear(NS ns)
{
  NS_CNLinear *cnl = (NS_CNLinear *)ns->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cnl->v0interp));
  PetscCall(VecDestroy(&cnl->phalf));
  PetscCall(MatDestroy(&cnl->B));
  PetscCall(PetscFree(ns->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView_CNLinear(NS ns, PetscViewer viewer)
{
  PetscFunctionBegin;
  // TODO: add view
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution_CNLinear(NS ns, PetscViewer viewer)
{
  NS_CNLinear *cnl = (NS_CNLinear *)ns->data;

  PetscFunctionBegin;
  PetscCall(VecView(cnl->phalf, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSLoadSolution_CNLinear(NS ns, PetscViewer viewer)
{
  NS_CNLinear *cnl = (NS_CNLinear *)ns->data;

  PetscFunctionBegin;
  PetscCall(FlucaVecLoad(cnl->phalf, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCreate_CNLinear(NS ns)
{
  NS_CNLinear *cnl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cnl));
  ns->data = (void *)cnl;

  cnl->v0interp  = NULL;
  cnl->phalf     = NULL;
  cnl->B         = NULL;
  cnl->Bcomputed = PETSC_FALSE;

  ns->ops->setfromoptions = NSSetFromOptions_CNLinear;
  ns->ops->setup          = NSSetup_CNLinear;
  ns->ops->iterate        = NSIterate_CNLinear;
  ns->ops->destroy        = NSDestroy_CNLinear;
  ns->ops->view           = NSView_CNLinear;
  ns->ops->viewsolution   = NSViewSolution_CNLinear;
  ns->ops->loadsolution   = NSLoadSolution_CNLinear;
  PetscFunctionReturn(PETSC_SUCCESS);
}
