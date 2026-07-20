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
  PetscBool    iscart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));

  /* Create intermediate solution vectors and spatial operators */
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_VECTOR, &vdm));
  PetscCall(MeshGetDM(ns->mesh, MESH_DM_STAG_VECTOR, &Vdm));

  PetscCall(MeshCreateGlobalVector(ns->mesh, MESH_DM_STAG_VECTOR, &cnl->v0interp));
  PetscCall(MeshCreateGlobalVector(ns->mesh, MESH_DM_SCALAR, &cnl->phalf));
  PetscCall(CreateOperatorFromDMToDM_Private(vdm, Vdm, &cnl->B));

  PetscCall(PetscObjectSetName((PetscObject)cnl->phalf, "PressureHalfStep"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSStep_CNLinear(NS ns)
{
  PetscInt  dim;
  PetscBool iscart;

  PetscFunctionBegin;
  PetscCall(MeshGetDimension(ns->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));
  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSStep_CNLinear_Cart2d_Internal(ns));
    else SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
    break;
  case 3:
    if (iscart) PetscCall(NSStep_CNLinear_Cart3d_Internal(ns));
    else SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFormJacobian_CNLinear(NS ns, Vec x, Mat J, NSFormJacobianType type)
{
  PetscInt  dim;
  PetscBool iscart;

  PetscFunctionBegin;
  PetscCall(MeshGetDimension(ns->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));
  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSFormJacobian_CNLinear_Cart2d_Internal(ns, x, J, type));
    else SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
    break;
  case 3:
    if (iscart) PetscCall(NSFormJacobian_CNLinear_Cart3d_Internal(ns, x, J, type));
    else SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFormFunction_CNLinear(NS ns, Vec x, Vec f)
{
  PetscInt  dim;
  PetscBool iscart;

  PetscFunctionBegin;
  PetscCall(MeshGetDimension(ns->mesh, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)ns->mesh, MESHCART, &iscart));
  switch (dim) {
  case 2:
    if (iscart) PetscCall(NSFormFunction_CNLinear_Cart2d_Internal(ns, x, f));
    else SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
    break;
  case 3:
    if (iscart) PetscCall(NSFormFunction_CNLinear_Cart3d_Internal(ns, x, f));
    else SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONG, "Unsupported Mesh type");
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
  ns->ops->step           = NSStep_CNLinear;
  ns->ops->formjacobian   = NSFormJacobian_CNLinear;
  ns->ops->formfunction   = NSFormFunction_CNLinear;
  ns->ops->destroy        = NSDestroy_CNLinear;
  ns->ops->view           = NSView_CNLinear;
  ns->ops->viewsolution   = NSViewSolution_CNLinear;
  ns->ops->loadsolution   = NSLoadSolution_CNLinear;
  PetscFunctionReturn(PETSC_SUCCESS);
}
