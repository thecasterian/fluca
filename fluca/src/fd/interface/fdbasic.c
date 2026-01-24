#include <fluca/private/flucafdimpl.h>
#include <flucaviewer.h>

PetscClassId FLUCAFD_CLASSID = 0;

PetscFunctionList FlucaFDList              = NULL;
PetscBool         FlucaFDRegisterAllCalled = PETSC_FALSE;

const char *FlucaFDDirections[]             = {"X", "Y", "Z", "FlucaFDDirection", "", NULL};
const char *FlucaFDBoundaryConditionTypes[] = {"NONE", "DIRICHLET", "NEUMANN", "PERIODIC", "FlucaFDBoundaryConditionType", "", NULL};

PetscErrorCode FlucaFDCreate(MPI_Comm comm, FlucaFD *fd)
{
  FlucaFD  f;
  PetscInt d;

  PetscFunctionBegin;
  PetscAssertPointer(fd, 2);

  PetscCall(FlucaFDInitializePackage());
  PetscCall(FlucaHeaderCreate(f, FLUCAFD_CLASSID, "FlucaFD", "Finite Difference", "FlucaFD", comm, FlucaFDDestroy, FlucaFDView));
  f->input_c    = 0;
  f->input_loc  = DMSTAG_ELEMENT;
  f->output_c   = 0;
  f->output_loc = DMSTAG_ELEMENT;
  f->cdm        = NULL;
  for (d = 0; d < 2 * FLUCAFD_MAX_DIM; ++d) {
    f->bcs[d].type  = FLUCAFD_BC_NONE;
    f->bcs[d].value = 0.;
  }
  f->dim = PETSC_DETERMINE;
  for (d = 0; d < FLUCAFD_MAX_DIM; ++d) {
    f->N[d]             = PETSC_DETERMINE;
    f->x[d]             = PETSC_DETERMINE;
    f->n[d]             = PETSC_DETERMINE;
    f->is_first_rank[d] = PETSC_FALSE;
    f->is_last_rank[d]  = PETSC_FALSE;
    f->arr_coord[d]     = NULL;
  }
  f->stencil_width   = PETSC_DETERMINE;
  f->slot_coord_prev = 0;
  f->slot_coord_elem = 0;
  f->termlink        = NULL;
  f->data            = NULL;
  f->setupcalled     = PETSC_FALSE;

  *fd = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetType(FlucaFD fd, FlucaFDType type)
{
  FlucaFDType old_type;
  PetscErrorCode (*impl_create)(FlucaFD);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);

  PetscCall(FlucaFDGetType(fd, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)fd, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(FlucaFDList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown FlucaFD type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(fd, destroy);
    PetscCall(PetscMemzero(fd->ops, sizeof(struct _FlucaFDOps)));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)fd, type));
  PetscCall((*impl_create)(fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDGetType(FlucaFD fd, FlucaFDType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscCall(FlucaFDRegisterAll());
  *type = ((PetscObject)fd)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDDestroy(FlucaFD *fd)
{
  PetscInt d;

  PetscFunctionBegin;
  if (!*fd) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*fd), FLUCAFD_CLASSID, 1);

  if (--((PetscObject)(*fd))->refct > 0) {
    *fd = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Restore coordinate arrays */
  for (d = 0; d < (*fd)->dim; ++d) {
    DM  subdm;
    Vec coordlocal;

    PetscCall(DMProductGetDM((*fd)->cdm, d, &subdm));
    PetscCall(DMGetCoordinatesLocal(subdm, &coordlocal));
    PetscCall(DMStagVecRestoreArrayRead(subdm, coordlocal, &(*fd)->arr_coord[d]));
  }
  PetscCall(DMDestroy(&(*fd)->cdm));

  /* Destroy term list */
  PetscCall(FlucaFDTermLinkDestroy_Internal(&(*fd)->termlink));

  /* Call type-specific destroy */
  PetscTryTypeMethod((*fd), destroy);

  PetscCall(PetscHeaderDestroy(fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDView(FlucaFD fd, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)fd), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(fd, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));

  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)fd, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Dimension: %" PetscInt_FMT "\n", fd->dim));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Input component: %" PetscInt_FMT ", stencil location: %s\n", fd->input_c, DMStagStencilLocations[fd->input_loc]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Output component: %" PetscInt_FMT ", stencil location: %s\n", fd->output_c, DMStagStencilLocations[fd->output_loc]));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }

  PetscTryTypeMethod(fd, view, viewer);

  if (isascii) {
    FlucaFDTermLink term;
    PetscInt        nterms, idx;
    char            deriv_order_str[FLUCAFD_MAX_DIM][16];
    char            accu_order_str[FLUCAFD_MAX_DIM][16];

    nterms = 0;
    for (term = fd->termlink; term; term = term->next) ++nterms;

    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Terms: %" PetscInt_FMT "\n", nterms));
    for (term = fd->termlink, idx = 0; term; term = term->next, ++idx) {
      for (PetscInt d = 0; d < FLUCAFD_MAX_DIM; ++d) {
        if (term->deriv_order[d] != -1) PetscCall(PetscSNPrintf(deriv_order_str[d], sizeof(deriv_order_str[d]), "%" PetscInt_FMT, term->deriv_order[d]));
        else PetscCall(PetscSNPrintf(deriv_order_str[d], sizeof(deriv_order_str[d]), "-"));
        if (term->accu_order[d] != PETSC_INT_MAX) PetscCall(PetscSNPrintf(accu_order_str[d], sizeof(accu_order_str[d]), "%" PetscInt_FMT, term->accu_order[d]));
        else PetscCall(PetscSNPrintf(accu_order_str[d], sizeof(accu_order_str[d]), "-"));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Term %" PetscInt_FMT ": deriv_order=(%s, %s, %s), accu_order=(%s, %s, %s), input=(%s, %" PetscInt_FMT ")\n", //
                                       idx, deriv_order_str[0], deriv_order_str[1], deriv_order_str[2], accu_order_str[0], accu_order_str[1], accu_order_str[2], DMStagStencilLocations[term->input_loc], term->input_c));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDViewFromOptions(FlucaFD fd, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  PetscCall(FlucaObjectViewFromOptions((PetscObject)fd, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDSetUp(FlucaFD fd)
{
  PetscBool isdmproduct;
  PetscInt  d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  if (fd->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  /* Validate coordinate DM */
  PetscCheck(fd->cdm, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Coordinate DM not set");
  PetscCall(PetscObjectTypeCompare((PetscObject)fd->cdm, DMPRODUCT, &isdmproduct));
  PetscCheck(isdmproduct, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONG, "Coordinate DM must be DMProduct (from DMStag)");

  /* Get dimension, sizes, and coordinate arrays/slots */
  PetscCall(DMGetDimension(fd->cdm, &fd->dim));
  for (d = 0; d < fd->dim; ++d) {
    DM  subdm;
    Vec coordlocal;

    PetscCall(DMProductGetDM(fd->cdm, d, &subdm));
    PetscCall(DMGetCoordinatesLocal(subdm, &coordlocal));
    PetscCall(DMStagGetGlobalSizes(subdm, &fd->N[d], NULL, NULL));
    PetscCall(DMStagGetCorners(subdm, &fd->x[d], NULL, NULL, &fd->n[d], NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMStagGetIsFirstRank(subdm, &fd->is_first_rank[d], NULL, NULL));
    PetscCall(DMStagGetIsLastRank(subdm, &fd->is_last_rank[d], NULL, NULL));
    PetscCall(DMStagVecGetArrayRead(subdm, coordlocal, &fd->arr_coord[d]));
    if (d == 0) {
      PetscCall(DMStagGetStencilWidth(subdm, &fd->stencil_width));
      PetscCall(DMStagGetLocationSlot(subdm, DMSTAG_LEFT, 0, &fd->slot_coord_prev));
      PetscCall(DMStagGetLocationSlot(subdm, DMSTAG_ELEMENT, 0, &fd->slot_coord_elem));
    }
  }

  /* Validate boundary conditions */
  for (d = 0; d < fd->dim; ++d) {
    /* Validate periodicity */
    PetscCheck((fd->bcs[2 * d].type == FLUCAFD_BC_PERIODIC) == (fd->bcs[2 * d + 1].type == FLUCAFD_BC_PERIODIC), PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONG, "A boundary is periodic while its opposite is non-periodic");
  }

  /* Call type-specific setup */
  PetscTryTypeMethod(fd, setup);

  fd->setupcalled = PETSC_TRUE;

  PetscCall(FlucaFDViewFromOptions(fd, NULL, "-flucafd_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
