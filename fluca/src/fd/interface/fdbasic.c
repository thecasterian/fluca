#include <fluca/private/flucafdimpl.h>
#include <flucaviewer.h>

PetscClassId FLUCAFD_CLASSID = 0;

PetscFunctionList FlucaFDList              = NULL;
PetscBool         FlucaFDRegisterAllCalled = PETSC_FALSE;

const char *FlucaFDDirections[]             = {"X", "Y", "Z", "FlucaFDDirection", "", NULL};
const char *FlucaFDBoundaryConditionTypes[] = {"NONE", "DIRICHLET", "NEUMANN", "FlucaFDBoundaryConditionType", "", NULL};

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
  f->dm         = NULL;
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
    f->periodic[d]      = PETSC_FALSE;
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
  PetscFunctionBegin;
  if (!*fd) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*fd), FLUCAFD_CLASSID, 1);

  if (--((PetscObject)(*fd))->refct > 0) {
    *fd = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Restore coordinate arrays (only if setup was called) */
  if ((*fd)->setupcalled) PetscCall(DMStagRestoreProductCoordinateArraysRead((*fd)->dm, &(*fd)->arr_coord[0], &(*fd)->arr_coord[1], &(*fd)->arr_coord[2]));
  PetscCall(DMDestroy(&(*fd)->dm));

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
  PetscBool      isdmstag;
  PetscInt       d;
  DMBoundaryType bt[FLUCAFD_MAX_DIM];
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);
  if (fd->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  /* Validate reference DM */
  PetscCheck(fd->dm, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Reference DM not set. Call FlucaFDSetDM() first");
  PetscCall(PetscObjectTypeCompare((PetscObject)fd->dm, DMSTAG, &isdmstag));
  PetscCheck(isdmstag, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONG, "Reference DM must be DMStag");

  /* Get grid info directly from DMStag */
  PetscCall(DMGetDimension(fd->dm, &fd->dim));
  PetscCall(DMStagGetGlobalSizes(fd->dm, &fd->N[0], &fd->N[1], &fd->N[2]));
  PetscCall(DMStagGetCorners(fd->dm, &fd->x[0], &fd->x[1], &fd->x[2], &fd->n[0], &fd->n[1], &fd->n[2], NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(fd->dm, &fd->is_first_rank[0], &fd->is_first_rank[1], &fd->is_first_rank[2]));
  PetscCall(DMStagGetIsLastRank(fd->dm, &fd->is_last_rank[0], &fd->is_last_rank[1], &fd->is_last_rank[2]));
  PetscCall(DMStagGetStencilWidth(fd->dm, &fd->stencil_width));

  /* Get coordinate arrays and slots directly from DMStag */
  PetscCall(DMStagGetProductCoordinateArraysRead(fd->dm, &fd->arr_coord[0], &fd->arr_coord[1], &fd->arr_coord[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(fd->dm, DMSTAG_LEFT, &fd->slot_coord_prev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(fd->dm, DMSTAG_ELEMENT, &fd->slot_coord_elem));

  /* Query periodicity from DMStag */
  PetscCall(DMStagGetBoundaryTypes(fd->dm, &bt[0], &bt[1], &bt[2]));
  for (d = 0; d < fd->dim; ++d) fd->periodic[d] = (bt[d] == DM_BOUNDARY_PERIODIC);

  /* Call type-specific setup */
  PetscTryTypeMethod(fd, setup);

  fd->setupcalled = PETSC_TRUE;

  PetscCall(FlucaFDViewFromOptions(fd, NULL, "-flucafd_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
