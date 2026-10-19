#include <fluca/private/flucafdimpl.h>

/* Domain boundary conditions of rows whose operand is NULL (all FLUCAFD_BC_NONE) */
static const FlucaFDBoundaryCondition nobcs[FLUCAFD_MAX_COMPONENT][2 * FLUCAFD_MAX_DIM];

static PetscErrorCode FlucaFDSetUp_IBSelect(FlucaFD fd)
{
  FlucaFD_IBSelect *sel = (FlucaFD_IBSelect *)fd->data;
  FlucaFDTermLink   src, dst;
  FlucaFD           op;
  PetscInt          type, nb, s[3];
  PetscInt          c, f;
  PetscBool         any = PETSC_FALSE, found;
  MeshIBPointType   ptype;

  PetscFunctionBegin;
  for (c = 0; c < FLUCAFD_MAX_COMPONENT; ++c) {
    for (f = 0; f < 2 * FLUCAFD_MAX_DIM; ++f)
      PetscCheck(fd->bcs[c][f].type == FLUCAFD_BC_NONE && !fd->bcs[c][f].fn, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Set domain boundary conditions on the operands, not on FLUCAFDIBSELECT");
  }
  for (type = 0; type < FLUCAFD_IB_NPOINTTYPES; ++type) {
    op = sel->operand[type];
    if (!op) continue;
    any = PETSC_TRUE;
    PetscCall(FlucaFDValidateOperand_Internal(fd, op));
    PetscCheck(op->input_loc == fd->input_loc && op->input_c == fd->input_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "%s operand input must match the select input", MeshIBPointTypes[type]);
    PetscCheck(op->output_loc == fd->output_loc && op->output_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "%s operand output must match the select output", MeshIBPointTypes[type]);
    for (src = op->termlink; src; src = src->next) {
      PetscCall(FlucaFDTermLinkFind_Internal(fd->termlink, src, &found));
      if (found) continue;
      PetscCall(FlucaFDTermLinkDuplicate_Internal(src, &dst));
      PetscCall(FlucaFDTermLinkAppend_Internal(&fd->termlink, dst));
    }
  }
  PetscCheck(any, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "No operand set");
  /* Fail in setup, not at the first stencil query, when the output location is not classified */
  PetscCall(MeshGetNumBodies(fd->mesh, &nb));
  if (nb) {
    PetscCall(DMStagGetCorners(fd->dm, &s[0], &s[1], &s[2], NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, s[0], fd->dim > 1 ? s[1] : 0, fd->dim > 2 ? s[2] : 0, &ptype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_IBSelect(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscBool pattern, PetscInt *npoints, FlucaFDStencilPoint points[])
{
  FlucaFD_IBSelect *sel = (FlucaFD_IBSelect *)fd->data;
  MeshIBPointType   type;

  PetscFunctionBegin;
  PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, i, j, k, &type)); /* FLUID everywhere when the Mesh has no bodies */
  if (sel->operand[type]) PetscCall(FlucaFDGetStencilRaw_Internal(sel->operand[type], i, j, k, pattern, npoints, points));
  else *npoints = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetBoundaryConditions_IBSelect(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, const FlucaFDBoundaryCondition (**bcs)[2 * FLUCAFD_MAX_DIM])
{
  FlucaFD_IBSelect *sel = (FlucaFD_IBSelect *)fd->data;
  MeshIBPointType   type;

  PetscFunctionBegin;
  PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, i, j, k, &type));
  /* Read the operand's table live, so boundary conditions set on it after setup take effect */
  if (sel->operand[type]) PetscCall(FlucaFDGetBoundaryConditions_Internal(sel->operand[type], i, j, k, bcs));
  else *bcs = nobcs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_IBSelect(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_IBSelect *sel = (FlucaFD_IBSelect *)fd->data;
  PetscInt          type;
  PetscBool         isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (!isascii) PetscFunctionReturn(PETSC_SUCCESS);
  for (type = 0; type < FLUCAFD_IB_NPOINTTYPES; ++type) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  %s:%s\n", MeshIBPointTypes[type], sel->operand[type] ? "" : " (none)"));
    if (!sel->operand[type]) continue;
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(FlucaFDView(sel->operand[type], viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_IBSelect(FlucaFD fd)
{
  FlucaFD_IBSelect *sel = (FlucaFD_IBSelect *)fd->data;
  PetscInt          type;

  PetscFunctionBegin;
  for (type = 0; type < FLUCAFD_IB_NPOINTTYPES; ++type) PetscCall(FlucaFDDestroy(&sel->operand[type]));
  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_IBSelect(FlucaFD fd)
{
  FlucaFD_IBSelect *sel;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sel));

  fd->data                       = (void *)sel;
  fd->ops->setup                 = FlucaFDSetUp_IBSelect;
  fd->ops->getstencilraw         = FlucaFDGetStencilRaw_IBSelect;
  fd->ops->getboundaryconditions = FlucaFDGetBoundaryConditions_IBSelect;
  fd->ops->destroy               = FlucaFDDestroy_IBSelect;
  fd->ops->view                  = FlucaFDView_IBSelect;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBSelectCreate(FlucaFD fluid, FlucaFD forcing, FlucaFD solid, FlucaFD *fd)
{
  FlucaFD  ops[FLUCAFD_IB_NPOINTTYPES];
  FlucaFD  first = NULL;
  MPI_Comm comm;
  PetscInt type;

  PetscFunctionBegin;
  if (fluid) PetscValidHeaderSpecific(fluid, FLUCAFD_CLASSID, 1);
  if (forcing) PetscValidHeaderSpecific(forcing, FLUCAFD_CLASSID, 2);
  if (solid) PetscValidHeaderSpecific(solid, FLUCAFD_CLASSID, 3);
  PetscAssertPointer(fd, 4);
  ops[MESH_IB_FLUID]   = fluid;
  ops[MESH_IB_FORCING] = forcing;
  ops[MESH_IB_SOLID]   = solid;
  for (type = 0; type < FLUCAFD_IB_NPOINTTYPES; ++type) {
    if (!ops[type]) continue;
    PetscCheck(ops[type]->setupcalled, PetscObjectComm((PetscObject)ops[type]), PETSC_ERR_ARG_WRONGSTATE, "%s operand must be set up before calling FlucaFDIBSelectCreate", MeshIBPointTypes[type]);
    if (!first) first = ops[type];
  }
  PetscCheck(first, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "At least one operand must be non-NULL");

  PetscCall(PetscObjectGetComm((PetscObject)first, &comm));
  PetscCall(FlucaFDCreate(comm, fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDIBSELECT));
  PetscCall(FlucaFDSetMesh(*fd, first->mesh));
  PetscCall(FlucaFDSetInputLocation(*fd, first->input_loc, first->input_c));
  PetscCall(FlucaFDSetOutputLocation(*fd, first->output_loc, first->output_c));
  for (type = 0; type < FLUCAFD_IB_NPOINTTYPES; ++type) PetscCall(FlucaFDIBSelectSetOperand(*fd, (MeshIBPointType)type, ops[type]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBSelectSetOperand(FlucaFD fd, MeshIBPointType type, FlucaFD operand)
{
  FlucaFD_IBSelect *sel;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDIBSELECT);
  sel = (FlucaFD_IBSelect *)fd->data;
  PetscCheck(type >= MESH_IB_FLUID && type <= MESH_IB_SOLID, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Invalid IB point type %d", (int)type);
  if (operand) {
    PetscValidHeaderSpecific(operand, FLUCAFD_CLASSID, 3);
    PetscCheckSameComm(fd, 1, operand, 3);
  }
  PetscCheck(!fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change an operand after FlucaFDSetUp()");
  if (operand) PetscCall(PetscObjectReference((PetscObject)operand));
  PetscCall(FlucaFDDestroy(&sel->operand[type]));
  sel->operand[type] = operand;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBSelectGetOperand(FlucaFD fd, MeshIBPointType type, FlucaFD *operand)
{
  FlucaFD_IBSelect *sel;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDIBSELECT);
  sel = (FlucaFD_IBSelect *)fd->data;
  PetscCheck(type >= MESH_IB_FLUID && type <= MESH_IB_SOLID, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Invalid IB point type %d", (int)type);
  PetscAssertPointer(operand, 3);
  *operand = sel->operand[type];
  PetscFunctionReturn(PETSC_SUCCESS);
}
