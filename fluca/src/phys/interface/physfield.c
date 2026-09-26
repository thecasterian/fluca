#include <fluca/private/physimpl.h>

const char *PhysFieldLocations[] = {"ELEMENT", "FACE", "PhysFieldLocation", "PHYS_FIELD_", NULL};

/* Face stencil locations indexed by direction: LEFT for x, DOWN for y, BACK for z */
static const DMStagStencilLocation face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* Empty the field table and destroy the solution DM, so that PhysSetUp() starts from scratch */
PetscErrorCode PhysResetFields_Internal(Phys phys)
{
  PetscInt f;

  PetscFunctionBegin;
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PetscFree(phys->fields[f].name));
    PetscCall(ISDestroy(&phys->fields[f].is));
  }
  phys->nfields = 0;
  PetscCall(DMDestroy(&phys->sol_dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Declare a solution field; called by the subtype's setup while PhysSetUp() runs */
PetscErrorCode PhysDeclareField_Internal(Phys phys, const char name[], PhysFieldLocation loc, PetscInt ncomp)
{
  PhysField *field;
  PetscInt   f, c0 = 0;
  PetscBool  same;

  PetscFunctionBegin;
  PetscCheck(!phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Cannot declare field %s after PhysSetUp()", name);
  PetscCheck(phys->nfields < PHYS_MAX_FIELDS, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Cannot declare more than %d fields", PHYS_MAX_FIELDS);
  PetscCheck(ncomp > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Field %s must have at least one component", name);
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PetscStrcmp(phys->fields[f].name, name, &same));
    PetscCheck(!same, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Field %s is already declared", name);
    if (phys->fields[f].loc == loc) c0 += phys->fields[f].ncomp;
  }
  field = &phys->fields[phys->nfields];
  PetscCall(PetscStrallocpy(name, &field->name));
  field->loc   = loc;
  field->c0    = c0;
  field->ncomp = ncomp;
  field->is    = NULL;
  ++phys->nfields;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One DMStag with the DOFs of every declared field, sharing the base DM's coordinates */
PetscErrorCode PhysCreateSolutionDM_Internal(Phys phys)
{
  PetscInt dof[2] = {0, 0};
  PetscInt f;
  DM       cdm;

  PetscFunctionBegin;
  for (f = 0; f < phys->nfields; ++f) dof[phys->fields[f].loc] += phys->fields[f].ncomp;
  switch (phys->dim) {
  case 2:
    PetscCall(DMStagCreateCompatibleDMStag(phys->base_dm, 0, dof[PHYS_FIELD_FACE], dof[PHYS_FIELD_ELEMENT], 0, &phys->sol_dm));
    break;
  case 3:
    PetscCall(DMStagCreateCompatibleDMStag(phys->base_dm, 0, 0, dof[PHYS_FIELD_FACE], dof[PHYS_FIELD_ELEMENT], &phys->sol_dm));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, phys->dim);
  }
  PetscCall(DMStagSetCoordinateDMType(phys->sol_dm, DMPRODUCT));
  PetscCall(DMGetCoordinateDM(phys->base_dm, &cdm));
  PetscCall(DMSetCoordinateDM(phys->sol_dm, cdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysFindField_Private(Phys phys, const char name[], PhysField **field)
{
  PetscInt  f;
  PetscBool same = PETSC_FALSE;

  PetscFunctionBegin;
  *field = NULL;
  for (f = 0; f < phys->nfields && !same; ++f) {
    PetscCall(PetscStrcmp(phys->fields[f].name, name, &same));
    if (same) *field = &phys->fields[f];
  }
  PetscCheck(*field, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Field %s is not declared", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetNumFields(Phys phys, PetscInt *nfields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(nfields, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetNumFields()");
  *nfields = phys->nfields;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetFieldName(Phys phys, PetscInt idx, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 3);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetFieldName()");
  PetscCheck(idx >= 0 && idx < phys->nfields, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Field index %" PetscInt_FMT " is out of range [0, %" PetscInt_FMT ")", idx, phys->nfields);
  *name = phys->fields[idx].name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetField(Phys phys, const char name[], PhysFieldLocation *loc, PetscInt *c0, PetscInt *ncomp)
{
  PhysField *field;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetField()");
  PetscCall(PhysFindField_Private(phys, name, &field));
  if (loc) *loc = field->loc;
  if (c0) *c0 = field->c0;
  if (ncomp) *ncomp = field->ncomp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The entries of a field in the solution vector. The IS is owned by the Phys; do not destroy it. */
PetscErrorCode PhysGetFieldIS(Phys phys, const char name[], IS *is)
{
  PhysField     *field;
  DMStagStencil *st;
  PetscInt       n = 0, nloc, d, c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(is, 3);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetFieldIS()");
  PetscCall(PhysFindField_Private(phys, name, &field));
  if (!field->is) {
    nloc = field->loc == PHYS_FIELD_FACE ? phys->dim : 1;
    PetscCall(PetscMalloc1(nloc * field->ncomp, &st));
    for (d = 0; d < nloc; ++d) {
      for (c = 0; c < field->ncomp; ++c) {
        st[n].i   = 0;
        st[n].j   = 0;
        st[n].k   = 0;
        st[n].loc = field->loc == PHYS_FIELD_FACE ? face_loc[d] : DMSTAG_ELEMENT;
        st[n].c   = field->c0 + c;
        ++n;
      }
    }
    PetscCall(DMStagCreateISFromStencils(phys->sol_dm, n, st, &field->is));
    PetscCall(PetscFree(st));
  }
  *is = field->is;
  PetscFunctionReturn(PETSC_SUCCESS);
}
