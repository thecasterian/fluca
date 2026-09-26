#include <fluca/private/physimpl.h>

const char *PhysFieldLocations[] = {"ELEMENT", "FACE", "PhysFieldLocation", "PHYS_FIELD_", NULL};

/* Face stencil locations indexed by direction: LEFT for x, DOWN for y, BACK for z */
static const DMStagStencilLocation face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* If the Phys is set up, destroy the solution DM and every cached field IS, and clear the setup state,
   so that a field declaration, removal or reset always leaves PhysSetUp() to run again */
static PetscErrorCode PhysClearSetUp_Private(Phys phys)
{
  PetscInt f;

  PetscFunctionBegin;
  if (!phys->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  for (f = 0; f < phys->nfields; ++f) PetscCall(ISDestroy(&phys->fields[f].is));
  PetscCall(DMDestroy(&phys->sol_dm));
  phys->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Resolve a field's component count, turning PETSC_DECIDE into one component per spatial dimension */
static PetscInt PhysFieldNComp_Private(Phys phys, const PhysField *field)
{
  return field->ncomp == PETSC_DECIDE ? phys->dim : field->ncomp;
}

/* Sum of the resolved component counts of the fields before fields[idx] at the same location */
static PetscInt PhysFieldC0_Private(Phys phys, PetscInt idx)
{
  PetscInt f, c0 = 0;

  for (f = 0; f < idx; ++f)
    if (phys->fields[f].loc == phys->fields[idx].loc) c0 += PhysFieldNComp_Private(phys, &phys->fields[f]);
  return c0;
}

static PetscErrorCode PhysFindField_Private(Phys phys, const char name[], PetscInt *idx)
{
  PetscInt  f;
  PetscBool same = PETSC_FALSE;

  PetscFunctionBegin;
  *idx = -1;
  for (f = 0; f < phys->nfields && !same; ++f) {
    PetscCall(PetscStrcmp(phys->fields[f].name, name, &same));
    if (same) *idx = f;
  }
  PetscCheck(*idx >= 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Field %s is not declared", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Append a solution field. May be called at any time; if the Phys is set up, this clears the setup state first */
PetscErrorCode PhysDeclareField(Phys phys, const char name[], PhysFieldLocation loc, PetscInt ncomp)
{
  PhysField *field;
  PetscInt   f;
  PetscBool  same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCheck(phys->nfields < PHYS_MAX_FIELDS, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Cannot declare more than %d fields", PHYS_MAX_FIELDS);
  PetscCheck(ncomp > 0 || ncomp == PETSC_DECIDE, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Field %s must have a positive component count or PETSC_DECIDE", name);
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PetscStrcmp(phys->fields[f].name, name, &same));
    PetscCheck(!same, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Field %s is already declared", name);
  }
  PetscCall(PhysClearSetUp_Private(phys));
  field = &phys->fields[phys->nfields];
  PetscCall(PetscStrallocpy(name, &field->name));
  field->loc   = loc;
  field->ncomp = ncomp;
  field->is    = NULL;
  ++phys->nfields;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Remove a field; later fields shift down keeping their order. May be called at any time */
PetscErrorCode PhysRemoveField(Phys phys, const char name[])
{
  PetscInt idx, f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCall(PhysFindField_Private(phys, name, &idx));
  PetscCall(PhysClearSetUp_Private(phys));
  PetscCall(PetscFree(phys->fields[idx].name));
  PetscCall(ISDestroy(&phys->fields[idx].is));
  for (f = idx; f < phys->nfields - 1; ++f) phys->fields[f] = phys->fields[f + 1];
  --phys->nfields;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Empty the field table. May be called at any time */
PetscErrorCode PhysResetFields(Phys phys)
{
  PetscInt f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCall(PhysClearSetUp_Private(phys));
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PetscFree(phys->fields[f].name));
    PetscCall(ISDestroy(&phys->fields[f].is));
  }
  phys->nfields = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One DMStag with the DOFs of every declared field, sharing the base DM's coordinates */
PetscErrorCode PhysCreateSolutionDM_Internal(Phys phys)
{
  PetscInt dof[2] = {0, 0};
  PetscInt f;
  DM       cdm;

  PetscFunctionBegin;
  for (f = 0; f < phys->nfields; ++f) dof[phys->fields[f].loc] += PhysFieldNComp_Private(phys, &phys->fields[f]);
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
  PetscInt idx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetField()");
  PetscCall(PhysFindField_Private(phys, name, &idx));
  if (loc) *loc = phys->fields[idx].loc;
  if (c0) *c0 = PhysFieldC0_Private(phys, idx);
  if (ncomp) *ncomp = PhysFieldNComp_Private(phys, &phys->fields[idx]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The entries of a field in the solution vector. The IS is owned by the Phys; do not destroy it. */
PetscErrorCode PhysGetFieldIS(Phys phys, const char name[], IS *is)
{
  PhysField     *field;
  DMStagStencil *st;
  PetscInt       n = 0, nloc, d, c, idx, c0, ncomp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(is, 3);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetFieldIS()");
  PetscCall(PhysFindField_Private(phys, name, &idx));
  field = &phys->fields[idx];
  if (!field->is) {
    ncomp = PhysFieldNComp_Private(phys, field);
    c0    = PhysFieldC0_Private(phys, idx);
    nloc  = field->loc == PHYS_FIELD_FACE ? phys->dim : 1;
    PetscCall(PetscMalloc1(nloc * ncomp, &st));
    for (d = 0; d < nloc; ++d) {
      for (c = 0; c < ncomp; ++c) {
        st[n].i   = 0;
        st[n].j   = 0;
        st[n].k   = 0;
        st[n].loc = field->loc == PHYS_FIELD_FACE ? face_loc[d] : DMSTAG_ELEMENT;
        st[n].c   = c0 + c;
        ++n;
      }
    }
    PetscCall(DMStagCreateISFromStencils(phys->sol_dm, n, st, &field->is));
    PetscCall(PetscFree(st));
  }
  *is = field->is;
  PetscFunctionReturn(PETSC_SUCCESS);
}
