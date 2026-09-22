#include <fluca/private/physimpl.h>

const char *PhysFieldLocations[] = {"ELEMENT", "FACE", "PhysFieldLocation", "PHYS_FIELD_", NULL};
const char *PhysEquationRoles[]  = {"MOMENTUM", "PRESSURE", "TRANSPORTED_SCALAR", "AUXILIARY", "PhysEquationRole", "PHYS_EQN_", NULL};

/* Face stencil locations indexed by direction: LEFT for x, DOWN for y, BACK for z */
static const DMStagStencilLocation face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* Declare a solution field. Idempotent: re-declaring a field with the same layout is a silent
   success, so that a Seg may declare an auxiliary field that its Phys has already declared. */
PetscErrorCode PhysDeclareField(Phys phys, const char name[], PhysFieldLocation loc, PetscInt ncomp, PhysEquationRole role)
{
  PhysField *field;
  PetscInt   f, c0 = 0;
  PetscBool  same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCheck(phys->nfields < PHYS_MAX_FIELDS, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Cannot register more than %d fields", PHYS_MAX_FIELDS);
  PetscCheck(ncomp > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Field %s must have at least one component", name);
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PetscStrcmp(phys->fields[f].name, name, &same));
    if (same) {
      PetscCheck(phys->fields[f].loc == loc && phys->fields[f].ncomp == ncomp && phys->fields[f].role == role, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Field %s is already registered with a different layout", name);
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (phys->fields[f].loc == loc) c0 += phys->fields[f].ncomp;
  }
  field = &phys->fields[phys->nfields];
  PetscCall(PetscStrallocpy(name, &field->name));
  field->loc             = loc;
  field->ncomp           = ncomp;
  field->c0              = c0;
  field->role            = role;
  field->nullspace_const = PETSC_FALSE;
  ++phys->nfields;
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
  PetscCheck(*field, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Field %s is not registered", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetField_Internal(Phys phys, const char name[], PhysFieldLocation *loc, PetscInt *c0, PetscInt *ncomp)
{
  PhysField *field;

  PetscFunctionBegin;
  PetscCall(PhysFindField_Private(phys, name, &field));
  if (loc) *loc = field->loc;
  if (c0) *c0 = field->c0;
  if (ncomp) *ncomp = field->ncomp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysDeclareConstantNullSpace_Internal(Phys phys, const char name[])
{
  PhysField *field;

  PetscFunctionBegin;
  PetscCall(PhysFindField_Private(phys, name, &field));
  field->nullspace_const = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetFieldRole(Phys phys, const char name[], PhysEquationRole *role)
{
  PhysField *field;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(role, 3);
  PetscCall(PhysFindField_Private(phys, name, &field));
  *role = field->role;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetFieldNullSpaceConstant(Phys phys, const char name[], PetscBool *flg)
{
  PhysField *field;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(flg, 3);
  PetscCall(PhysFindField_Private(phys, name, &field));
  *flg = field->nullspace_const;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetFieldIS_Internal(Phys phys, const char name[], IS *is)
{
  PhysField     *field;
  DMStagStencil *st;
  PetscInt       n = 0, d, c;

  PetscFunctionBegin;
  PetscCall(PhysFindField_Private(phys, name, &field));
  PetscCall(PetscMalloc1(field->ncomp * phys->dim, &st));
  for (d = 0; d < (field->loc == PHYS_FIELD_FACE ? phys->dim : 1); ++d) {
    for (c = 0; c < field->ncomp; ++c) {
      st[n].i   = 0;
      st[n].j   = 0;
      st[n].k   = 0;
      st[n].loc = field->loc == PHYS_FIELD_FACE ? face_loc[d] : DMSTAG_ELEMENT;
      st[n].c   = field->c0 + c;
      ++n;
    }
  }
  PetscCall(DMStagCreateISFromStencils(phys->sol_dm, n, st, is));
  PetscCall(PetscFree(st));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetField(Phys phys, const char name[], PhysFieldLocation *loc, PetscInt *c0, PetscInt *ncomp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetField()");
  PetscCall(PhysGetField_Internal(phys, name, loc, c0, ncomp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetFieldIS(Phys phys, const char name[], IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(is, 3);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetFieldIS()");
  PetscCheck(phys->sol_dm, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Solution DM does not exist yet; call SegSetUp() on a Seg attached to this Phys first");
  PetscCall(PhysGetFieldIS_Internal(phys, name, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}
