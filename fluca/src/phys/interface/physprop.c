#include <fluca/private/physimpl.h>

const char *PhysPropertySources[] = {"CONSTANT", "FUNCTION", "FIELD", "PhysPropertySource", "PHYS_PROPERTY_", NULL};

static PetscErrorCode PhysFindProperty_Private(Phys phys, const char name[], PhysProperty **prop)
{
  PetscInt  p;
  PetscBool same = PETSC_FALSE;

  PetscFunctionBegin;
  *prop = NULL;
  for (p = 0; p < phys->nprops && !same; ++p) {
    PetscCall(PetscStrcmp(phys->props[p].name, name, &same));
    if (same) *prop = &phys->props[p];
  }
  PetscCheck(*prop, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Property %s is not registered", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysRegisterProperty_Internal(Phys phys, const char name[], PhysFieldLocation loc, PhysPropertySource source)
{
  PhysProperty *prop;
  PetscInt      p;
  PetscBool     same;

  PetscFunctionBegin;
  PetscCheck(phys->nprops < PHYS_MAX_PROPERTIES, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Cannot register more than %d properties", PHYS_MAX_PROPERTIES);
  PetscCheck(source == PHYS_PROPERTY_CONSTANT, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Property source %s is reserved and not yet supported; only CONSTANT is accepted", PhysPropertySources[source]);
  for (p = 0; p < phys->nprops; ++p) {
    PetscCall(PetscStrcmp(phys->props[p].name, name, &same));
    PetscCheck(!same, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Property %s is already registered", name);
  }
  prop = &phys->props[phys->nprops];
  PetscCall(PetscStrallocpy(name, &prop->name));
  prop->loc      = loc;
  prop->source   = source;
  prop->constant = 0.;
  ++phys->nprops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetPropertyConstant_Internal(Phys phys, const char name[], PetscScalar value)
{
  PhysProperty *prop;

  PetscFunctionBegin;
  PetscCall(PhysFindProperty_Private(phys, name, &prop));
  PetscCheck(prop->source == PHYS_PROPERTY_CONSTANT, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Property %s is not a constant", name);
  prop->constant = value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetPropertySource(Phys phys, const char name[], PhysPropertySource *source)
{
  PhysProperty *prop;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(source, 3);
  PetscCall(PhysFindProperty_Private(phys, name, &prop));
  *source = prop->source;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetPropertyLocation(Phys phys, const char name[], PhysFieldLocation *loc)
{
  PhysProperty *prop;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(loc, 3);
  PetscCall(PhysFindProperty_Private(phys, name, &prop));
  *loc = prop->loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetPropertyConstant(Phys phys, const char name[], PetscScalar *value)
{
  PhysProperty *prop;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(value, 3);
  PetscCall(PhysFindProperty_Private(phys, name, &prop));
  PetscCheck(prop->source == PHYS_PROPERTY_CONSTANT, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Property %s is not a constant", name);
  *value = prop->constant;
  PetscFunctionReturn(PETSC_SUCCESS);
}
