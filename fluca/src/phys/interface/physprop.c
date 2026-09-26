#include <fluca/private/physimpl.h>

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

/* Register a constant material property with its default value */
PetscErrorCode PhysRegisterProperty_Internal(Phys phys, const char name[], PetscScalar value)
{
  PhysProperty *prop;
  PetscInt      p;
  PetscBool     same;

  PetscFunctionBegin;
  PetscCheck(phys->nprops < PHYS_MAX_PROPERTIES, PetscObjectComm((PetscObject)phys), PETSC_ERR_SUP, "Cannot register more than %d properties", PHYS_MAX_PROPERTIES);
  for (p = 0; p < phys->nprops; ++p) {
    PetscCall(PetscStrcmp(phys->props[p].name, name, &same));
    PetscCheck(!same, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONG, "Property %s is already registered", name);
  }
  prop = &phys->props[phys->nprops];
  PetscCall(PetscStrallocpy(name, &prop->name));
  prop->value = value;
  ++phys->nprops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysSetProperty_Private(Phys phys, const char name[], PetscScalar value)
{
  PhysProperty *prop;

  PetscFunctionBegin;
  PetscCall(PhysFindProperty_Private(phys, name, &prop));
  prop->value = value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetProperty(Phys phys, const char name[], PetscScalar *value)
{
  PhysProperty *prop;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(value, 3);
  PetscCall(PhysFindProperty_Private(phys, name, &prop));
  *value = prop->value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetDensity(Phys phys, PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(phys, rho, 2);
  PetscCheck(rho > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Density must be positive, got %g", (double)rho);
  PetscCall(PhysSetProperty_Private(phys, PHYS_PROPERTY_DENSITY, rho));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetDensity(Phys phys, PetscReal *rho)
{
  PetscScalar value;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(rho, 2);
  PetscCall(PhysGetProperty(phys, PHYS_PROPERTY_DENSITY, &value));
  *rho = PetscRealPart(value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetViscosity(Phys phys, PetscReal mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(phys, mu, 2);
  PetscCheck(mu >= 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Viscosity must be non-negative, got %g", (double)mu);
  PetscCall(PhysSetProperty_Private(phys, PHYS_PROPERTY_VISCOSITY, mu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetViscosity(Phys phys, PetscReal *mu)
{
  PetscScalar value;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(mu, 2);
  PetscCall(PhysGetProperty(phys, PHYS_PROPERTY_VISCOSITY, &value));
  *mu = PetscRealPart(value);
  PetscFunctionReturn(PETSC_SUCCESS);
}
