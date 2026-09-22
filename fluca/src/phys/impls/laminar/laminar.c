#include <fluca/private/physlaminarimpl.h>

static PetscErrorCode PhysRegisterFields_Laminar(Phys phys)
{
  PetscFunctionBegin;
  PetscCall(PhysDeclareField_Internal(phys, PHYS_FIELD_VELOCITY, PHYS_FIELD_ELEMENT, phys->dim, PHYS_EQN_MOMENTUM));
  PetscCall(PhysDeclareField_Internal(phys, PHYS_FIELD_PRESSURE, PHYS_FIELD_ELEMENT, 1, PHYS_EQN_PRESSURE));
  PetscCall(PhysDeclareConstantNullSpace_Internal(phys, PHYS_FIELD_PRESSURE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysSetFromOptions_Laminar(Phys phys, PetscOptionItems PetscOptionsObject)
{
  PetscScalar rho, mu;
  PetscReal   rho_new, mu_new;

  PetscFunctionBegin;
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho));
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
  rho_new = PetscRealPart(rho);
  mu_new  = PetscRealPart(mu);
  PetscOptionsHeadBegin(PetscOptionsObject, "Laminar Options");
  PetscCall(PetscOptionsReal("-phys_laminar_density", "Density", "PhysLaminarSetDensity", PetscRealPart(rho), &rho_new, NULL));
  PetscCall(PetscOptionsReal("-phys_laminar_viscosity", "Dynamic viscosity", "PhysLaminarSetViscosity", PetscRealPart(mu), &mu_new, NULL));
  PetscOptionsHeadEnd();
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_DENSITY, rho_new));
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_VISCOSITY, mu_new));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysDestroy_Laminar(Phys phys)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(phys->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysView_Laminar(Phys phys, PetscViewer viewer)
{
  PetscBool   isascii;
  PetscScalar rho, mu;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho));
    PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Density: %g\n", (double)PetscRealPart(rho)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Viscosity: %g\n", (double)PetscRealPart(mu)));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreate_Laminar(Phys phys)
{
  Phys_Laminar *ins;
  PetscInt      f;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ins));
  PetscCall(PhysRegisterProperty_Internal(phys, PHYS_PROPERTY_DENSITY, PHYS_FIELD_ELEMENT, PHYS_PROPERTY_CONSTANT));
  PetscCall(PhysRegisterProperty_Internal(phys, PHYS_PROPERTY_VISCOSITY, PHYS_FIELD_FACE, PHYS_PROPERTY_CONSTANT));
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_DENSITY, 1.));
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_VISCOSITY, 1.));

  /* Initialize BCs to NONE */
  for (f = 0; f < PHYS_LAMINAR_MAX_FACES; f++) {
    ins->bcs[f].type       = PHYS_LAMINAR_BC_NONE;
    ins->bcs[f].fn         = NULL;
    ins->bcs[f].ctx        = NULL;
    ins->bcs[f].fn_dot     = NULL;
    ins->bcs[f].fn_dot_ctx = NULL;
  }

  phys->data                = ins;
  phys->ops->registerfields = PhysRegisterFields_Laminar;
  phys->ops->setfromoptions = PhysSetFromOptions_Laminar;
  phys->ops->destroy        = PhysDestroy_Laminar;
  phys->ops->view           = PhysView_Laminar;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Public Laminar-specific setters/getters ------------------------------ */

PetscErrorCode PhysLaminarSetDensity(Phys phys, PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSLAMINAR);
  PetscValidLogicalCollectiveReal(phys, rho, 2);
  PetscCheck(rho > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Density must be positive, got %g", (double)rho);
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_DENSITY, rho));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysLaminarGetDensity(Phys phys, PetscReal *rho)
{
  PetscScalar rho_val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSLAMINAR);
  PetscAssertPointer(rho, 2);
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho_val));
  *rho = PetscRealPart(rho_val);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysLaminarSetViscosity(Phys phys, PetscReal mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSLAMINAR);
  PetscValidLogicalCollectiveReal(phys, mu, 2);
  PetscCheck(mu >= 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Viscosity must be non-negative, got %g", (double)mu);
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_VISCOSITY, mu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysLaminarGetViscosity(Phys phys, PetscReal *mu)
{
  PetscScalar mu_val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSLAMINAR);
  PetscAssertPointer(mu, 2);
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu_val));
  *mu = PetscRealPart(mu_val);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysLaminarSetBoundaryCondition(Phys phys, PetscInt face, PhysLaminarBC bc)
{
  Phys_Laminar *ins = (Phys_Laminar *)phys->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSLAMINAR);
  PetscCheck(face >= 0 && face < PHYS_LAMINAR_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_LAMINAR_MAX_FACES);
  ins->bcs[face] = bc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysLaminarGetBoundaryCondition(Phys phys, PetscInt face, PhysLaminarBC *bc)
{
  Phys_Laminar *ins = (Phys_Laminar *)phys->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSLAMINAR);
  PetscAssertPointer(bc, 3);
  PetscCheck(face >= 0 && face < PHYS_LAMINAR_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_LAMINAR_MAX_FACES);
  *bc = ins->bcs[face];
  PetscFunctionReturn(PETSC_SUCCESS);
}
