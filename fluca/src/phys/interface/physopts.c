#include <fluca/private/physimpl.h>

PetscErrorCode PhysSetBaseDM(Phys phys, DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 2, DMSTAG);
  PetscCheckSameComm(phys, 1, dm, 2);
  PetscCheck(!phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Cannot change base DM after PhysSetUp()");
  PetscCall(DMDestroy(&phys->base_dm));
  phys->base_dm = dm;
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetBaseDM(Phys phys, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  *dm = phys->base_dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetSolutionDM(Phys phys, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  PetscCheck(phys->sol_dm, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Solution DM does not exist yet; call PhysSetUp() first");
  *dm = phys->sol_dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetFromOptions(Phys phys)
{
  const char *default_type;
  char        type[256];
  PetscBool   flg;
  PetscScalar rho, mu;
  PetscReal   rho_new, mu_new;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (!((PetscObject)phys)->type_name) default_type = PHYSLAMINAR;
  else default_type = ((PetscObject)phys)->type_name;
  PetscCall(PhysRegisterAll());

  PetscObjectOptionsBegin((PetscObject)phys);
  PetscCall(PetscOptionsFList("-phys_type", "Physical model type", "PhysSetType", PhysList, default_type, type, sizeof(type), &flg));
  if (flg) PetscCall(PhysSetType(phys, type));
  else if (!((PetscObject)phys)->type_name) PetscCall(PhysSetType(phys, default_type));

  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho));
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
  rho_new = PetscRealPart(rho);
  mu_new  = PetscRealPart(mu);
  PetscCall(PetscOptionsReal("-phys_density", "Density", "PhysSetDensity", rho_new, &rho_new, NULL));
  PetscCall(PetscOptionsReal("-phys_viscosity", "Dynamic viscosity", "PhysSetViscosity", mu_new, &mu_new, NULL));
  /* Through the public setters, so that option values get the same range checks */
  PetscCall(PhysSetDensity(phys, rho_new));
  PetscCall(PhysSetViscosity(phys, mu_new));

  PetscTryTypeMethod(phys, setfromoptions, PetscOptionsObject);
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)phys, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetOptionsPrefix(Phys phys, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)phys, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysAppendOptionsPrefix(Phys phys, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)phys, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetOptionsPrefix(Phys phys, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)phys, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetBodyForce(Phys phys, PhysBodyForceFn *fn, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  phys->bodyforce     = fn;
  phys->bodyforce_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The body force set by PhysSetBodyForce(), or NULL if none was set. Either output may be NULL. */
PetscErrorCode PhysGetBodyForce(Phys phys, PhysBodyForceFn **fn, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (fn) *fn = phys->bodyforce;
  if (ctx) *ctx = phys->bodyforce_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Material properties (base class) -------------------------------------- */

PetscErrorCode PhysSetDensity(Phys phys, PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(phys, rho, 2);
  PetscCheck(rho > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Density must be positive, got %g", (double)rho);
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_DENSITY, rho));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetDensity(Phys phys, PetscReal *rho)
{
  PetscScalar rho_val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(rho, 2);
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho_val));
  *rho = PetscRealPart(rho_val);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetViscosity(Phys phys, PetscReal mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(phys, mu, 2);
  PetscCheck(mu >= 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Viscosity must be non-negative, got %g", (double)mu);
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_VISCOSITY, mu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetViscosity(Phys phys, PetscReal *mu)
{
  PetscScalar mu_val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(mu, 2);
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu_val));
  *mu = PetscRealPart(mu_val);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Boundary conditions (base class) --------------------------------------- */

PetscErrorCode PhysSetBoundaryCondition(Phys phys, PetscInt face, PhysBC bc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCheck(face >= 0 && face < FLUCA_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, FLUCA_MAX_FACES);
  /* A Seg copies the boundary conditions into its operators when it is set up, so a later change would be ignored */
  PetscCheck(!phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Cannot change a boundary condition after PhysSetUp()");
  phys->bcs[face] = bc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetBoundaryCondition(Phys phys, PetscInt face, PhysBC *bc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(bc, 3);
  PetscCheck(face >= 0 && face < FLUCA_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, FLUCA_MAX_FACES);
  *bc = phys->bcs[face];
  PetscFunctionReturn(PETSC_SUCCESS);
}
