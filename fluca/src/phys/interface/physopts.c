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
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysGetSolutionDM()");
  *dm = phys->sol_dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetFromOptions(Phys phys)
{
  const char *default_type;
  char        type[256];
  PetscReal   rho, mu;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (!((PetscObject)phys)->type_name) default_type = PHYSLAMINAR;
  else default_type = ((PetscObject)phys)->type_name;
  PetscCall(PhysRegisterAll());

  PetscObjectOptionsBegin((PetscObject)phys);
  PetscCall(PetscOptionsFList("-phys_type", "Physical model type", "PhysSetType", PhysList, default_type, type, sizeof(type), &flg));
  if (flg) PetscCall(PhysSetType(phys, type));
  else if (!((PetscObject)phys)->type_name) PetscCall(PhysSetType(phys, default_type));
  PetscCall(PhysGetDensity(phys, &rho));
  PetscCall(PetscOptionsReal("-phys_density", "Density", "PhysSetDensity", rho, &rho, &flg));
  if (flg) PetscCall(PhysSetDensity(phys, rho));
  PetscCall(PhysGetViscosity(phys, &mu));
  PetscCall(PetscOptionsReal("-phys_viscosity", "Dynamic viscosity", "PhysSetViscosity", mu, &mu, &flg));
  if (flg) PetscCall(PhysSetViscosity(phys, mu));
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

PetscErrorCode PhysGetBodyForce(Phys phys, PhysBodyForceFn **fn, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (fn) *fn = phys->bodyforce;
  if (ctx) *ctx = phys->bodyforce_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysSetBoundaryCondition(Phys phys, PetscInt face, PhysBC bc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCheck(!phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Cannot change a boundary condition after PhysSetUp()");
  PetscCheck(face >= 0 && face < PHYS_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_MAX_FACES);
  phys->bcs[face] = bc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetBoundaryCondition(Phys phys, PetscInt face, PhysBC *bc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(bc, 3);
  PetscCheck(face >= 0 && face < PHYS_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_MAX_FACES);
  *bc = phys->bcs[face];
  PetscFunctionReturn(PETSC_SUCCESS);
}
