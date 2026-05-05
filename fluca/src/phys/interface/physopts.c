#include <fluca/private/physimpl.h>

PetscErrorCode PhysSetIB(Phys phys, FlucaIB ib)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 2);
  PetscCheckSameComm(phys, 1, ib, 2);
  PetscCheck(!phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Cannot change IB after PhysSetUp()");
  PetscCall(FlucaIBDestroy(&phys->ib));
  phys->ib = ib;
  PetscCall(PetscObjectReference((PetscObject)ib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetIB(Phys phys, FlucaIB *ib)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(ib, 2);
  *ib = phys->ib;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysGetBaseDM(Phys phys, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  *dm = phys->dm;
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
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  if (!((PetscObject)phys)->type_name) default_type = PHYSINS;
  else default_type = ((PetscObject)phys)->type_name;
  PetscCall(PhysRegisterAll());

  PetscObjectOptionsBegin((PetscObject)phys);
  PetscCall(PetscOptionsFList("-phys_type", "Physical model type", "PhysSetType", PhysList, default_type, type, sizeof(type), &flg));
  if (flg) PetscCall(PhysSetType(phys, type));
  else if (!((PetscObject)phys)->type_name) PetscCall(PhysSetType(phys, default_type));
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
