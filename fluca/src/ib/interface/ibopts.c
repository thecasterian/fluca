#include <fluca/private/flucaibimpl.h>

PetscErrorCode FlucaIBSetFromOptions(FlucaIB ib)
{
  const char *default_type;
  char        type[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  if (!((PetscObject)ib)->type_name) default_type = FLUCAIB_NONE;
  else default_type = ((PetscObject)ib)->type_name;
  PetscCall(FlucaIBRegisterAll());

  PetscObjectOptionsBegin((PetscObject)ib);
  PetscCall(PetscOptionsFList("-ib_type", "FlucaIB subtype", "FlucaIBSetType", FlucaIBList, default_type, type, sizeof(type), &flg));
  if (flg) PetscCall(FlucaIBSetType(ib, type));
  else if (!((PetscObject)ib)->type_name) PetscCall(FlucaIBSetType(ib, default_type));
  PetscTryTypeMethod(ib, setfromoptions, PetscOptionsObject);
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)ib, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBSetOptionsPrefix(FlucaIB ib, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ib, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBAppendOptionsPrefix(FlucaIB ib, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)ib, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBGetOptionsPrefix(FlucaIB ib, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ib, FLUCAIB_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ib, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}
