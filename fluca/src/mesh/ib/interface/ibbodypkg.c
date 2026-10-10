#include <fluca/private/ibbodyimpl.h>

static PetscBool IBBodyPackageInitialized = PETSC_FALSE;

PetscErrorCode IBBodyInitializePackage(void)
{
  char         log_list[256];
  PetscBool    opt, pkg;
  PetscClassId classids[1];

  PetscFunctionBegin;
  if (IBBodyPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  IBBodyPackageInitialized = PETSC_TRUE;

  /* Register class */
  PetscCall(PetscClassIdRegister("IBBody", &IBBODY_CLASSID));
  /* Register constructors */
  PetscCall(IBBodyRegisterAll());

  /* Process Info */
  classids[0] = IBBODY_CLASSID;
  PetscCall(PetscInfoProcessClass("ibbody", 1, classids));
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", log_list, sizeof(log_list), &opt));
  if (opt) {
    PetscCall(PetscStrInList("ibbody", log_list, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(IBBODY_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(IBBodyFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&IBBodyList));
  IBBodyPackageInitialized = PETSC_FALSE;
  IBBodyRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
