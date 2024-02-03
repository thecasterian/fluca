#include <fluca/private/nsimpl.h>

static PetscBool NSPackageInitialized = PETSC_FALSE;

PetscErrorCode NSInitializePackage(void)
{
  char         log_list[256];
  PetscBool    opt, pkg;
  PetscClassId classids[1];

  PetscFunctionBegin;
  if (NSPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  NSPackageInitialized = PETSC_TRUE;

  /* Register class */
  PetscCall(PetscClassIdRegister("NS", &NS_CLASSID));
  /* Register constructors */
  PetscCall(NSRegisterAll());
  /* Register events */
  PetscCall(PetscLogEventRegister("NSSetUp", NS_CLASSID, &NS_SetUp));
  PetscCall(PetscLogEventRegister("NSSolve", NS_CLASSID, &NS_Solve));

  /* Process Info */
  classids[0] = NS_CLASSID;
  PetscCall(PetscInfoProcessClass("ns", 1, classids));
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", log_list, sizeof(log_list), &opt));
  if (opt) {
    PetscCall(PetscStrInList("ns", log_list, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(NS_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(NSFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&NSList));
  NSPackageInitialized = PETSC_FALSE;
  NSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
