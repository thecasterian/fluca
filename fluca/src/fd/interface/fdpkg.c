#include <fluca/private/flucafdimpl.h>

static PetscBool FlucaFDPackageInitialized = PETSC_FALSE;

PetscErrorCode FlucaFDFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&FlucaFDList));
  FlucaFDPackageInitialized = PETSC_FALSE;
  FlucaFDRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (FlucaFDPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  FlucaFDPackageInitialized = PETSC_TRUE;

  /* Register class */
  PetscCall(PetscClassIdRegister("Finite Difference", &FLUCAFD_CLASSID));

  /* Register constructors */
  PetscCall(FlucaFDRegisterAll());

  /* Process info exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-info_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("flucafd", logList, ',', &pkg));
    if (pkg) PetscCall(PetscInfoDeactivateClass(FLUCAFD_CLASSID));
  }

  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("flucafd", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventDeactivateClass(FLUCAFD_CLASSID));
  }

  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(FlucaFDFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
