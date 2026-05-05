#include <fluca/private/flucaibimpl.h>

static PetscBool FlucaIBPackageInitialized = PETSC_FALSE;

PetscErrorCode FlucaIBInitializePackage(void)
{
  char         logList[256];
  PetscBool    opt, pkg;
  PetscClassId classids[1];

  PetscFunctionBegin;
  if (FlucaIBPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  FlucaIBPackageInitialized = PETSC_TRUE;

  /* Register class */
  PetscCall(PetscClassIdRegister("Immersed Boundary", &FLUCAIB_CLASSID));
  /* Register constructors */
  PetscCall(FlucaIBRegisterAll());
  /* Register events */
  PetscCall(PetscLogEventRegister("FlucaIBSetUp", FLUCAIB_CLASSID, &FLUCAIB_SetUp));
  PetscCall(PetscLogEventRegister("FlucaIBApplyBC", FLUCAIB_CLASSID, &FLUCAIB_ApplyBC));

  /* Process info */
  classids[0] = FLUCAIB_CLASSID;
  PetscCall(PetscInfoProcessClass("ib", 1, classids));
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("ib", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(FLUCAIB_CLASSID));
  }

  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(FlucaIBFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaIBFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&FlucaIBList));
  FlucaIBPackageInitialized = PETSC_FALSE;
  FlucaIBRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
