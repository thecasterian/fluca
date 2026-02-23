#include <fluca/private/physimpl.h>

static PetscBool PhysPackageInitialized = PETSC_FALSE;

PetscErrorCode PhysFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PhysList));
  PhysPackageInitialized = PETSC_FALSE;
  PhysRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysInitializePackage(void)
{
  char         logList[256];
  PetscBool    opt, pkg;
  PetscClassId classids[1];

  PetscFunctionBegin;
  if (PhysPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PhysPackageInitialized = PETSC_TRUE;

  /* Register class */
  PetscCall(PetscClassIdRegister("Physical Model", &PHYS_CLASSID));
  /* Register constructors */
  PetscCall(PhysRegisterAll());
  /* Register log events */
  PetscCall(PetscLogEventRegister("PhysSetUp", PHYS_CLASSID, &Phys_SetUp));

  /* Process info */
  classids[0] = PHYS_CLASSID;
  PetscCall(PetscInfoProcessClass("phys", 1, classids));
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("phys", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventDeactivateClass(PHYS_CLASSID));
  }

  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PhysFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
