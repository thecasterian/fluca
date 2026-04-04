#include <fluca/private/segimpl.h>

static PetscBool SegPackageInitialized = PETSC_FALSE;

PetscErrorCode SegFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&SegList));
  SegPackageInitialized = PETSC_FALSE;
  SegRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegInitializePackage(void)
{
  char         logList[256];
  PetscBool    opt, pkg;
  PetscClassId classids[1];

  PetscFunctionBegin;
  if (SegPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  SegPackageInitialized = PETSC_TRUE;

  /* Initialize subpackages */
  PetscCall(SegSRKInitializePackage());

  /* Register class */
  PetscCall(PetscClassIdRegister("Segregated Time Integrator", &SEG_CLASSID));
  /* Register constructors */
  PetscCall(SegRegisterAll());
  /* Register events */
  PetscCall(PetscLogEventRegister("SegSetUp", SEG_CLASSID, &SEG_SetUp));
  PetscCall(PetscLogEventRegister("SegStep", SEG_CLASSID, &SEG_Step));

  /* Process info */
  classids[0] = SEG_CLASSID;
  PetscCall(PetscInfoProcessClass("seg", 1, classids));
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("seg", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(SEG_CLASSID));
  }

  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(SegFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
