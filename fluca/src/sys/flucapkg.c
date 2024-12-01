#include <flucasys.h>
#include <flucaviewer.h>

PETSC_EXTERN PetscErrorCode PetscViewerCreate_FlucaCGNS(PetscViewer);

static PetscBool FlucaSysPackageInitialized = PETSC_FALSE;

PetscErrorCode FlucaSysInitializePackage(void)
{
  PetscFunctionBegin;
  if (FlucaSysPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  FlucaSysPackageInitialized = PETSC_TRUE;

  /* Register PETSc extensions */
  PetscCall(PetscViewerRegister(PETSCVIEWERFLUCACGNS, PetscViewerCreate_FlucaCGNS));

  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(FlucaSysFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaSysFinalizePackage(void)
{
  PetscFunctionBegin;
  FlucaSysPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
