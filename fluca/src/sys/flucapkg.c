#include <flucamap.h>

static PetscBool FlucaSysPackageInitialized = PETSC_FALSE;

PetscErrorCode FlucaSysInitializePackage(void) {
    PetscClassId classids[1];

    PetscFunctionBegin;

    if (FlucaSysPackageInitialized)
        PetscFunctionReturn(PETSC_SUCCESS);
    FlucaSysPackageInitialized = PETSC_TRUE;

    /* Register classes */
    PetscCall(PetscClassIdRegister("Map", &FLUCA_MAP_CLASSID));

    /* Process Info */
    classids[0] = FLUCA_MAP_CLASSID;
    PetscCall(PetscInfoProcessClass("map", 1, classids));
    /* Register package finalizer */
    PetscCall(PetscRegisterFinalize(FlucaSysFinalizePackage));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaSysFinalizePackage(void) {
    PetscFunctionBegin;
    FlucaSysPackageInitialized = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
}
