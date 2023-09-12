#include <fluca/private/solimpl.h>

static PetscBool SolPackageInitialized = PETSC_FALSE;

PetscErrorCode SolInitializePackage(void) {
    char log_list[256];
    PetscBool opt, pkg;
    PetscClassId classids[1];

    PetscFunctionBegin;

    if (SolPackageInitialized)
        PetscFunctionReturn(PETSC_SUCCESS);
    SolPackageInitialized = PETSC_TRUE;

    /* Register class */
    PetscCall(PetscClassIdRegister("Sol", &SOL_CLASSID));
    /* Register constructors */
    PetscCall(SolRegisterAll());

    /* Process Info */
    classids[0] = SOL_CLASSID;
    PetscCall(PetscInfoProcessClass("sol", 1, classids));
    /* Process summary exclusions */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", log_list, sizeof(log_list), &opt));
    if (opt) {
        PetscCall(PetscStrInList("sol", log_list, ',', &pkg));
        if (pkg)
            PetscCall(PetscLogEventExcludeClass(SOL_CLASSID));
    }
    /* Register package finalizer */
    PetscCall(PetscRegisterFinalize(SolFinalizePackage));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolFinalizePackage(void) {
    PetscFunctionBegin;
    PetscCall(PetscFunctionListDestroy(&SolList));
    SolPackageInitialized = PETSC_FALSE;
    SolRegisterAllCalled = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
}
