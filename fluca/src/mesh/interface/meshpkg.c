#include <impl/meshimpl.h>

static PetscBool MeshPackageInitialized = PETSC_FALSE;

PetscErrorCode MeshInitializePackage(void) {
    char log_list[256];
    PetscBool opt, pkg;
    PetscClassId classids[1];

    PetscFunctionBegin;

    if (MeshPackageInitialized)
        PetscFunctionReturn(PETSC_SUCCESS);
    MeshPackageInitialized = PETSC_TRUE;

    /* Register class */
    PetscCall(PetscClassIdRegister("Mesh", &MESH_CLASSID));
    /* Register constructors */
    PetscCall(MeshRegisterAll());
    /* Register events */
    PetscCall(PetscLogEventRegister("MeshSetUp", MESH_CLASSID, &MESH_SetUp));

    /* Process Info */
    classids[0] = MESH_CLASSID;
    PetscCall(PetscInfoProcessClass("mesh", 1, classids));
    /* Process summary exclusions */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", log_list, sizeof(log_list), &opt));
    if (opt) {
        PetscCall(PetscStrInList("mesh", log_list, ',', &pkg));
        if (pkg)
            PetscCall(PetscLogEventExcludeClass(MESH_CLASSID));
    }
    /* Register package finalizer */
    PetscCall(PetscRegisterFinalize(MeshFinalizePackage));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshFinalizePackage(void) {
    PetscFunctionBegin;
    PetscCall(PetscFunctionListDestroy(&MeshList));
    MeshPackageInitialized = PETSC_FALSE;
    MeshRegisterAllCalled = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
}
