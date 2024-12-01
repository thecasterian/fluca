#include <flucasys.h>

PetscBool FlucaBeganPetsc       = PETSC_FALSE;
PetscBool FlucaInitializeCalled = PETSC_FALSE;
PetscBool FlucaFinalizeCalled   = PETSC_FALSE;

PetscErrorCode FlucaInitialize(int *argc, char ***args, const char file[], const char help[])
{
  PetscBool flg;

  PetscFunctionBegin;
  if (FlucaInitializeCalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscInitialized(&flg));
  if (!flg) {
    PetscCall(PetscInitialize(argc, args, file, help));
    FlucaBeganPetsc = PETSC_TRUE;
  }

  PetscCall(FlucaSysInitializePackage());

  FlucaInitializeCalled = PETSC_TRUE;
  FlucaFinalizeCalled   = PETSC_FALSE;
  PetscCall(PetscInfo(NULL, "Fluca successfully started\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaInitializeNoArguments(void)
{
  int    argc = 0;
  char **args = NULL;

  PetscFunctionBegin;
  PetscCall(FlucaInitialize(&argc, &args, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFinalize(void)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscCheck(FlucaInitializeCalled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "FlucaInitialize() must be called before FlucaFinalize()");

  PetscCall(PetscInfo(NULL, "FlucaFinalize() called\n"));

  if (FlucaBeganPetsc) {
    PetscCall(PetscFinalized(&flg));
    PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_LIB, "PetscFinalize() has already been called, even though PetscInitialize() was called by FlucaInitialize()");
    PetscCall(PetscFinalize());
    FlucaBeganPetsc = PETSC_FALSE;
  }

  FlucaInitializeCalled = PETSC_FALSE;
  FlucaFinalizeCalled   = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaInitialized(PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = FlucaInitializeCalled;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFinalized(PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = FlucaFinalizeCalled;
  PetscFunctionReturn(PETSC_SUCCESS);
}
