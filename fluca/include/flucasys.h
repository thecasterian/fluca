#pragma once

#include <petscsys.h>

#define FLUCA_VISIBILITY_PUBLIC   __attribute__((visibility("default")))
#define FLUCA_VISIBILITY_INTERNAL __attribute__((visibility("hidden")))

#if defined(__cplusplus)
  #define FLUCA_EXTERN extern "C" FLUCA_VISIBILITY_PUBLIC
  #define FLUCA_INTERN extern "C" FLUCA_VISIBILITY_INTERNAL
#else
  #define FLUCA_EXTERN extern FLUCA_VISIBILITY_PUBLIC
  #define FLUCA_INTERN extern FLUCA_VISIBILITY_INTERNAL
#endif

/* Maximum spatial dimension Fluca supports, and the number of boundary faces of a Cartesian cell.
   Every module sizes its fixed-capacity per-direction arrays with these, so that the constant is
   stated exactly once. */
#define FLUCA_MAX_DIM   3
#define FLUCA_MAX_FACES (2 * FLUCA_MAX_DIM)

FLUCA_EXTERN PetscBool FlucaInitializeCalled;
FLUCA_EXTERN PetscBool FlucaFinalizeCalled;

FLUCA_EXTERN PetscErrorCode FlucaInitialize(int *, char ***, const char[], const char[]);
FLUCA_EXTERN PetscErrorCode FlucaInitializeNoArguments(void);
FLUCA_EXTERN PetscErrorCode FlucaFinalize(void);
FLUCA_EXTERN PetscErrorCode FlucaInitialized(PetscBool *);
FLUCA_EXTERN PetscErrorCode FlucaFinalized(PetscBool *);

FLUCA_EXTERN PetscErrorCode FlucaSysInitializePackage(void);
FLUCA_EXTERN PetscErrorCode FlucaSysFinalizePackage(void);
