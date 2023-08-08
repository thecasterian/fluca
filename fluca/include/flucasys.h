#if !defined(FLUCASYS_H)
#define FLUCASYS_H

#include <petscsys.h>

#define FLUCA_VISIBILITY_PUBLIC __attribute__((visibility("default")))
#define FLUCA_VISIBILITY_INTERNAL __attribute__((visibility("hidden")))

#if defined(__cplusplus)
#define FLUCA_EXTERN extern "C" FLUCA_VISIBILITY_PUBLIC
#define FLUCA_INTERN extern "C" FLUCA_VISIBILITY_INTERNAL
#else
#define FLUCA_EXTERN extern FLUCA_VISIBILITY_PUBLIC
#define FLUCA_INTERN extern FLUCA_VISIBILITY_INTERNAL
#endif

FLUCA_EXTERN PetscBool FlucaBeganPetsc;
FLUCA_EXTERN PetscBool FlucaInitializeCalled;
FLUCA_EXTERN PetscBool FlucaFinalizeCalled;

FLUCA_EXTERN PetscErrorCode FlucaInitialize(int *, char ***, const char *, const char *);
FLUCA_EXTERN PetscErrorCode FlucaInitializeNoArguments(void);
FLUCA_EXTERN PetscErrorCode FlucaFinalize(void);
FLUCA_EXTERN PetscErrorCode FlucaInitialized(PetscBool *);
FLUCA_EXTERN PetscErrorCode FlucaFinalized(PetscBool *);

#endif
