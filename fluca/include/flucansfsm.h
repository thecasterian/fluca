#include <flucans.h>

PETSC_EXTERN PetscErrorCode NSFSMGetVelocity(NS, Vec *);
PETSC_EXTERN PetscErrorCode NSFSMGetIntermediateVelocity(NS, Vec *);
PETSC_EXTERN PetscErrorCode NSFSMGetConvection(NS, Vec *);
PETSC_EXTERN PetscErrorCode NSFSMGetPreviousConvection(NS, Vec *);
PETSC_EXTERN PetscErrorCode NSFSMGetPressure(NS, Vec *);
PETSC_EXTERN PetscErrorCode NSFSMGetHalfStepPressure(NS, Vec *);
