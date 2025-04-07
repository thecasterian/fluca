#pragma once

#include <flucameshtypes.h>
#include <flucasys.h>
#include <petscvec.h>

typedef struct _p_Sol *Sol;

typedef const char *SolType;
#define SOLFSM "fsm"

FLUCA_EXTERN PetscClassId SOL_CLASSID;

FLUCA_EXTERN PetscErrorCode SolInitializePackage(void);
FLUCA_EXTERN PetscErrorCode SolFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode SolCreate(MPI_Comm, Sol *);
FLUCA_EXTERN PetscErrorCode SolSetType(Sol, SolType);
FLUCA_EXTERN PetscErrorCode SolGetType(Sol, SolType *);
FLUCA_EXTERN PetscErrorCode SolSetMesh(Sol, Mesh);
FLUCA_EXTERN PetscErrorCode SolGetMesh(Sol, Mesh *);
FLUCA_EXTERN PetscErrorCode SolGetVelocity(Sol, Vec *, Vec *, Vec *);
FLUCA_EXTERN PetscErrorCode SolGetPressure(Sol, Vec *);
FLUCA_EXTERN PetscErrorCode SolView(Sol, PetscViewer);
FLUCA_EXTERN PetscErrorCode SolViewFromOptions(Sol, PetscObject, const char[]);
FLUCA_EXTERN PetscErrorCode SolDestroy(Sol *);

FLUCA_EXTERN PetscErrorCode SolLoadFromFile(Sol, const char[]);
FLUCA_EXTERN PetscErrorCode SolLoadCGNS(Sol, PetscInt);
FLUCA_EXTERN PetscErrorCode SolLoadCGNSFromFile(Sol, const char[]);

FLUCA_EXTERN PetscFunctionList SolList;
FLUCA_EXTERN PetscErrorCode    SolRegister(const char[], PetscErrorCode (*)(Sol));
