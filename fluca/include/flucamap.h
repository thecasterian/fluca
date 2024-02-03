#pragma once

#include <flucasys.h>

typedef struct _p_FlucaMap *FlucaMap;

FLUCA_EXTERN PetscClassId FLUCA_MAP_CLASSID;

FLUCA_EXTERN PetscErrorCode FlucaMapCreate(MPI_Comm, FlucaMap *, PetscErrorCode (*)(PetscObject, PetscInt *), PetscErrorCode (*)(PetscObject, PetscObject, PetscBool *));
FLUCA_EXTERN PetscErrorCode FlucaMapGetSize(FlucaMap, PetscInt *);
FLUCA_EXTERN PetscErrorCode FlucaMapInsert(FlucaMap, PetscObject, PetscObject);
FLUCA_EXTERN PetscErrorCode FlucaMapRemove(FlucaMap, PetscObject);
FLUCA_EXTERN PetscErrorCode FlucaMapGetValue(FlucaMap, PetscObject, PetscObject *);
FLUCA_EXTERN PetscErrorCode FlucaMapDestroy(FlucaMap *);
