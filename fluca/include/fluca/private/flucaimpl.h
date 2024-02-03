#pragma once

#include <flucasys.h>
#include <petsc/private/petscimpl.h>

FLUCA_INTERN PetscBool FlucaBeganPetsc;

#define FlucaHeaderCreate(h, classid, class_name, descr, mansec, comm, destroy, view) \
  (FlucaInitializeCalled ? PetscHeaderCreate(h, classid, class_name, descr, mansec, comm, destroy, view) : PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ORDER, PETSC_ERROR_INITIAL, "Must call FlucaInitialize to use Fluca classes"))
