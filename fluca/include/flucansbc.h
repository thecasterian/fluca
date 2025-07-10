#pragma once

#include <flucasys.h>

typedef enum {
  NS_BC_NONE,
  NS_BC_VELOCITY,
  NS_BC_PERIODIC,
} NSBoundaryConditionType;
FLUCA_EXTERN const char *NSBoundaryConditionTypes[];

typedef PetscErrorCode (*NSBoundaryConditionFunction)(PetscInt, PetscReal, const PetscReal[], PetscScalar[], void *);

typedef struct {
  NSBoundaryConditionType     type;
  NSBoundaryConditionFunction velocity;
  void                       *ctx_velocity;
} NSBoundaryCondition;
