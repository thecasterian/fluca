#pragma once

#include <flucasys.h>

typedef enum {
  NS_BC_NONE,
  NS_BC_VELOCITY,
  NS_BC_PRESSURE_OUTLET,
  NS_BC_PERIODIC,
  NS_BC_SYMMETRY,
} NSBoundaryConditionType;
FLUCA_EXTERN const char *NSBoundaryConditionTypes[];

typedef PetscErrorCode (*NSBoundaryConditionFunction)(PetscInt, PetscReal, const PetscReal[], PetscScalar[], void *);

typedef struct {
  NSBoundaryConditionType     type;
  NSBoundaryConditionFunction velocity;
  void                       *ctx_velocity;
  NSBoundaryConditionFunction pressure;
  void                       *ctx_pressure;
} NSBoundaryCondition;
