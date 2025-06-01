#pragma once

#include <flucasys.h>

typedef enum {
  NS_BOUNDARY_NONE,
} NSBoundaryConditionType;
FLUCA_EXTERN const char *NSBoundaryConditionTypes[];

typedef struct {
  NSBoundaryConditionType type;
} NSBoundaryCondition;
