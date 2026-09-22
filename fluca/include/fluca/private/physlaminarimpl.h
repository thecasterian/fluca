#pragma once

#include <fluca/private/physimpl.h>

typedef struct {
  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysLaminarBC bcs[FLUCA_MAX_FACES];
} Phys_Laminar;
