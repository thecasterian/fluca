#pragma once

#include <fluca/private/physimpl.h>

#define PHYS_LAMINAR_MAX_DIM   3
#define PHYS_LAMINAR_MAX_FACES (2 * PHYS_LAMINAR_MAX_DIM)

typedef struct {
  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysLaminarBC bcs[PHYS_LAMINAR_MAX_FACES];
} Phys_Laminar;
