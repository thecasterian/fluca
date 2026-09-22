#pragma once

#include <fluca/private/physimpl.h>
#include <fluca/private/segimpl.h>

typedef struct {
  Mat          M;     /* coupled system (13) on the solution DM */
  Mat          P;     /* MATNEST carrying the field index sets that PCABF reads */
  IS           is[3]; /* velocity, face velocity, pressure */
  MatNullSpace nullspace;
  Vec          f, x;
} Seg_CNLinear;
