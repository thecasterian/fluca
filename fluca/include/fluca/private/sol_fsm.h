#pragma once

#include <fluca/private/solimpl.h>
#include <petscvec.h>

typedef struct {
  Vec v_star[3];
  Vec v_tilde[3];
  Vec N[3];
  Vec N_prev[3];
  Vec fv;
  Vec fv_star;
  Vec p_half; /* p^{n+1/2} */
  Vec p_prime;
  Vec p_half_prev; /* p^{n-1/2} */
} Sol_FSM;
