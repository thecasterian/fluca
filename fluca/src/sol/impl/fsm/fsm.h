#if !defined(FLUCA_SOL_SINGLEZONE_H)
#define FLUCA_SOL_SINGLEZONE_H

#include <impl/solimpl.h>
#include <petscvec.h>

typedef struct {
    Vec v_star[3];
    Vec v_tilde[3];
    Vec N[3];
    Vec N_prev[3];
    Vec fv;
    Vec fv_star;
    Vec p_half;
    Vec p_prime;
    Vec p_half_prev;
} Sol_FSM;

#endif
