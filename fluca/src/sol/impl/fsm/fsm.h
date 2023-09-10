#if !defined(FLUCA_SOL_SINGLEZONE_H)
#define FLUCA_SOL_SINGLEZONE_H

#include <impl/solimpl.h>
#include <petscvec.h>

typedef struct {
    Vec p_half;
    Vec UVW;
    Vec u_star, v_star, w_star;
    Vec UVW_star;
    Vec p_prime;
    Vec Nu, Nv, Nw;
    Vec p_prev;
    Vec Nu_prev, Nv_prev, Nw_prev;

    Vec u_tilde, v_tilde, w_tilde;
} Sol_FSM;

#endif
