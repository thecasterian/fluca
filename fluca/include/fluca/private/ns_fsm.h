#if !defined(FLUCA_PRIVATE_NS_FSM_H)
#define FLUCA_PRIVATE_NS_FSM_H

#include <fluca/private/nsimpl.h>
#include <petscksp.h>

typedef struct {
    KSP kspv[3]; /* for solving intermediate velocities */
    KSP kspp;    /* for solving pressure correction */
} NS_FSM;

#endif
