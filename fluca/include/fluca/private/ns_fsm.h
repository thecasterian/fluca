#if !defined(FLUCA_PRIVATE_NS_FSM_H)
#define FLUCA_PRIVATE_NS_FSM_H

#include <fluca/private/nsimpl.h>
#include <petscksp.h>

typedef struct {
    KSP ksp; /* for solving intermediate velocities and pressure correction */
} NS_FSM;

#endif
