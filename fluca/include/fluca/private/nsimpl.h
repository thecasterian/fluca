#if !defined(FLUCANSIMPL_H)
#define FLUCANSIMPL_H

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>
#include <flucans.h>
#include <flucasol.h>

#define MAXNSMONITORS 10

FLUCA_EXTERN PetscBool NSRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode NSRegisterAll(void);
FLUCA_EXTERN PetscLogEvent NS_SetUp;
FLUCA_EXTERN PetscLogEvent NS_Solve;

typedef struct _NSOps *NSOps;

struct _NSOps {
    PetscErrorCode (*setup)(NS);
    PetscErrorCode (*solve_init)(NS);
    PetscErrorCode (*solve_iter)(NS);
    PetscErrorCode (*destroy)(NS);
    PetscErrorCode (*view)(NS, PetscViewer);
};

typedef enum {
    NS_STATE_INITIAL,
    NS_STATE_SETUP,
    NS_STATE_SOLUTION_INITIALIZED,
} NSStateType;

struct _p_NS {
    PETSCHEADER(struct _NSOps);

    /* Parameters ----------------------------------------------------------- */
    PetscReal rho; /* density */
    PetscReal mu;  /* dynamic viscosity */
    PetscReal dt;  /* time step size */

    /* Data ----------------------------------------------------------------- */
    PetscInt step; /* current time step */
    PetscReal t;   /* current time */
    Mesh mesh;
    Sol sol;
    void *data; /* implementation-specific data */

    /* State ---------------------------------------------------------------- */
    NSStateType state;

    /* Monitor -------------------------------------------------------------- */
    PetscInt num_mons;
    PetscErrorCode (*mons[MAXNSMONITORS])(NS, PetscInt, PetscReal, Sol, void *);
    void *mon_ctxs[MAXNSMONITORS];
    PetscErrorCode (*mon_ctx_destroys[MAXNSMONITORS])(void **);
};

#endif
