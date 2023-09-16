#if !defined(FLUCANSIMPL_H)
#define FLUCANSIMPL_H

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>
#include <flucans.h>
#include <flucasol.h>

FLUCA_EXTERN PetscBool NSRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode NSRegisterAll(void);
FLUCA_EXTERN PetscLogEvent NS_SetUp;

typedef struct _NSOps *NSOps;

struct _NSOps {
    PetscErrorCode (*setup)(NS);
    PetscErrorCode (*destroy)(NS);
    PetscErrorCode (*view)(NS, PetscViewer);
};

typedef enum {
    NS_STATE_INITIAL,
    NS_STATE_SETUP,
} NSStateType;

struct _p_NS {
    PETSCHEADER(struct _NSOps);

    /* Parameters ----------------------------------------------------------- */
    PetscScalar rho;
    PetscScalar mu;
    PetscScalar dt;

    /* Data ----------------------------------------------------------------- */
    PetscScalar t; /* current time */
    Mesh mesh;
    Sol sol;
    void *data; /* implementation-specific data */

    /* State ---------------------------------------------------------------- */
    NSStateType state;
};

#endif
