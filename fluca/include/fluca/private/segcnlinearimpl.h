#pragma once

#include <fluca/private/segimpl.h>
#include <fluca/private/segopsimpl.h>
#include <flucaphys.h>

/* Row blocks of the coupled system, in the order SegSetUp_CNLinear() fills seg->fields */
typedef enum {
  SEG_CNLINEAR_FIELD_VELOCITY,
  SEG_CNLINEAR_FIELD_FACE_VELOCITY,
  SEG_CNLINEAR_FIELD_PRESSURE,
  SEG_CNLINEAR_NUM_FIELDS,
} SegCNLinearField;

typedef struct {
  Mat          M; /* coupled system (13) on the solution DM */
  Mat          P; /* MATNEST carrying the field index sets that PCABF reads */
  MatNullSpace nullspace;
  Vec          f, x;

  /* Momentum rows of the coupled system (13): A = I + (dt/2) J - (dt/2) nu lap and G = (dt/rho) grad */
  DM      dm_face;                                     /* one DOF per face */
  Vec     ubar[FLUCA_MAX_DIM];                         /* ubar_d^n: u_d^n linearly interpolated to every face */
  FlucaFD fd_interp_vel[FLUCA_MAX_DIM][FLUCA_MAX_DIM]; /* [d][e]: u_d -> faces normal to e, onto dm_face */
  FlucaFD fd_visc[FLUCA_MAX_DIM];                      /* (dt/(2 rho)) fd_laplacian[d] = -(dt/2) nu lap(u_d) */
  FlucaFD fd_conv[FLUCA_MAX_DIM];                      /* (dt/2) sum_e d/dx_e(ubar_d U_e^n + ubar_d^n ubar_e) */
  FlucaFD fd_conv_U[FLUCA_MAX_DIM][FLUCA_MAX_DIM];     /* [d][e]: ubar_d on faces normal to e, times U_e^n */
  FlucaFD fd_conv_ubar[FLUCA_MAX_DIM][FLUCA_MAX_DIM];  /* [d][e]: ubar_e on faces normal to e, times ubar_d^n */
  FlucaFD fd_grad[FLUCA_MAX_DIM];                      /* (dt/rho) fd_grad_p[d]: the operator G */
} Seg_CNLinear;

/* Defined in impls/cnlinear/cnlinearsystem.c. Public visibility: the tests link against the shared
   library and assemble the coupled system directly. */
FLUCA_EXTERN PetscErrorCode SegCNLinearComputeMomentumSystem_Internal(Seg, PetscReal, PetscReal, Vec, Mat, Vec);
FLUCA_EXTERN PetscErrorCode SegCNLinearComputeCouplingSystem_Internal(Seg, PetscReal, PetscReal, Mat, Vec);
