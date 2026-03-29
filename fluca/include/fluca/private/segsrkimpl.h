#pragma once

#include <fluca/private/segimpl.h>
#include <petscksp.h>

#define SEG_SRK_MAX_DIM 3

typedef struct {
  /* Tableau (ARKIMEX L2 initially) */
  PetscInt   s;     /* number of stages */
  PetscInt   order; /* temporal order */
  PetscReal *At;    /* implicit tableau (s x s, row-major) */
  PetscReal *A;     /* explicit tableau (s x s, row-major) */
  PetscReal *bt;    /* implicit weights [s] */
  PetscReal *b;     /* explicit weights [s] */
  PetscReal *ct;    /* implicit abscissae [s] */
  PetscReal *c;     /* explicit abscissae [s] */
  PetscBool  stiffly_accurate;
  PetscBool  fsal;
  PetscBool  explicit_first_stage;

  /* Stage vectors (full solution-sized, owned) */
  Vec *Y;          /* stage solutions [s]: (u_j, p_j) */
  Vec *K_u;        /* implicit velocity slopes [s]: K_j^u = F_diff(u_j) */
  Vec *K_hat_u;    /* explicit velocity slopes [s]: K_hat_j^u = C_j^u - G(p_j)/rho */
  Vec  K_hat_prev; /* FSAL: K_hat_u from previous step's last stage */
  Vec  Z;          /* accumulation vector (u_{j,*}) */
  Vec  U_prev;     /* u^{n-1} saved at step start (for Baumgarte) */
  Vec  work1;      /* work vector (full size) */
  Vec  work2;      /* work vector (full size) */
  Vec  work3;      /* work vector (full size) */

  /* FlucaFD operators (borrowed, not owned) */
  FlucaFD fd_laplacian[SEG_SRK_MAX_DIM]; /* viscous Laplacian per velocity component */
  FlucaFD fd_grad_p[SEG_SRK_MAX_DIM];    /* pressure gradient per direction */
  FlucaFD fd_div;                        /* divergence (includes rho) */
  FlucaFD fd_pstab;                      /* pressure stabilization σ₀·S(p) */

  /* Physical parameters */
  PetscReal rho;

  /* Spatial dimension */
  PetscInt dim;

  /* Field decomposition (borrowed) */
  IS is_comp[SEG_SRK_MAX_DIM]; /* per-component velocity IS */
  IS is_vel;                   /* combined velocity IS */
  IS is_p;                     /* pressure IS */

  /* Laplacian sub-matrices (owned, dt-independent) */
  Mat L_helm[SEG_SRK_MAX_DIM]; /* per-component Laplacian sub-matrix */

  /* Solvers (owned) */
  KSP ksp_helm[SEG_SRK_MAX_DIM]; /* Helmholtz KSP per component */
  KSP ksp_pres;                  /* Pressure KSP */
  Mat A_helm[SEG_SRK_MAX_DIM];   /* Helmholtz matrices (L_helm + shift*I) */
  Mat A_pres;                    /* Pressure matrix (-L, SPD compact Laplacian) */

  /* Pressure Laplacian FlucaFD (owned, built in setup) */
  FlucaFD fd_pres_lap;

  /* Null space for pressure (owned) */
  MatNullSpace pres_nullspace;

  /* Assembly state */
  PetscReal dt_assembled; /* dt used for current Helmholtz matrices */
  PetscBool first_step;
} Seg_SRK;

FLUCA_INTERN PetscErrorCode SegStep_SRK(Seg);
FLUCA_INTERN PetscErrorCode SegSRKAssembleHelmholtz(Seg);
