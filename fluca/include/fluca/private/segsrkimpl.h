#pragma once

#include <fluca/private/segimpl.h>
#include <petscksp.h>

/* Forward declaration — full definition in srktab.h (internal to srk/) */
typedef struct _SRKTableau *SRKTableau;

#define SEG_SRK_MAX_DIM 3

typedef struct {
  /* Tableau (pointer into global registry, not owned) */
  SRKTableau tableau;

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

  /* mu/mu_tilde recurrence for pressure prediction (Section 5.3) */
  Vec *mu_tilde; /* per-stage pressure vectors [s] (pressure-sized, owned) */
  Vec  mu_work;  /* pressure-sized work vector for mu_j computation */

  /* FlucaFD operators (borrowed, not owned) */
  FlucaFD fd_diff[SEG_SRK_MAX_DIM];   /* viscous diffusion F_diff per velocity component */
  FlucaFD fd_grad_p[SEG_SRK_MAX_DIM]; /* pressure gradient per direction */
  FlucaFD fd_div;                     /* divergence (includes rho) */
  FlucaFD fd_pres_lap;                /* compact pressure Laplacian L */

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

  /* Null space for pressure (owned) */
  MatNullSpace pres_nullspace;

  /* Assembly state */
  PetscReal dt_assembled; /* dt used for current Helmholtz matrices */
  PetscBool first_step;
} Seg_SRK;

FLUCA_INTERN PetscErrorCode SegStep_SRK(Seg);
FLUCA_INTERN PetscErrorCode SegSRKAssembleHelmholtz(Seg);
