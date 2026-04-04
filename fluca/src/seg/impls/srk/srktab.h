#pragma once

#include <flucasys.h>

/* SRK Tableau: stores Butcher coefficients for an IMEX Runge-Kutta scheme.
   Registry owns the data; Seg_SRK holds a pointer into the registry. */
typedef struct _SRKTableau *SRKTableau;
struct _SRKTableau {
  char      *name;
  PetscInt   order;                     /* temporal order of accuracy */
  PetscInt   s;                         /* number of stages */
  PetscReal *A;                         /* implicit Butcher matrix (s*s, row-major) */
  PetscReal *At;                        /* explicit Butcher matrix (s*s, row-major) */
  PetscReal *b;                         /* implicit weights [s] */
  PetscReal *bt;                        /* explicit weights [s] */
  PetscReal *c;                         /* implicit abscissae [s] */
  PetscReal *ct;                        /* explicit abscissae [s] */
  PetscBool  stiffly_accurate;          /* implicit: last row of A == b */
  PetscBool  explicit_stiffly_accurate; /* explicit: last row of At == bt */
  PetscBool  fsal;
  PetscBool  explicit_first_stage;
};

/* Linked list node for the global tableau registry */
typedef struct _SRKTableauLink *SRKTableauLink;
struct _SRKTableauLink {
  struct _SRKTableau tab;
  SRKTableauLink     next;
};

FLUCA_INTERN PetscErrorCode SegSRKLookupTableau_Internal(const char[], SRKTableau *);
FLUCA_INTERN PetscErrorCode SegSRKGetTableauList_Internal(SRKTableauLink *);
