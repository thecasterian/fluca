#pragma once

#include <flucasys.h>

/* SRK Tableau: stores Butcher coefficients for an IMEX Runge-Kutta scheme.
   Registry owns the data; Seg_SRK holds a pointer into the registry. */
typedef struct _SRKTableau *SRKTableau;
struct _SRKTableau {
  char      *name;
  PetscInt   order;                     /* temporal order of accuracy */
  PetscInt   s;                         /* number of stages */
  PetscReal *At;                        /* implicit Butcher matrix (s*s, row-major) */
  PetscReal *A;                         /* explicit Butcher matrix (s*s, row-major) */
  PetscReal *bt;                        /* implicit weights [s] */
  PetscReal *b;                         /* explicit weights [s] */
  PetscReal *ct;                        /* implicit abscissae [s] */
  PetscReal *c;                         /* explicit abscissae [s] */
  PetscBool  stiffly_accurate;          /* implicit: last row of At == bt */
  PetscBool  explicit_stiffly_accurate; /* explicit: last row of A == b */
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
