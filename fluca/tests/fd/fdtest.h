#pragma once

#include <petscdmstag.h>
#include <flucafd.h>

/* Boundary names for printing (indexed by -c - 1) */
static const char *const FlucaFDBoundaryNames[] = {"left", "right", "down", "up", "back", "front"};

static int CompareDMStagStencil(const void *a, const void *b, void *ctx)
{
  const DMStagStencil *sa = (const DMStagStencil *)a;
  const DMStagStencil *sb = (const DMStagStencil *)b;

  /* Boundary value markers (c < 0) go to the end */
  if (sa->c < 0 && sb->c >= 0) return 1;
  if (sa->c >= 0 && sb->c < 0) return -1;

  if (sa->c < sb->c) return -1;
  if (sa->c > sb->c) return 1;

  if (sa->loc < sb->loc) return -1;
  if (sa->loc > sb->loc) return 1;

  if (sa->i < sb->i) return -1;
  if (sa->i > sb->i) return 1;

  if (sa->j < sb->j) return -1;
  if (sa->j > sb->j) return 1;

  if (sa->k < sb->k) return -1;
  if (sa->k > sb->k) return 1;

  return 0;
}

static PetscErrorCode SortStencil(PetscInt ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscFunctionBegin;
  if (ncols > 0) PetscCall(PetscTimSortWithArray(ncols, col, sizeof(DMStagStencil), v, sizeof(PetscScalar), CompareDMStagStencil, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
