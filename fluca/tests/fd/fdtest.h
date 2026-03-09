#pragma once

#include <petscdmstag.h>
#include <flucafd.h>

/* Boundary face names for printing (indexed by boundary_face) */
static const char *const FlucaFDBoundaryNames[] = {"left", "right", "down", "up", "back", "front"};

static int CompareFlucaFDStencilPoint(const void *a, const void *b, void *ctx)
{
  const FlucaFDStencilPoint *sa = (const FlucaFDStencilPoint *)a;
  const FlucaFDStencilPoint *sb = (const FlucaFDStencilPoint *)b;

  /* Grid points first, then boundary, then constant */
  if (sa->type != sb->type) {
    if (sa->type == FLUCAFD_STENCIL_GRID) return -1;
    if (sb->type == FLUCAFD_STENCIL_GRID) return 1;
    if (sa->type == FLUCAFD_STENCIL_BOUNDARY) return -1;
    if (sb->type == FLUCAFD_STENCIL_BOUNDARY) return 1;
  }

  if (sa->type == FLUCAFD_STENCIL_CONSTANT) return 0;

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

  if (sa->type == FLUCAFD_STENCIL_BOUNDARY) {
    if (sa->boundary_face < sb->boundary_face) return -1;
    if (sa->boundary_face > sb->boundary_face) return 1;
  }

  return 0;
}

static PetscErrorCode SortStencil(PetscInt npoints, FlucaFDStencilPoint points[])
{
  PetscFunctionBegin;
  if (npoints > 0) PetscCall(PetscTimSort(npoints, points, sizeof(FlucaFDStencilPoint), CompareFlucaFDStencilPoint, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
