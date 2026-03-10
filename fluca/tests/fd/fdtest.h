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

/* Sort and print stencil points. Caller is responsible for printing the header
   and npoints lines.
   dim: 1, 2, or 3 — controls which coordinates (i / i,j / i,j,k) are printed.
   When a point has nscales > 0, scale ref coordinates are appended automatically. */
static PetscErrorCode PrintStencil(PetscInt dim, PetscInt npoints, FlucaFDStencilPoint points[])
{
  PetscInt c;
  char     coord[128], comp[128], scales[256];

  PetscFunctionBegin;
  PetscCall(SortStencil(npoints, points));
  for (c = 0; c < npoints; ++c) {
    /* Build coordinate string based on dimension */
    if (points[c].type == FLUCAFD_STENCIL_CONSTANT) {
      coord[0] = '\0';
    } else if (dim == 1) {
      PetscCall(PetscSNPrintf(coord, sizeof(coord), "i=%" PetscInt_FMT ", loc=%s, ", points[c].i, DMStagStencilLocations[points[c].loc]));
    } else if (dim == 2) {
      PetscCall(PetscSNPrintf(coord, sizeof(coord), "i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", loc=%s, ", points[c].i, points[c].j, DMStagStencilLocations[points[c].loc]));
    } else {
      PetscCall(PetscSNPrintf(coord, sizeof(coord), "i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", k=%" PetscInt_FMT ", loc=%s, ", points[c].i, points[c].j, points[c].k, DMStagStencilLocations[points[c].loc]));
    }

    /* Build type string */
    if (points[c].type == FLUCAFD_STENCIL_CONSTANT) {
      PetscCall(PetscSNPrintf(comp, sizeof(comp), "type=constant"));
    } else if (points[c].type == FLUCAFD_STENCIL_BOUNDARY) {
      PetscCall(PetscSNPrintf(comp, sizeof(comp), "type=boundary(%s), %sc=%" PetscInt_FMT, FlucaFDBoundaryNames[points[c].boundary_face], coord, points[c].c));
    } else {
      PetscCall(PetscSNPrintf(comp, sizeof(comp), "type=grid, %sc=%" PetscInt_FMT, coord, points[c].c));
    }

    /* Build scale refs suffix when present */
    if (points[c].nscales > 0) {
      PetscInt s;
      size_t   pos = 0;

      pos += (size_t)snprintf(scales + pos, sizeof(scales) - pos, ", scales=[");
      for (s = 0; s < points[c].nscales; ++s) {
        const FlucaFDScaleRef *ref = &points[c].scales[s];

        if (s > 0) pos += (size_t)snprintf(scales + pos, sizeof(scales) - pos, ", ");
        if (ref->dim == 1) pos += (size_t)snprintf(scales + pos, sizeof(scales) - pos, "(i=%" PetscInt_FMT ")", ref->i);
        else if (ref->dim == 2) pos += (size_t)snprintf(scales + pos, sizeof(scales) - pos, "(i=%" PetscInt_FMT ", j=%" PetscInt_FMT ")", ref->i, ref->j);
        else pos += (size_t)snprintf(scales + pos, sizeof(scales) - pos, "(i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", k=%" PetscInt_FMT ")", ref->i, ref->j, ref->k);
      }
      pos += (size_t)snprintf(scales + pos, sizeof(scales) - pos, "]");
      (void)pos;
    } else {
      scales[0] = '\0';
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: %s, v=%g%s\n", c, comp, (double)PetscRealPart(points[c].v), scales));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
