#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test deferred vector scale evaluation\n"
                           "Options:\n"
                           "  -i <int>  : Index at which to compute stencil\n";

static PetscErrorCode FillScaleVector(DM, Vec);

int main(int argc, char **argv)
{
  DM                  dm;
  FlucaFD             fd_deriv, fd_scale, fd_comp;
  Vec                 vec;
  PetscInt            M, idx, c, npoints_raw, npoints;
  FlucaFDStencilPoint raw_points[32], points[32];
  PetscBool           any_unresolved;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 1, 1, DMSTAG_STENCIL_STAR, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 0., 0., 0.));

  /* Create a first-derivative operator (elem -> elem) */
  PetscCall(FlucaFDCreate(PETSC_COMM_WORLD, &fd_deriv));
  PetscCall(FlucaFDSetType(fd_deriv, FLUCAFDDERIVATIVE));
  PetscCall(FlucaFDSetDM(fd_deriv, dm));
  PetscCall(FlucaFDSetInputLocation(fd_deriv, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOutputLocation(fd_deriv, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOptionsPrefix(fd_deriv, "deriv_"));
  PetscCall(FlucaFDSetFromOptions(fd_deriv));
  PetscCall(FlucaFDSetUp(fd_deriv));

  /* Create a scale vector with known values */
  PetscCall(DMCreateGlobalVector(dm, &vec));
  PetscCall(FillScaleVector(dm, vec));

  /* Create a vector-scaled operator */
  PetscCall(FlucaFDScaleCreateVector(fd_deriv, vec, 0, &fd_scale));
  PetscCall(FlucaFDSetOptionsPrefix(fd_scale, "scale_"));
  PetscCall(FlucaFDSetFromOptions(fd_scale));
  PetscCall(FlucaFDSetUp(fd_scale));

  PetscCall(DMStagGetGlobalSizes(dm, &M, NULL, NULL));
  idx = M / 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-i", &idx, NULL));

  /* TEST 1: GetStencilRaw should return nscales > 0, unresolved v */
  PetscCall(FlucaFDGetStencilRaw(fd_scale, idx, 0, 0, &npoints_raw, raw_points));
  any_unresolved = PETSC_FALSE;
  for (c = 0; c < npoints_raw; c++) {
    if (raw_points[c].nscales > 0) {
      any_unresolved = PETSC_TRUE;
      break;
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "GetStencilRaw at i=%" PetscInt_FMT ":\n", idx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints_raw));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  has_unresolved_scales = %s\n", any_unresolved ? "true" : "false"));
  {
    FlucaFDStencilPoint sorted_raw[32];
    for (c = 0; c < npoints_raw; c++) sorted_raw[c] = raw_points[c];
    PetscCall(SortStencil(npoints_raw, sorted_raw));
    for (c = 0; c < npoints_raw; c++)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g, nscales=%" PetscInt_FMT "\n", c, sorted_raw[c].i, DMStagStencilLocations[sorted_raw[c].loc], sorted_raw[c].c,
                            (double)PetscRealPart(sorted_raw[c].v), sorted_raw[c].nscales));
  }

  /* TEST 2: GetStencil should return nscales == 0, fully resolved v */
  PetscCall(FlucaFDGetStencil(fd_scale, idx, 0, 0, &npoints, points));
  PetscCall(SortStencil(npoints, points));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "GetStencil at i=%" PetscInt_FMT ":\n", idx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints));
  for (c = 0; c < npoints; c++)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g, nscales=%" PetscInt_FMT "\n", c, points[c].i, DMStagStencilLocations[points[c].loc], points[c].c,
                          (double)PetscRealPart(points[c].v), points[c].nscales));

  /* TEST 3: Composition with inner vector-scaled operator */
  {
    FlucaFD             fd_outer;
    FlucaFDStencilPoint comp_raw[32], comp_resolved[32];
    PetscInt            npoints_comp_raw, npoints_comp;

    PetscCall(FlucaFDCreate(PETSC_COMM_WORLD, &fd_outer));
    PetscCall(FlucaFDSetType(fd_outer, FLUCAFDDERIVATIVE));
    PetscCall(FlucaFDSetDM(fd_outer, dm));
    PetscCall(FlucaFDSetInputLocation(fd_outer, DMSTAG_ELEMENT, 0));
    PetscCall(FlucaFDSetOutputLocation(fd_outer, DMSTAG_ELEMENT, 0));
    PetscCall(FlucaFDSetOptionsPrefix(fd_outer, "outer_"));
    PetscCall(FlucaFDSetFromOptions(fd_outer));
    PetscCall(FlucaFDSetUp(fd_outer));

    /* comp = outer(fd_scale) -- outer derivative of vector-scaled inner derivative */
    PetscCall(FlucaFDCompositionCreate(fd_scale, fd_outer, &fd_comp));
    PetscCall(FlucaFDSetOptionsPrefix(fd_comp, "comp_"));
    PetscCall(FlucaFDSetFromOptions(fd_comp));
    PetscCall(FlucaFDSetUp(fd_comp));

    PetscCall(FlucaFDGetStencilRaw(fd_comp, idx, 0, 0, &npoints_comp_raw, comp_raw));
    any_unresolved = PETSC_FALSE;
    for (c = 0; c < npoints_comp_raw; c++) {
      if (comp_raw[c].nscales > 0) {
        any_unresolved = PETSC_TRUE;
        break;
      }
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Composition GetStencilRaw at i=%" PetscInt_FMT ":\n", idx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints_comp_raw));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  has_unresolved_scales = %s\n", any_unresolved ? "true" : "false"));

    PetscCall(FlucaFDGetStencil(fd_comp, idx, 0, 0, &npoints_comp, comp_resolved));
    PetscCall(SortStencil(npoints_comp, comp_resolved));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Composition GetStencil at i=%" PetscInt_FMT ":\n", idx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints_comp));
    for (c = 0; c < npoints_comp; c++)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g, nscales=%" PetscInt_FMT "\n", c, comp_resolved[c].i, DMStagStencilLocations[comp_resolved[c].loc], comp_resolved[c].c,
                            (double)PetscRealPart(comp_resolved[c].v), comp_resolved[c].nscales));

    PetscCall(FlucaFDDestroy(&fd_comp));
    PetscCall(FlucaFDDestroy(&fd_outer));
  }

  PetscCall(VecDestroy(&vec));
  PetscCall(FlucaFDDestroy(&fd_scale));
  PetscCall(FlucaFDDestroy(&fd_deriv));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

static PetscErrorCode FillScaleVector(DM dm, Vec vec)
{
  Vec           vec_local;
  PetscInt      x, m, nExtrax, i, slot_left, slot_elem;
  PetscScalar **arr;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &vec_local));
  PetscCall(DMGlobalToLocal(dm, vec, INSERT_VALUES, vec_local));
  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, &nExtrax, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, 0, &slot_left));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot_elem));
  PetscCall(DMStagVecGetArray(dm, vec_local, &arr));
  for (i = x; i < x + m + nExtrax; ++i) arr[i][slot_left] = 2 * i;
  for (i = x; i < x + m; ++i) arr[i][slot_elem] = 2 * i + 1;
  PetscCall(DMStagVecRestoreArray(dm, vec_local, &arr));
  PetscCall(DMLocalToGlobal(dm, vec_local, INSERT_VALUES, vec));
  PetscCall(DMRestoreLocalVector(dm, &vec_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  test:
    suffix: vec_scale_raw_vs_resolved
    nsize: 1
    args: -deriv_flucafd_deriv_order 1 -deriv_flucafd_accu_order 2 -i 3

  test:
    suffix: comp_vec_scale_raw_vs_resolved
    nsize: 1
    args: -deriv_flucafd_deriv_order 1 -deriv_flucafd_accu_order 2 -outer_flucafd_deriv_order 1 -outer_flucafd_accu_order 2 -i 3

TEST*/
