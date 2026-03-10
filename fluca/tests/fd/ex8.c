#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FlucaFD vector scale operator\n"
                           "Options:\n"
                           "  -i <int>  : Index at which to compute stencil\n";

static PetscErrorCode FillScaleVector(DM, Vec);

int main(int argc, char **argv)
{
  DM                  dm;
  FlucaFD             fd_deriv, fd_scale;
  Vec                 vec;
  PetscInt            M, idx, c, npoints_raw, npoints;
  FlucaFDStencilPoint raw_points[16], points[16];
  PetscBool           any_unresolved;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 1, 1, DMSTAG_STENCIL_STAR, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 0., 0., 0.));

  PetscCall(FlucaFDCreate(PETSC_COMM_WORLD, &fd_deriv));
  PetscCall(FlucaFDSetType(fd_deriv, FLUCAFDDERIVATIVE));
  PetscCall(FlucaFDSetDM(fd_deriv, dm));
  PetscCall(FlucaFDSetInputLocation(fd_deriv, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOutputLocation(fd_deriv, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOptionsPrefix(fd_deriv, "deriv_"));
  PetscCall(FlucaFDSetFromOptions(fd_deriv));
  PetscCall(FlucaFDSetUp(fd_deriv));

  PetscCall(DMCreateGlobalVector(dm, &vec));
  PetscCall(FillScaleVector(dm, vec));

  PetscCall(FlucaFDScaleCreateVector(fd_deriv, vec, 0, &fd_scale));
  PetscCall(FlucaFDSetOptionsPrefix(fd_scale, "scale_"));
  PetscCall(FlucaFDSetFromOptions(fd_scale));
  PetscCall(FlucaFDSetUp(fd_scale));

  PetscCall(DMStagGetGlobalSizes(dm, &M, NULL, NULL));
  idx = M / 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-i", &idx, NULL));

  /* GetStencilRaw: should have nscales > 0 (unresolved vector scale) */
  PetscCall(FlucaFDGetStencilRaw(fd_scale, idx, 0, 0, &npoints_raw, raw_points));
  PetscCall(SortStencil(npoints_raw, raw_points));
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
  for (c = 0; c < npoints_raw; c++) {
    if (raw_points[c].type == FLUCAFD_STENCIL_BOUNDARY)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%s_boundary, v=%g, nscales=%" PetscInt_FMT "\n", c, raw_points[c].i, DMStagStencilLocations[raw_points[c].loc],
                            FlucaFDBoundaryNames[raw_points[c].boundary_face], (double)PetscRealPart(raw_points[c].v), raw_points[c].nscales));
    else
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g, nscales=%" PetscInt_FMT "\n", c, raw_points[c].i, DMStagStencilLocations[raw_points[c].loc], raw_points[c].c,
                            (double)PetscRealPart(raw_points[c].v), raw_points[c].nscales));
  }

  /* GetStencil: should have nscales == 0 (fully resolved) */
  PetscCall(FlucaFDGetStencil(fd_scale, idx, 0, 0, &npoints, points));
  PetscCall(SortStencil(npoints, points));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "GetStencil at i=%" PetscInt_FMT ":\n", idx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints));
  for (c = 0; c < npoints; c++) {
    if (points[c].type == FLUCAFD_STENCIL_BOUNDARY)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%s_boundary, v=%g, nscales=%" PetscInt_FMT "\n", c, points[c].i, DMStagStencilLocations[points[c].loc], FlucaFDBoundaryNames[points[c].boundary_face],
                            (double)PetscRealPart(points[c].v), points[c].nscales));
    else
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g, nscales=%" PetscInt_FMT "\n", c, points[c].i, DMStagStencilLocations[points[c].loc], points[c].c,
                            (double)PetscRealPart(points[c].v), points[c].nscales));
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
    suffix: first_deriv_scale_vector_1
    nsize: 1
    args: -deriv_flucafd_deriv_order 1 -deriv_flucafd_accu_order 2 -i 3

  test:
    suffix: first_deriv_scale_vector_2
    nsize: 1
    args: -deriv_flucafd_deriv_order 1 -deriv_flucafd_accu_order 2 -i 6

  test:
    suffix: first_deriv_scale_vector_input_loc_elem_output_loc_left
    nsize: 1
    args: -deriv_flucafd_input_loc element -deriv_flucafd_output_loc left -deriv_flucafd_deriv_order 1 -deriv_flucafd_accu_order 2 -scale_flucafd_input_loc left -scale_flucafd_output_loc left -scale_flucafd_vec_loc left

  test:
    suffix: first_deriv_scale_vector_input_loc_elem_output_loc_left_left_bc_neumann
    nsize: 1
    args: -deriv_flucafd_input_loc element -deriv_flucafd_output_loc left -deriv_flucafd_deriv_order 1 -deriv_flucafd_accu_order 2 -scale_flucafd_input_loc left -scale_flucafd_output_loc left -scale_flucafd_vec_loc left -scale_flucafd_left_bc_type neumann -i 0

TEST*/
