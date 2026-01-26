#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FlucaFD derivative operator\n"
                           "Options:\n"
                           "  -i <int>       : Index at which to compute stencil\n";

int main(int argc, char **argv)
{
  DM            dm, cdm;
  FlucaFD       fd;
  PetscInt      M, idx, c, ncols;
  DMStagStencil col[64];
  PetscScalar   v[64];

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 0., 0., 0.));
  PetscCall(DMGetCoordinateDM(dm, &cdm));

  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_X, 1, 1, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd));
  PetscCall(FlucaFDSetFromOptions(fd));
  PetscCall(FlucaFDSetUp(fd));

  PetscCall(DMStagGetGlobalSizes(dm, &M, NULL, NULL));
  idx = M / 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-i", &idx, NULL));

  PetscCall(FlucaFDGetStencil(fd, idx, 0, 0, &ncols, col, v));
  PetscCall(SortStencil(ncols, col, v));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Stencil at i=%" PetscInt_FMT ":\n", idx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ncols = %" PetscInt_FMT "\n", ncols));
  for (c = 0; c < ncols; ++c) {
    if (col[c].c < 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%s_boundary, v=%g\n", c, col[c].i, DMStagStencilLocations[col[c].loc], FlucaFDBoundaryNames[-col[c].c - 1], v[c]));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g\n", c, col[c].i, DMStagStencilLocations[col[c].loc], col[c].c, v[c]));
  }

  PetscCall(FlucaFDDestroy(&fd));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: first_deriv
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2

  test:
    suffix: second_deriv
    nsize: 1
    args: -flucafd_deriv_order 2 -flucafd_accu_order 2

  test:
    suffix: second_deriv_left_bc_none
    nsize: 1
    args: -flucafd_deriv_order 2 -flucafd_accu_order 2 -i 0

  test:
    suffix: second_deriv_left_bc_dirichlet
    nsize: 1
    args: -flucafd_left_bc_type dirichlet -flucafd_deriv_order 2 -flucafd_accu_order 2 -i 0

  test:
    suffix: second_deriv_right_bc_none
    nsize: 1
    args: -flucafd_deriv_order 2 -flucafd_accu_order 2 -i 7

  test:
    suffix: second_deriv_right_bc_neumann
    nsize: 1
    args: -flucafd_right_bc_type neumann -flucafd_deriv_order 2 -flucafd_accu_order 2 -i 7

  test:
    suffix: second_deriv_refined
    nsize: 1
    args: -stag_grid_x 16 -flucafd_deriv_order 2 -flucafd_accu_order 2

  test:
    suffix: third_deriv_left_bc_periodic
    nsize: 1
    args: -stag_stencil_width 2 -stag_boundary_type_x periodic -flucafd_left_bc_type periodic -flucafd_right_bc_type periodic -flucafd_deriv_order 3 -flucafd_accu_order 2 -i 0

  test:
    suffix: third_deriv_right_bc_periodic
    nsize: 1
    args: -stag_stencil_width 2 -stag_boundary_type_x periodic -flucafd_left_bc_type periodic -flucafd_right_bc_type periodic -flucafd_deriv_order 3 -flucafd_accu_order 2 -i 7

  test:
    suffix: third_deriv_right_bc_periodic_stencil_width_1
    nsize: 1
    args: -stag_stencil_width 1 -stag_boundary_type_x periodic -flucafd_left_bc_type periodic -flucafd_right_bc_type periodic -flucafd_deriv_order 3 -flucafd_accu_order 2 -i 7

  test:
    suffix: first_deriv_input_loc_elem_output_loc_left
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc element -flucafd_output_loc left

  test:
    suffix: first_deriv_input_loc_elem_output_loc_left_left_bc_none
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc element -flucafd_output_loc left -i 0

  test:
    suffix: first_deriv_input_loc_elem_output_loc_left_left_bc_neumann
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc element -flucafd_output_loc left -flucafd_left_bc_type neumann -i 0

  test:
    suffix: first_deriv_input_loc_elem_output_loc_left_left_bc_periodic
    nsize: 1
    args: -stag_boundary_type_x periodic -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc element -flucafd_output_loc left -flucafd_left_bc_type periodic -flucafd_right_bc_type periodic -i 0

  test:
    suffix: first_deriv_input_loc_left_output_loc_left
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_left_bc_none
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -i 0

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_left_bc_dirichlet
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -flucafd_left_bc_type dirichlet -i 0

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_left_bc_neumann
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -flucafd_left_bc_type neumann -i 0

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_left_bc_periodic
    nsize: 1
    args: -stag_boundary_type_x periodic -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -flucafd_left_bc_type periodic -flucafd_right_bc_type periodic -i 0

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_right_bc_none
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -i 8

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_right_bc_dirichlet
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -flucafd_right_bc_type dirichlet -i 8

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_right_bc_neumann
    nsize: 1
    args: -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -flucafd_right_bc_type neumann -i 8

  test:
    suffix: first_deriv_input_loc_left_output_loc_left_right_bc_periodic
    nsize: 1
    args: -stag_boundary_type_x periodic -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -flucafd_left_bc_type periodic -flucafd_right_bc_type periodic -i 7

  test:
    suffix: second_deriv_input_loc_left_output_loc_left_right_bc_periodic
    nsize: 1
    args: -stag_boundary_type_x periodic -flucafd_deriv_order 2 -flucafd_accu_order 2 -flucafd_input_loc left -flucafd_output_loc left -flucafd_left_bc_type periodic -flucafd_right_bc_type periodic -i 7

TEST*/
