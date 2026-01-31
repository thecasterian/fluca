#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FlucaFD sum operator\n"
                           "Options:\n"
                           "  -i <int>       : X-index at which to compute stencil\n"
                           "  -j <int>       : Y-index at which to compute stencil\n"
                           "  -k <int>       : Z-index at which to compute stencil\n";

int main(int argc, char **argv)
{
  DM            dm, cdm;
  FlucaFD       fd_deriv[3], fd_sum;
  PetscInt      c, d, ncols;
  PetscInt      N[3], idx[3];
  DMStagStencil col[64];
  PetscScalar   v[64];

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, 8, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));
  PetscCall(DMGetCoordinateDM(dm, &cdm));

  for (d = 0; d < 3; ++d) {
    char prefix[PETSC_MAX_OPTION_NAME];

    PetscCall(FlucaFDCreate(PETSC_COMM_WORLD, &fd_deriv[d]));
    PetscCall(FlucaFDSetType(fd_deriv[d], FLUCAFDDERIVATIVE));
    PetscCall(FlucaFDSetCoordinateDM(fd_deriv[d], cdm));
    PetscCall(FlucaFDSetInputLocation(fd_deriv[d], DMSTAG_ELEMENT, 0));
    PetscCall(FlucaFDSetOutputLocation(fd_deriv[d], DMSTAG_ELEMENT, 0));
    PetscCall(FlucaFDDerivativeSetDirection(fd_deriv[d], (FlucaFDDirection)d));
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_OPTION_NAME, "%c_", 'x' + d));
    PetscCall(FlucaFDSetOptionsPrefix(fd_deriv[d], prefix));
    PetscCall(FlucaFDSetFromOptions(fd_deriv[d]));
    PetscCall(FlucaFDSetUp(fd_deriv[d]));
  }

  PetscCall(FlucaFDSumCreate(3, fd_deriv, &fd_sum));
  PetscCall(FlucaFDSetOptionsPrefix(fd_sum, "sum_"));
  PetscCall(FlucaFDSetFromOptions(fd_sum));
  PetscCall(FlucaFDSetUp(fd_sum));

  PetscCall(DMStagGetGlobalSizes(dm, &N[0], &N[1], &N[2]));
  for (d = 0; d < 3; ++d) {
    char opt[PETSC_MAX_OPTION_NAME];

    idx[d] = N[d] / 2;
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-%c", 'i' + d));
    PetscCall(PetscOptionsGetInt(NULL, NULL, opt, &idx[d], NULL));
  }

  PetscCall(FlucaFDGetStencil(fd_sum, idx[0], idx[1], idx[2], &ncols, col, v));
  PetscCall(SortStencil(ncols, col, v));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Sum stencil at (i,j,k)=(%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "):\n", idx[0], idx[1], idx[2]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ncols = %" PetscInt_FMT "\n", ncols));
  for (c = 0; c < ncols; ++c) {
    if (col[c].c < 0)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", k=%" PetscInt_FMT ", loc=%s, c=%s_boundary, v=%g\n", c, col[c].i, col[c].j, col[c].k, DMStagStencilLocations[col[c].loc],
                            FlucaFDBoundaryNames[-col[c].c - 1], v[c]));
    else
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", k=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g\n", c, col[c].i, col[c].j, col[c].k, DMStagStencilLocations[col[c].loc], col[c].c, v[c]));
  }

  PetscCall(FlucaFDDestroy(&fd_sum));
  for (d = 0; d < 3; ++d) PetscCall(FlucaFDDestroy(&fd_deriv[d]));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: all_first_deriv
    nsize: 1
    args: -x_flucafd_deriv_order 1 -y_flucafd_deriv_order 1 -z_flucafd_deriv_order 1

  test:
    suffix: all_second_deriv
    nsize: 1
    args: -x_flucafd_deriv_order 2 -y_flucafd_deriv_order 2 -z_flucafd_deriv_order 2

  test:
    suffix: all_second_deriv_left_bc_none
    nsize: 1
    args: -x_flucafd_deriv_order 2 -y_flucafd_deriv_order 2 -z_flucafd_deriv_order 2 -i 0

  test:
    suffix: all_second_deriv_up_bc_neumann
    nsize: 1
    args: -x_flucafd_deriv_order 2 -y_flucafd_deriv_order 2 -z_flucafd_deriv_order 2 -sum_flucafd_up_bc_type neumann -j 7

  test:
    suffix: all_second_deriv_back_bc_periodic
    nsize: 1
    args: -stag_boundary_type_z periodic -x_flucafd_deriv_order 2 -y_flucafd_deriv_order 2 -z_flucafd_deriv_order 2 -z_flucafd_back_bc_type periodic -z_flucafd_front_bc_type periodic -sum_flucafd_back_bc_type periodic -sum_flucafd_front_bc_type periodic -k 0

  test:
    suffix: all_second_deriv_all_loc_down
    nsize: 1
    args: -x_flucafd_deriv_order 2 -x_flucafd_input_loc down -x_flucafd_output_loc down -y_flucafd_deriv_order 2 -y_flucafd_input_loc down -y_flucafd_output_loc down -z_flucafd_deriv_order 2 -z_flucafd_input_loc down -z_flucafd_output_loc down -sum_flucafd_input_loc down -sum_flucafd_output_loc down

  test:
    suffix: all_second_deriv_all_loc_down_left_bc_dirichlet
    nsize: 1
    args: -x_flucafd_deriv_order 2 -x_flucafd_input_loc down -x_flucafd_output_loc down -y_flucafd_deriv_order 2 -y_flucafd_input_loc down -y_flucafd_output_loc down -z_flucafd_deriv_order 2 -z_flucafd_input_loc down -z_flucafd_output_loc down -sum_flucafd_input_loc down -sum_flucafd_output_loc down -sum_flucafd_left_bc_type dirichlet -i 0

  test:
    suffix: all_second_deriv_all_loc_down_left
    nsize: 1
    args: -x_flucafd_deriv_order 2 -x_flucafd_input_loc down_left -x_flucafd_output_loc down_left -y_flucafd_deriv_order 2 -y_flucafd_input_loc down_left -y_flucafd_output_loc down_left -z_flucafd_deriv_order 2 -z_flucafd_input_loc down_left -z_flucafd_output_loc down_left -sum_flucafd_input_loc down_left -sum_flucafd_output_loc down_left

  test:
    suffix: all_second_deriv_all_loc_down_left_left_bc_dirichlet_up_bc_neumann
    nsize: 1
    args: -x_flucafd_deriv_order 2 -x_flucafd_input_loc down_left -x_flucafd_output_loc down_left -y_flucafd_deriv_order 2 -y_flucafd_input_loc down_left -y_flucafd_output_loc down_left -z_flucafd_deriv_order 2 -z_flucafd_input_loc down_left -z_flucafd_output_loc down_left -sum_flucafd_input_loc down_left -sum_flucafd_output_loc down_left -sum_flucafd_left_bc_type dirichlet -sum_flucafd_up_bc_type neumann -i 0 -j 8

  test:
    suffix: all_first_deriv_input_loc_face_output_loc_elem
    nsize: 1
    args: -x_flucafd_deriv_order 1 -x_flucafd_input_loc left -x_flucafd_output_loc element -y_flucafd_deriv_order 1 -y_flucafd_input_loc down -y_flucafd_output_loc element -z_flucafd_deriv_order 1 -z_flucafd_input_loc back -z_flucafd_output_loc element -sum_flucafd_input_loc element -sum_flucafd_output_loc element

TEST*/
