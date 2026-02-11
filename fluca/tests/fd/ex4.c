#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FlucaFD composition operator\n"
                           "Options:\n"
                           "  -i <int>       : X-index at which to compute stencil\n"
                           "  -j <int>       : Y-index at which to compute stencil\n";

int main(int argc, char **argv)
{
  DM            dm;
  FlucaFD       fd_inner, fd_outer, fd_comp;
  PetscInt      c, d, ncols;
  PetscInt      N[2], idx[2];
  DMStagStencil col[64];
  PetscScalar   v[64];

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, DMSTAG_STENCIL_BOX, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  PetscCall(FlucaFDCreate(PETSC_COMM_WORLD, &fd_inner));
  PetscCall(FlucaFDSetType(fd_inner, FLUCAFDDERIVATIVE));
  PetscCall(FlucaFDSetDM(fd_inner, dm));
  PetscCall(FlucaFDSetInputLocation(fd_inner, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOutputLocation(fd_inner, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOptionsPrefix(fd_inner, "inner_"));
  PetscCall(FlucaFDSetFromOptions(fd_inner));
  PetscCall(FlucaFDSetUp(fd_inner));

  PetscCall(FlucaFDCreate(PETSC_COMM_WORLD, &fd_outer));
  PetscCall(FlucaFDSetType(fd_outer, FLUCAFDDERIVATIVE));
  PetscCall(FlucaFDSetDM(fd_outer, dm));
  PetscCall(FlucaFDSetInputLocation(fd_outer, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOutputLocation(fd_outer, DMSTAG_ELEMENT, 0));
  PetscCall(FlucaFDSetOptionsPrefix(fd_outer, "outer_"));
  PetscCall(FlucaFDSetFromOptions(fd_outer));
  PetscCall(FlucaFDSetUp(fd_outer));

  PetscCall(FlucaFDCompositionCreate(fd_inner, fd_outer, &fd_comp));
  PetscCall(FlucaFDSetOptionsPrefix(fd_comp, "comp_"));
  PetscCall(FlucaFDSetFromOptions(fd_comp));
  PetscCall(FlucaFDSetUp(fd_comp));

  PetscCall(DMStagGetGlobalSizes(dm, &N[0], &N[1], NULL));
  for (d = 0; d < 2; ++d) {
    char opt[PETSC_MAX_OPTION_NAME];

    idx[d] = N[d] / 2;
    PetscCall(PetscSNPrintf(opt, PETSC_MAX_OPTION_NAME, "-%c", 'i' + d));
    PetscCall(PetscOptionsGetInt(NULL, NULL, opt, &idx[d], NULL));
  }

  PetscCall(FlucaFDGetStencil(fd_comp, idx[0], idx[1], 0, &ncols, col, v));
  PetscCall(SortStencil(ncols, col, v));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Sum stencil at (i,j)=(%" PetscInt_FMT ",%" PetscInt_FMT "):\n", idx[0], idx[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ncols = %" PetscInt_FMT "\n", ncols));
  for (c = 0; c < ncols; ++c) {
    if (col[c].c < 0)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", loc=%s, c=%s_boundary, v=%g\n", c, col[c].i, col[c].j, DMStagStencilLocations[col[c].loc], FlucaFDBoundaryNames[-col[c].c - 1], v[c]));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", j=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g\n", c, col[c].i, col[c].j, DMStagStencilLocations[col[c].loc], col[c].c, v[c]));
  }

  PetscCall(FlucaFDDestroy(&fd_comp));
  PetscCall(FlucaFDDestroy(&fd_outer));
  PetscCall(FlucaFDDestroy(&fd_inner));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: second_deriv
    nsize: 1
    args: -inner_flucafd_deriv_order 1 -inner_flucafd_accu_order 2 -outer_flucafd_deriv_order 1 -outer_flucafd_accu_order 2

  test:
    suffix: second_deriv_compact
    nsize: 1
    args: -inner_flucafd_deriv_order 1 -inner_flucafd_accu_order 2 -inner_flucafd_output_loc left -outer_flucafd_deriv_order 1 -outer_flucafd_accu_order 2 -outer_flucafd_input_loc left

  test:
    suffix: all_first_deriv_first_accuracy_right_boundary
    nsize: 1
    args: -inner_flucafd_deriv_order 1 -inner_flucafd_accu_order 1 -outer_flucafd_deriv_order 1 -outer_flucafd_accu_order 1 -i 7

  test:
    suffix: xy_first_deriv_second_accuracy
    nsize: 1
    args: -inner_flucafd_dir x -inner_flucafd_deriv_order 1 -inner_flucafd_accu_order 2 -outer_flucafd_dir y -outer_flucafd_deriv_order 1 -outer_flucafd_accu_order 2

  test:
    suffix: yx_first_deriv_second_accuracy
    nsize: 1
    args: -inner_flucafd_dir y -inner_flucafd_deriv_order 1 -inner_flucafd_accu_order 2 -outer_flucafd_dir x -outer_flucafd_deriv_order 1 -outer_flucafd_accu_order 2

  test:
    suffix: xy_first_deriv_second_accuracy_left_bc_dirichlet
    nsize: 1
    args: -inner_flucafd_dir x -inner_flucafd_deriv_order 1 -inner_flucafd_accu_order 2 -outer_flucafd_dir y -outer_flucafd_deriv_order 1 -outer_flucafd_accu_order 2 -comp_flucafd_left_bc_type dirichlet -i 0

TEST*/
