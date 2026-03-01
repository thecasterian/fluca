#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FlucaFDSecondOrderTVD operator\n"
                           "Options:\n"
                           "  -i <int>                     : Index at which to compute stencil\n"
                           "  -flucafd_limiter <limiter>   : Flux limiter (minmod/vanleer/superbee/mc/vanalbada)\n";

int main(int argc, char **argv)
{
  DM                  input_dm, output_dm;
  FlucaFD             fd_tvd;
  Vec                 phi, mass_flux;
  PetscInt            M, x, m, nExtrax, i, c, ncols, idx, slot_elem, slot_face;
  PetscScalar       **arr_phi, **arr_mass_flux;
  Vec                 phi_local, mass_flux_local;
  const PetscScalar **arr_coord;
  PetscInt            slot_coord_elem;
  DMStagStencil       col[64];
  PetscScalar         v[64];

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create separate input DM (element DOFs only) and output DM (face DOFs only) */
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, &input_dm));
  PetscCall(DMSetUp(input_dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(input_dm, 0., 1., 0., 0., 0., 0.));

  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 1, 0, DMSTAG_STENCIL_BOX, 1, NULL, &output_dm));
  PetscCall(DMSetUp(output_dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(output_dm, 0., 1., 0., 0., 0., 0.));

  PetscCall(DMStagGetCorners(input_dm, &x, NULL, NULL, &m, NULL, NULL, &nExtrax, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(input_dm, DMSTAG_ELEMENT, 0, &slot_elem));
  PetscCall(DMStagGetLocationSlot(output_dm, DMSTAG_LEFT, 0, &slot_face));
  PetscCall(DMStagGetProductCoordinateArraysRead(input_dm, &arr_coord, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(input_dm, DMSTAG_ELEMENT, &slot_coord_elem));

  /* Create and fill phi vector (element-centered): phi = sin(pi * x / 2) */
  PetscCall(DMCreateGlobalVector(input_dm, &phi));
  PetscCall(DMGetLocalVector(input_dm, &phi_local));
  PetscCall(VecZeroEntries(phi_local));
  PetscCall(DMStagVecGetArray(input_dm, phi_local, &arr_phi));
  for (i = x; i < x + m; ++i) arr_phi[i][slot_elem] = PetscSinScalar(PETSC_PI * arr_coord[i][slot_coord_elem] / 2.);
  PetscCall(DMStagVecRestoreArray(input_dm, phi_local, &arr_phi));
  PetscCall(DMLocalToGlobal(input_dm, phi_local, INSERT_VALUES, phi));
  PetscCall(DMRestoreLocalVector(input_dm, &phi_local));

  /* Create and fill mass flux vector (face-centered) with constant value */
  PetscCall(DMCreateGlobalVector(output_dm, &mass_flux));
  PetscCall(DMGetLocalVector(output_dm, &mass_flux_local));
  PetscCall(VecZeroEntries(mass_flux_local));
  PetscCall(DMStagVecGetArray(output_dm, mass_flux_local, &arr_mass_flux));
  for (i = x; i < x + m + nExtrax; ++i) arr_mass_flux[i][slot_face] = 1.0;
  PetscCall(DMStagVecRestoreArray(output_dm, mass_flux_local, &arr_mass_flux));
  PetscCall(DMLocalToGlobal(output_dm, mass_flux_local, INSERT_VALUES, mass_flux));
  PetscCall(DMRestoreLocalVector(output_dm, &mass_flux_local));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(input_dm, &arr_coord, NULL, NULL));

  /* Create TVD operator: element phi -> face phi */
  PetscCall(FlucaFDSecondOrderTVDCreate(input_dm, FLUCAFD_X, 0, 0, &fd_tvd));

  /* Set boundary conditions */
  {
    FlucaFDBoundaryCondition bcs[2] = {{0}};

    bcs[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs[0].value = 0.;
    bcs[1].type  = FLUCAFD_BC_DIRICHLET;
    bcs[1].value = 1.;
    PetscCall(FlucaFDSetBoundaryConditions(fd_tvd, bcs));
  }
  PetscCall(FlucaFDSetFromOptions(fd_tvd));
  PetscCall(FlucaFDSetUp(fd_tvd));

  /* Set boundary values for BCs set in FlucaFDSetFromOptions() */
  {
    FlucaFDBoundaryCondition bcs[2];

    PetscCall(FlucaFDGetBoundaryConditions(fd_tvd, bcs));
    switch (bcs[0].type) {
    case FLUCAFD_BC_DIRICHLET:
      bcs[0].value = 0.;
      break;
    case FLUCAFD_BC_NEUMANN:
      bcs[0].value = PETSC_PI / 2.;
      break;
    default:
      break;
    }
    switch (bcs[1].type) {
    case FLUCAFD_BC_DIRICHLET:
      bcs[1].value = 1.;
      break;
    case FLUCAFD_BC_NEUMANN:
      bcs[1].value = 0.;
      break;
    default:
      break;
    }
    PetscCall(FlucaFDSetBoundaryConditions(fd_tvd, bcs));
  }

  PetscCall(FlucaFDSecondOrderTVDSetMassFlux(fd_tvd, mass_flux, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(fd_tvd, phi));

  PetscCall(DMStagGetGlobalSizes(output_dm, &M, NULL, NULL));
  idx = M / 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-i", &idx, NULL));

  PetscCall(FlucaFDGetStencil(fd_tvd, idx, 0, 0, &ncols, col, v));
  PetscCall(SortStencil(ncols, col, v));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Stencil at i=%" PetscInt_FMT ":\n", idx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ncols = %" PetscInt_FMT "\n", ncols));
  for (c = 0; c < ncols; ++c) {
    if (col[c].c == FLUCAFD_CONSTANT) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: constant, v=%g\n", c, v[c]));
    else if (col[c].c < 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%s_boundary, v=%g\n", c, col[c].i, DMStagStencilLocations[col[c].loc], FlucaFDBoundaryNames[-col[c].c - 1], v[c]));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  col[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g\n", c, col[c].i, DMStagStencilLocations[col[c].loc], col[c].c, v[c]));
  }

  PetscCall(FlucaFDDestroy(&fd_tvd));
  PetscCall(VecDestroy(&mass_flux));
  PetscCall(VecDestroy(&phi));
  PetscCall(DMDestroy(&output_dm));
  PetscCall(DMDestroy(&input_dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: vanleer
    nsize: 1
    args: -i 4 -flucafd_limiter vanleer

  test:
    suffix: upwind
    nsize: 1
    args: -i 4 -flucafd_limiter upwind

  test:
    suffix: left_bc_dirichlet
    nsize: 1
    args: -i 0 -flucafd_left_bc_type dirichlet

  test:
    suffix: left_bc_neumann
    nsize: 1
    args: -i 0 -flucafd_left_bc_type neumann

  test:
    suffix: right_bc_dirichlet
    nsize: 1
    args: -i 8 -flucafd_right_bc_type dirichlet

  test:
    suffix: right_bc_neumann
    nsize: 1
    args: -i 8 -flucafd_right_bc_type neumann

TEST*/
