#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FlucaFD constant scale operator\n"
                           "Options:\n"
                           "  -i <int>  : Index at which to compute stencil\n";

int main(int argc, char **argv)
{
  DM                  dm;
  FlucaFD             fd_deriv, fd_scale;
  PetscInt            M, idx, c, npoints;
  FlucaFDStencilPoint points[16];

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

  PetscCall(FlucaFDScaleCreateConstant(fd_deriv, 1., &fd_scale));
  PetscCall(FlucaFDSetOptionsPrefix(fd_scale, "scale_"));
  PetscCall(FlucaFDSetFromOptions(fd_scale));
  PetscCall(FlucaFDSetUp(fd_scale));

  PetscCall(DMStagGetGlobalSizes(dm, &M, NULL, NULL));
  idx = M / 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-i", &idx, NULL));

  PetscCall(FlucaFDGetStencil(fd_scale, idx, 0, 0, &npoints, points));
  PetscCall(SortStencil(npoints, points));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Scaled stencil at i=%" PetscInt_FMT ":\n", idx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints));
  for (c = 0; c < npoints; ++c) {
    if (points[c].type == FLUCAFD_STENCIL_BOUNDARY)
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%s_boundary, v=%g\n", c, points[c].i, DMStagStencilLocations[points[c].loc], FlucaFDBoundaryNames[points[c].boundary_face],
                            (double)PetscRealPart(points[c].v)));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  points[%" PetscInt_FMT "]: i=%" PetscInt_FMT ", loc=%s, c=%" PetscInt_FMT ", v=%g\n", c, points[c].i, DMStagStencilLocations[points[c].loc], points[c].c, (double)PetscRealPart(points[c].v)));
  }

  PetscCall(FlucaFDDestroy(&fd_scale));
  PetscCall(FlucaFDDestroy(&fd_deriv));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: first_deriv_scale_const
    nsize: 1
    args: -deriv_flucafd_deriv_order 1 -deriv_flucafd_accu_order 2 -scale_flucafd_constant 1.5

  test:
    suffix: second_deriv_right_bc_dirichlet_scale_const
    nsize: 1
    args: -deriv_flucafd_deriv_order 2 -deriv_flucafd_accu_order 2 -scale_flucafd_constant 1.5 -scale_flucafd_right_bc_type dirichlet -i 7

TEST*/
