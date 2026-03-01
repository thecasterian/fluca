#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test convergence of TVD convection term d(u*phi)/dx\n"
                           "phi(x) = sin(2*pi*x), domain [0, 1] with periodic BCs\n"
                           "Applies FlucaFDScale(TVD) and FlucaFDDerivative in two steps\n"
                           "Options:\n"
                           "  -stag_grid_x <int>           : Number of grid cells (default: 8)\n"
                           "  -flucafd_limiter <limiter>   : Flux limiter\n"
                           "  -vel <real>                  : Uniform velocity (default: 1)\n";

int main(int argc, char **argv)
{
  DM                  dm, mass_flux_dm;
  FlucaFD             fd_tvd, fd_scaled_tvd, fd_deriv;
  Vec                 phi, mass_flux, u_phi_face, conv;
  Vec                 phi_local, mass_flux_local;
  PetscInt            N, x, m, nExtrax, i, slot_elem, slot_face;
  PetscScalar       **arr_phi, **arr_mass_flux;
  const PetscScalar **arr_coord;
  PetscInt            slot_coord_elem;
  PetscReal           error, tol, u;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create periodic DM (element DOFs only) for cell-centered phi.
     Stencil width 2 is needed for the TVD limiter ratio computation. */
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 8, 0, 1, DMSTAG_STENCIL_BOX, 2, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagGetGlobalSizes(dm, &N, NULL, NULL));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 0., 0., 0.));

  /* Create face DM for mass flux and face-interpolated phi */
  PetscCall(DMStagCreateCompatibleDMStag(dm, 1, 0, 0, 0, &mass_flux_dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(mass_flux_dm, 0., 1., 0., 0., 0., 0.));

  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, &nExtrax, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot_elem));
  PetscCall(DMStagGetLocationSlot(mass_flux_dm, DMSTAG_LEFT, 0, &slot_face));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord_elem));

  /* Create and fill phi = sin(2*pi*x) at cell centers */
  PetscCall(DMCreateGlobalVector(dm, &phi));
  PetscCall(DMGetLocalVector(dm, &phi_local));
  PetscCall(VecZeroEntries(phi_local));
  PetscCall(DMStagVecGetArray(dm, phi_local, &arr_phi));
  for (i = x; i < x + m; ++i) arr_phi[i][slot_elem] = PetscSinReal(2. * PETSC_PI * PetscRealPart(arr_coord[i][slot_coord_elem]));
  PetscCall(DMStagVecRestoreArray(dm, phi_local, &arr_phi));
  PetscCall(DMLocalToGlobal(dm, phi_local, INSERT_VALUES, phi));
  PetscCall(DMRestoreLocalVector(dm, &phi_local));

  /* Create and fill uniform mass flux at all faces (rho=1, so F = u) */
  u = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vel", &u, NULL));
  PetscCall(DMCreateGlobalVector(mass_flux_dm, &mass_flux));
  PetscCall(DMGetLocalVector(mass_flux_dm, &mass_flux_local));
  PetscCall(VecZeroEntries(mass_flux_local));
  PetscCall(DMStagVecGetArray(mass_flux_dm, mass_flux_local, &arr_mass_flux));
  for (i = x; i < x + m + nExtrax; ++i) arr_mass_flux[i][slot_face] = u;
  PetscCall(DMStagVecRestoreArray(mass_flux_dm, mass_flux_local, &arr_mass_flux));
  PetscCall(DMLocalToGlobal(mass_flux_dm, mass_flux_local, INSERT_VALUES, mass_flux));
  PetscCall(DMRestoreLocalVector(mass_flux_dm, &mass_flux_local));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));

  /* Create TVD operator: cell-center phi -> face phi */
  PetscCall(FlucaFDSecondOrderTVDCreate(dm, FLUCAFD_X, 0, 0, &fd_tvd));
  PetscCall(FlucaFDSetFromOptions(fd_tvd));
  PetscCall(FlucaFDSetUp(fd_tvd));
  PetscCall(FlucaFDSecondOrderTVDSetMassFlux(fd_tvd, mass_flux, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(fd_tvd, phi));

  /* Scale TVD by velocity: element -> face producing u*phi_face */
  PetscCall(FlucaFDScaleCreateConstant(fd_tvd, u, &fd_scaled_tvd));
  PetscCall(FlucaFDSetUp(fd_scaled_tvd));

  /* Step 1: Apply scaled TVD to get u*phi at faces */
  PetscCall(DMCreateGlobalVector(mass_flux_dm, &u_phi_face));
  PetscCall(FlucaFDApply(fd_scaled_tvd, dm, mass_flux_dm, phi, u_phi_face));

  /* First derivative: face -> element, 2nd order accuracy */
  PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &fd_deriv));
  PetscCall(FlucaFDSetUp(fd_deriv));

  /* Step 2: Apply derivative to get d(u*phi)/dx at cell centers */
  PetscCall(DMCreateGlobalVector(dm, &conv));
  PetscCall(FlucaFDApply(fd_deriv, mass_flux_dm, dm, u_phi_face, conv));

  /* Compare against exact: d(u*phi)/dx = u * 2*pi*cos(2*pi*x) */
  {
    Vec           exact;
    PetscScalar **arr_exact;
    Vec           exact_local;

    PetscCall(DMCreateGlobalVector(dm, &exact));
    PetscCall(DMGetLocalVector(dm, &exact_local));
    PetscCall(VecZeroEntries(exact_local));
    PetscCall(DMStagVecGetArray(dm, exact_local, &arr_exact));
    PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
    for (i = x; i < x + m; ++i) arr_exact[i][slot_elem] = u * 2. * PETSC_PI * PetscCosReal(2. * PETSC_PI * PetscRealPart(arr_coord[i][slot_coord_elem]));
    PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
    PetscCall(DMStagVecRestoreArray(dm, exact_local, &arr_exact));
    PetscCall(DMLocalToGlobal(dm, exact_local, INSERT_VALUES, exact));
    PetscCall(DMRestoreLocalVector(dm, &exact_local));

    PetscCall(VecAXPY(exact, -1.0, conv));
    PetscCall(VecNorm(exact, NORM_INFINITY, &error));
    PetscCall(VecDestroy(&exact));
  }

  /* Assert O(h) convergence: error < C * h.
     TVD limiters clip to first-order upwind at extrema of sin(2*pi*x),
     reducing inf-norm convergence to O(h). The leading error coefficient
     is |phi''|/2 = 2*pi^2 ~ 19.7, so C = 25 gives comfortable margin. */
  tol = 25. / (PetscReal)N;
  PetscCheck(error < tol, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Convection error too large: %g (tol %g, N=%" PetscInt_FMT ")", (double)error, (double)tol, N);

  /* Cleanup in reverse creation order */
  PetscCall(VecDestroy(&conv));
  PetscCall(FlucaFDDestroy(&fd_deriv));
  PetscCall(VecDestroy(&u_phi_face));
  PetscCall(FlucaFDDestroy(&fd_scaled_tvd));
  PetscCall(FlucaFDDestroy(&fd_tvd));
  PetscCall(VecDestroy(&mass_flux));
  PetscCall(VecDestroy(&phi));
  PetscCall(DMDestroy(&mass_flux_dm));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: superbee_16
    nsize: 1
    args: -stag_grid_x 16 -flucafd_limiter superbee
    output_file: output/empty.out

  test:
    suffix: superbee_32
    nsize: 1
    args: -stag_grid_x 32 -flucafd_limiter superbee
    output_file: output/empty.out

  test:
    suffix: superbee_64
    nsize: 1
    args: -stag_grid_x 64 -flucafd_limiter superbee
    output_file: output/empty.out

TEST*/
