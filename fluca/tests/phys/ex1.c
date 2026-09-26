#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test Phys: material properties, boundary condition guards and solution DM layout\n";

int main(int argc, char **argv)
{
  DM             dm, sol_dm;
  Phys           phys;
  PhysBC         bc;
  PetscReal      rho, mu;
  PetscScalar    value;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(phys, dm));

  PetscCall(PhysGetDensity(phys, &rho));
  PetscCall(PhysGetViscosity(phys, &mu));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Default density: %g, viscosity: %g\n", (double)rho, (double)mu));
  PetscCall(PhysSetDensity(phys, 2.));
  PetscCall(PhysSetViscosity(phys, 0.5));
  PetscCall(PhysGetProperty(phys, PHYS_PROPERTY_DENSITY, &value));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Density: %g, ", (double)PetscRealPart(value)));
  PetscCall(PhysGetProperty(phys, PHYS_PROPERTY_VISCOSITY, &value));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "viscosity: %g\n", (double)PetscRealPart(value)));

  bc.type       = PHYS_BC_VELOCITY;
  bc.fn         = NULL;
  bc.ctx        = NULL;
  bc.fn_dot     = NULL;
  bc.fn_dot_ctx = NULL;
  for (f = 0; f < 4; f++) PetscCall(PhysSetBoundaryCondition(phys, f, bc));

  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMView(sol_dm, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PhysSetBoundaryCondition(phys, 0, bc);
  PetscCall(PetscPopErrorHandler());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Boundary condition change after PhysSetUp(): %s\n", ierr == PETSC_ERR_ARG_WRONGSTATE ? "rejected" : "accepted"));

  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PhysSetDensity(phys, -1.);
  PetscCall(PetscPopErrorHandler());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Negative density: %s\n", ierr == PETSC_ERR_ARG_OUTOFRANGE ? "rejected" : "accepted"));

  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d
    nsize: 1

TEST*/
