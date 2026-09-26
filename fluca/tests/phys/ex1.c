#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test Phys: material properties, guards and the field table of the solution DM\n"
                           "Options:\n"
                           "  -dim <int> : spatial dimension, 2 or 3 (default: 2)\n";

int main(int argc, char **argv)
{
  DM                dm, sol_dm;
  Phys              phys;
  PhysBC            bc;
  PhysFieldLocation loc;
  IS                is;
  PetscReal         rho, mu;
  PetscScalar       value;
  PetscInt          dim = 2, nfields, f, k, c0, ncomp, n, dof[4] = {0, 0, 0, 0};
  const char       *name;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));

  if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  else PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, 4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));

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

  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PhysGetNumFields(phys, &nfields);
  PetscCall(PetscPopErrorHandler());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Field query before PhysSetUp(): %s\n", ierr == PETSC_ERR_ARG_WRONGSTATE ? "rejected" : "accepted"));

  bc.type       = PHYS_BC_VELOCITY;
  bc.fn         = NULL;
  bc.ctx        = NULL;
  bc.fn_dot     = NULL;
  bc.fn_dot_ctx = NULL;
  for (f = 0; f < 2 * dim; f++) PetscCall(PhysSetBoundaryCondition(phys, f, bc));

  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMStagGetDOF(sol_dm, &dof[0], &dof[1], &dof[2], &dof[3]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution DM DOF per face: %" PetscInt_FMT ", per element: %" PetscInt_FMT "\n", dof[dim - 1], dof[dim]));
  PetscCall(PhysGetNumFields(phys, &nfields));
  for (k = 0; k < nfields; ++k) {
    PetscCall(PhysGetFieldName(phys, k, &name));
    PetscCall(PhysGetField(phys, name, &loc, &c0, &ncomp));
    PetscCall(PhysGetFieldIS(phys, name, &is));
    PetscCall(ISGetSize(is, &n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: location %s, c0 %" PetscInt_FMT ", ncomp %" PetscInt_FMT ", entries %" PetscInt_FMT "\n", name, PhysFieldLocations[loc], c0, ncomp, n));
  }

  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PhysSetBoundaryCondition(phys, 0, bc);
  PetscCall(PetscPopErrorHandler());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Boundary condition change after PhysSetUp(): %s\n", ierr == PETSC_ERR_ARG_WRONGSTATE ? "rejected" : "accepted"));

  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PhysSetType(phys, PHYSLAMINAR);
  PetscCall(PetscPopErrorHandler());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Type change after PhysSetUp(): %s\n", ierr == PETSC_ERR_ARG_WRONGSTATE ? "rejected" : "accepted"));

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

  test:
    suffix: 3d
    nsize: 1
    args: -dim 3

TEST*/
