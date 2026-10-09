#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "2D lid-driven cavity flow with NS\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : grid cells per direction (default: 256)\n"
                           "  -Re <real> : Reynolds number; density 1, viscosity 1/Re (default: 100)\n"
                           "  -ns_monitor_solution cgns:cavity-%d.cgns : write the solution to CGNS every step (see -ns_monitor_solution_interval)\n";

/* u = 1 on the lid (the up face, where this BC is attached) */
static PetscErrorCode LidVelocity_Private(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = comp == 0 ? 1. : 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  Mesh      mesh;
  Phys      phys;
  NS        ns;
  Vec       sol;
  PhysBC    wall = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL}, lid = {PHYS_BC_VELOCITY, LidVelocity_Private, NULL, NULL, NULL};
  PetscReal Re = 100.;
  PetscInt  f;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Re", &Re, NULL));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 256, 256, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(phys, mesh));
  PetscCall(PhysSetDensity(phys, 1.));
  PetscCall(PhysSetViscosity(phys, 1. / Re));
  for (f = 0; f < 3; f++) PetscCall(PhysSetBoundaryCondition(phys, f, wall));
  PetscCall(PhysSetBoundaryCondition(phys, 3, lid));
  PetscCall(PhysSetFromOptions(phys));

  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSCNLINEAR));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(VecZeroEntries(sol));
  PetscCall(NSSolve(ns));

  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: re100
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -ns_time_step_size 0.01 -ns_max_steps 5

TEST*/
