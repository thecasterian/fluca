#include <flucaib.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test Phys INS subtype: verify solution DM DOF layout\n";

static PetscErrorCode BCVelocityZero(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  FlucaIB   ib;
  Phys      phys;
  PetscInt  f;
  PhysINSBC bc;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create 2D base DMStag: 1 element DOF */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  /* Create Phys INS, set zero velocity BCs on all faces */
  PetscCall(FlucaIBCreateNone(PETSC_COMM_WORLD, dm, &ib));
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetIB(phys, ib));
  PetscCall(FlucaIBDestroy(&ib));

  bc.type       = PHYS_INS_BC_VELOCITY;
  bc.fn         = BCVelocityZero;
  bc.ctx        = NULL;
  bc.fn_dot     = NULL;
  bc.fn_dot_ctx = NULL;
  for (f = 0; f < 4; f++) PetscCall(PhysINSSetBoundaryCondition(phys, f, bc));

  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  /* View solution DM */
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMView(sol_dm, PETSC_VIEWER_STDOUT_WORLD));

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
