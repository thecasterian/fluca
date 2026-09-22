#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test Phys INS subtype: verify solution DM DOF layout and field registry\n";

static PetscErrorCode BCVelocityZero(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                dm, sol_dm;
  Phys              phys;
  PetscInt          f, c0, ncomp, n;
  PhysINSBC         bc;
  PhysFieldLocation loc;
  IS                is;
  const char       *names[] = {PHYS_FIELD_VELOCITY, PHYS_FIELD_PRESSURE, PHYS_FIELD_FACE_VELOCITY};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create 2D base DMStag: 1 element DOF, stencil width 4 as required by PhysINS */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  /* Create Phys INS, set zero velocity BCs on all faces */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));

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

  /* Field registry */
  for (f = 0; f < 3; f++) {
    PetscCall(PhysGetField(phys, names[f], &loc, &c0, &ncomp));
    PetscCall(PhysGetFieldIS(phys, names[f], &is));
    PetscCall(ISGetSize(is, &n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Field %s: location %s, first component %" PetscInt_FMT ", components %" PetscInt_FMT ", entries %" PetscInt_FMT "\n", names[f], PhysFieldLocations[loc], c0, ncomp, n));
    PetscCall(ISDestroy(&is));
  }

  {
    const char      *names[3] = {PHYS_FIELD_VELOCITY, PHYS_FIELD_PRESSURE, PHYS_FIELD_FACE_VELOCITY};
    PhysEquationRole role;
    PetscBool        nsconst;
    PetscInt         f;

    for (f = 0; f < 3; ++f) {
      PetscCall(PhysGetFieldRole(phys, names[f], &role));
      PetscCall(PhysGetFieldNullSpaceConstant(phys, names[f], &nsconst));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: role %s, nullspace_const %s\n", names[f], PhysEquationRoles[role], PetscBools[nsconst]));
    }
  }

  {
    const char        *props[2] = {PHYS_PROPERTY_DENSITY, PHYS_PROPERTY_VISCOSITY};
    PhysPropertySource src;
    PhysFieldLocation  loc;
    PetscScalar        val;
    PetscInt           p;

    for (p = 0; p < 2; ++p) {
      PetscCall(PhysGetPropertySource(phys, props[p], &src));
      PetscCall(PhysGetPropertyLocation(phys, props[p], &loc));
      PetscCall(PhysGetPropertyConstant(phys, props[p], &val));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: source %s, location %s, value %g\n", props[p], PhysPropertySources[src], PhysFieldLocations[loc], (double)PetscRealPart(val)));
    }
  }

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
