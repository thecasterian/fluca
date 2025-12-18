#include <flucameshcart.h>
#include <flucans.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <math.h>

const char *help = "3D lid-driven cavity flow\n";

static PetscErrorCode wall_velocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar val[], void *ctx)
{
  val[0] = 0.;
  val[1] = 0.;
  val[2] = 0.;
  return PETSC_SUCCESS;
}

static PetscErrorCode moving_wall_velocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar val[], void *ctx)
{
  val[0] = 1.;
  val[1] = 0.;
  val[2] = 0.;
  return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
  Mesh mesh;
  NS   ns;

  PetscReal Re = 100.; /* Reynolds number */
  PetscReal rho, mu;
  Vec       sol;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Re", &Re, NULL));
  rho = 1.;
  mu  = 1. / Re;

  PetscCall(MeshCartCreate3d(PETSC_COMM_WORLD, MESHCART_BOUNDARY_NONE, MESHCART_BOUNDARY_NONE, MESHCART_BOUNDARY_NONE, 64, 64, 32, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, NULL, NULL, NULL, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));
  PetscCall(MeshCartSetUniformCoordinates(mesh, 0., 1., 0., 1., 0., 0.5));

  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSCNLINEAR));
  PetscCall(NSSetMesh(ns, mesh));
  PetscCall(NSSetDensity(ns, rho));
  PetscCall(NSSetViscosity(ns, mu));

  {
    NSBoundaryCondition wallbc = {
      .type         = NS_BC_VELOCITY,
      .velocity     = wall_velocity,
      .ctx_velocity = NULL,
    };
    NSBoundaryCondition movingwallbc = {
      .type         = NS_BC_VELOCITY,
      .velocity     = moving_wall_velocity,
      .ctx_velocity = NULL,
    };
    NSBoundaryCondition symbc = {
      .type = NS_BC_SYMMETRY,
    };
    PetscInt ileftb, irightb, idownb, iupb, ibackb, ifrontb;

    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_LEFT, &ileftb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_RIGHT, &irightb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_DOWN, &idownb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_UP, &iupb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_BACK, &ibackb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_FRONT, &ifrontb));
    PetscCall(NSSetBoundaryCondition(ns, ileftb, wallbc));
    PetscCall(NSSetBoundaryCondition(ns, irightb, wallbc));
    PetscCall(NSSetBoundaryCondition(ns, idownb, wallbc));
    PetscCall(NSSetBoundaryCondition(ns, iupb, movingwallbc));
    PetscCall(NSSetBoundaryCondition(ns, ibackb, symbc));
    PetscCall(NSSetBoundaryCondition(ns, ifrontb, wallbc));
  }

  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(VecSet(sol, 0.));

  PetscCall(NSSolve(ns));

  PetscCall(MeshDestroy(&mesh));
  PetscCall(NSDestroy(&ns));
  PetscCall(FlucaFinalize());
}
