#include <flucameshcart.h>
#include <flucansfsm.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <math.h>

const char *help = "cavity flow\n";

static PetscErrorCode wall_velocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar val[], void *ctx)
{
  val[0] = 0.;
  val[1] = 0.;
  return PETSC_SUCCESS;
}

static PetscErrorCode moving_wall_velocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar val[], void *ctx)
{
  val[0] = 1.;
  val[1] = 0.;
  return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
  Mesh mesh;
  NS   ns;

  PetscReal Re     = 100.;   /* Reynolds number */
  PetscReal dt     = 0.0005; /* Time step size */
  PetscInt  nsteps = 1;      /* Number of time steps */
  PetscReal rho, mu;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Re", &Re, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsteps", &nsteps, NULL));
  rho = 1.;
  mu  = 1. / Re;

  PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD, MESHCART_BOUNDARY_NONE, MESHCART_BOUNDARY_NONE, 256, 256, PETSC_DECIDE, PETSC_DECIDE, NULL, NULL, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));
  PetscCall(MeshCartSetUniformCoordinates(mesh, 0., 1., 0., 1., 0., 0.));

  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSFSM));
  PetscCall(NSSetMesh(ns, mesh));
  PetscCall(NSSetDensity(ns, rho));
  PetscCall(NSSetViscosity(ns, mu));
  PetscCall(NSSetTimeStepSize(ns, dt));

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
    PetscInt ileftb, irightb, idownb, iupb;

    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_LEFT, &ileftb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_RIGHT, &irightb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_DOWN, &idownb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_UP, &iupb));
    PetscCall(NSSetBoundaryCondition(ns, ileftb, wallbc));
    PetscCall(NSSetBoundaryCondition(ns, irightb, wallbc));
    PetscCall(NSSetBoundaryCondition(ns, idownb, wallbc));
    PetscCall(NSSetBoundaryCondition(ns, iupb, movingwallbc));
  }

  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  {
    Vec u, v, p, Nu, Nv, Nu_prev, Nv_prev;

    PetscCall(NSFSMGetVelocity(ns, &u, &v, NULL));
    PetscCall(NSFSMGetHalfStepPressure(ns, &p));
    PetscCall(NSFSMGetConvection(ns, &Nu, &Nv, NULL));
    PetscCall(NSFSMGetPreviousConvection(ns, &Nu_prev, &Nv_prev, NULL));

    PetscCall(VecSet(u, 0.));
    PetscCall(VecSet(v, 0.));
    PetscCall(VecSet(p, 0.));
    PetscCall(VecSet(Nu, 0.));
    PetscCall(VecSet(Nv, 0.));
    PetscCall(VecSet(Nu_prev, 0.));
    PetscCall(VecSet(Nv_prev, 0.));
  }

  PetscCall(NSSolve(ns, nsteps));

  PetscCall(MeshDestroy(&mesh));
  PetscCall(NSDestroy(&ns));

  PetscCall(FlucaFinalize());
}
