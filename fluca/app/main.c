#include <flucameshcart.h>
#include <flucans.h>
#include <flucasys.h>
#include <math.h>

const char *help = "lid-driven cavity flow test\n";

static PetscErrorCode wall_velocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar val[], void *ctx)
{
  PetscInt i;

  for (i = 0; i < dim; ++i) val[i] = 0.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode moving_wall_velocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar val[], void *ctx)
{
  PetscInt i;

  for (i = 0; i < dim; ++i) val[i] = (i == 0) ? 1.0 : 0.0; // u = 1.0, other components = 0.0
  return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
  Mesh      mesh;
  NS        ns;
  PetscBool mesh_cart_create_from_file, ns_load_solution_from_file;
  char      mesh_cart_filename[PETSC_MAX_PATH_LEN];
  char      ns_filename[PETSC_MAX_PATH_LEN];
  PetscInt  ns_steps = 10;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_cart_create_from_file", mesh_cart_filename, PETSC_MAX_PATH_LEN, &mesh_cart_create_from_file));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ns_load_solution_from_file", ns_filename, PETSC_MAX_PATH_LEN, &ns_load_solution_from_file));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ns_steps", &ns_steps, NULL));

  if (!mesh_cart_create_from_file) {
    PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD, MESHCART_BOUNDARY_NONE, MESHCART_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, NULL, NULL, &mesh));
    PetscCall(MeshSetFromOptions(mesh));
    PetscCall(MeshSetUp(mesh));

    {
      PetscInt      M, N, x, y, m, n;
      PetscScalar **arrcx, **arrcy;
      PetscInt      i, j, iprev;

      PetscCall(MeshCartGetGlobalSizes(mesh, &M, &N, NULL));
      PetscCall(MeshCartGetCorners(mesh, &x, &y, NULL, &m, &n, NULL));
      PetscCall(MeshCartGetCoordinateArrays(mesh, &arrcx, &arrcy, NULL));
      PetscCall(MeshCartGetCoordinateLocationSlot(mesh, MESHCART_PREV, &iprev));
      for (i = x; i <= x + m; ++i) arrcx[i][iprev] = (PetscScalar)i / M;
      for (j = y; j <= y + n; ++j) arrcy[j][iprev] = (PetscScalar)j / N;
      PetscCall(MeshCartRestoreCoordinateArrays(mesh, &arrcx, &arrcy, NULL));
    }
  } else {
    PetscCall(MeshCartCreateFromFile(PETSC_COMM_WORLD, mesh_cart_filename, NULL, &mesh));
  }

  {
    PetscInt            ileftb, irightb, idownb, iupb;
    NSBoundaryCondition bcwall = {
      .type         = NS_BC_VELOCITY,
      .velocity     = wall_velocity,
      .ctx_velocity = NULL,
    };
    NSBoundaryCondition bcmovingwall = {
      .type         = NS_BC_VELOCITY,
      .velocity     = moving_wall_velocity,
      .ctx_velocity = NULL,
    };

    PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
    PetscCall(NSSetType(ns, NSFSM));
    PetscCall(NSSetMesh(ns, mesh));
    PetscCall(NSSetDensity(ns, 400.0));
    PetscCall(NSSetViscosity(ns, 1.0));
    PetscCall(NSSetTimeStepSize(ns, 0.002));

    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_LEFT, &ileftb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_RIGHT, &irightb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_DOWN, &idownb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_UP, &iupb));
    PetscCall(NSSetBoundaryCondition(ns, ileftb, bcwall));
    PetscCall(NSSetBoundaryCondition(ns, irightb, bcwall));
    PetscCall(NSSetBoundaryCondition(ns, idownb, bcwall));
    PetscCall(NSSetBoundaryCondition(ns, iupb, bcmovingwall));

    PetscCall(NSSetFromOptions(ns));
    PetscCall(NSSetUp(ns));
    if (ns_load_solution_from_file) PetscCall(NSLoadSolutionFromFile(ns, ns_filename));
    PetscCall(NSSolve(ns, ns_steps));
  }

  PetscCall(MeshDestroy(&mesh));
  PetscCall(NSDestroy(&ns));

  PetscCall(FlucaFinalize());
}
