#include <flucameshcart.h>
#include <flucans.h>
#include <flucasys.h>
#include <math.h>

const char *help = "lid-driven cavity flow test\n";

int main(int argc, char **argv)
{
  Mesh      mesh;
  NS        ns;
  PetscBool mesh_cart_create_from_file, ns_initialize_from_file;
  char      mesh_cart_filename[PETSC_MAX_PATH_LEN];
  char      ns_filename[PETSC_MAX_PATH_LEN];
  PetscInt  ns_steps = 10;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_cart_create_from_file", mesh_cart_filename, PETSC_MAX_PATH_LEN, &mesh_cart_create_from_file));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ns_initialize_from_file", ns_filename, PETSC_MAX_PATH_LEN, &ns_initialize_from_file));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ns_steps", &ns_steps, NULL));

  if (!mesh_cart_create_from_file) {
    PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD, MESH_BOUNDARY_NONE, MESH_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, NULL, NULL, &mesh));
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
    PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
    PetscCall(NSSetType(ns, NSFSM));
    PetscCall(NSSetMesh(ns, mesh));
    PetscCall(NSSetDensity(ns, 400.0));
    PetscCall(NSSetViscosity(ns, 1.0));
    PetscCall(NSSetTimeStepSize(ns, 0.002));
    PetscCall(NSSetFromOptions(ns));
    PetscCall(NSSetUp(ns));
    if (ns_initialize_from_file) PetscCall(NSInitializeFromFile(ns, ns_filename));
    else PetscCall(NSInitialize(ns));
    PetscCall(NSSolve(ns, ns_steps));
  }

  PetscCall(MeshDestroy(&mesh));
  PetscCall(NSDestroy(&ns));

  PetscCall(FlucaFinalize());
}
