#include <flucameshcart.h>
#include <flucans.h>
#include <flucasys.h>
#include <math.h>

const char *help = "lid-driven cavity flow test\n";

int main(int argc, char **argv)
{
  Mesh mesh;
  NS   ns;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  {
    PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD, MESH_BOUNDARY_NONE, MESH_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, NULL, NULL, &mesh));
    PetscCall(MeshSetFromOptions(mesh));
    PetscCall(MeshSetUp(mesh));
  }

  {
    PetscInt    M, N, xs, ys, xm, ym;
    PetscReal **arrcx, **arrcy;
    PetscInt    i, j, iprev;

    PetscCall(MeshCartGetGlobalSizes(mesh, &M, &N, NULL));
    PetscCall(MeshCartGetCorners(mesh, &xs, &ys, NULL, &xm, &ym, NULL));
    PetscCall(MeshCartGetCoordinateArrays(mesh, &arrcx, &arrcy, NULL));
    PetscCall(MeshCartGetCoordinateLocationSlot(mesh, MESHCART_PREV, &iprev));
    for (i = xs; i <= xs + xm; ++i) arrcx[i][iprev] = (PetscReal)i / M;
    for (j = ys; j <= ys + ym; ++j) arrcy[j][iprev] = (PetscReal)j / N;
    PetscCall(MeshCartRestoreCoordinateArrays(mesh, &arrcx, &arrcy, NULL));
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
    PetscCall(NSSolve(ns, 10));
  }

  PetscCall(MeshDestroy(&mesh));
  PetscCall(NSDestroy(&ns));

  PetscCall(FlucaFinalize());
}
