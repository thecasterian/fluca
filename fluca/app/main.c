#include <flucamesh.h>
#include <flucasys.h>
#include <math.h>

const char *help = "mesh test\n";

PetscReal CoordTransform1d(PetscReal x, void *ctx) {
    (void)ctx;
    return 0.5 - 0.5 * cos(PETSC_PI * x);
}

int main(int argc, char **argv) {
    Mesh mesh;

    PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

    PetscCall(MeshCreate(PETSC_COMM_WORLD, &mesh));
    PetscCall(MeshSetType(mesh, MESHCARTESIAN));
    PetscCall(MeshSetDim(mesh, 2));
    PetscCall(MeshCartesianSetSize(mesh, 8, 8, 1));
    PetscCall(
        MeshCartesianSetBoundaryType(mesh, MESH_BOUNDARY_PERIODIC, MESH_BOUNDARY_PERIODIC, MESH_BOUNDARY_NOT_PERIODIC));
    PetscCall(MeshSetUp(mesh));
    PetscCall(MeshCartesianSetUniformCoordinates(mesh, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
    PetscCall(MeshCartesianTransformCoordinates(mesh, CoordTransform1d, CoordTransform1d, NULL, NULL));
    PetscCall(MeshDestroy(&mesh));

    PetscCall(FlucaFinalize());
}
