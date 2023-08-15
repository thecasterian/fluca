#include <flucasys.h>
#include <flucamesh.h>
#include <petscviewer.h>

const char *help = "mesh test\n";

int main(int argc, char **argv) {
    Mesh mesh;

    PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

    PetscCall(MeshCreate(PETSC_COMM_WORLD, &mesh));
    PetscCall(MeshSetType(mesh, MESHCARTESIAN));
    PetscCall(MeshSetDim(mesh, 2));
    PetscCall(MeshCartesianSetSize(mesh, 8, 8, 1));
    PetscCall(MeshCartesianSetBoundaryType(mesh, MESH_BOUNDARY_PERIODIC, MESH_BOUNDARY_PERIODIC, MESH_BOUNDARY_NOT_PERIODIC));
    PetscCall(MeshCartesianSetStencilWidth(mesh, 1));
    PetscCall(MeshSetUp(mesh));
    PetscCall(MeshDestroy(&mesh));

    PetscCall(FlucaFinalize());
}
