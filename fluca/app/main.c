#include <flucamesh.h>
#include <flucasys.h>
#include <math.h>
#include <petscviewer.h>

const char *help = "mesh test\n";

int main(int argc, char **argv) {
    Mesh mesh;
    PetscViewer viewer;

    PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

    PetscCall(MeshCreate(PETSC_COMM_WORLD, &mesh));
    PetscCall(MeshSetType(mesh, MESHCARTESIAN));
    PetscCall(MeshSetDim(mesh, 2));
    PetscCall(MeshCartesianSetSizes(mesh, 8, 8, 1));
    PetscCall(
        MeshCartesianSetBoundaryType(mesh, MESH_BOUNDARY_PERIODIC, MESH_BOUNDARY_PERIODIC, MESH_BOUNDARY_NOT_PERIODIC));
    PetscCall(MeshSetUp(mesh));

    PetscCall(MeshCartesianSetUniformCoordinates(mesh, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

    PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)mesh), &viewer));
    PetscCall(PetscViewerSetType(viewer, PETSCVIEWERCGNS));
    PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
    PetscCall(PetscViewerFileSetName(viewer, "mesh-%d.cgns"));
    PetscCall(MeshView(mesh, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MeshDestroy(&mesh));

    PetscCall(FlucaFinalize());
}
