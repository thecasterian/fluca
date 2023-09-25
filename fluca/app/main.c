#include <flucamesh.h>
#include <flucasol.h>
#include <flucasys.h>
#include <math.h>
#include <petscviewer.h>

const char *help = "mesh test\n";

int main(int argc, char **argv) {
    Mesh mesh;
    Sol sol;

    PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

    {
        PetscCall(MeshCreate(PETSC_COMM_WORLD, &mesh));
        PetscCall(MeshSetType(mesh, MESHCARTESIAN));
        PetscCall(MeshSetDim(mesh, 2));
        PetscCall(MeshCartesianSetSizes(mesh, 8, 8, 1));
        PetscCall(MeshCartesianSetBoundaryType(mesh, MESH_BOUNDARY_NOT_PERIODIC, MESH_BOUNDARY_NOT_PERIODIC,
                                               MESH_BOUNDARY_NOT_PERIODIC));
        PetscCall(MeshSetFromOptions(mesh));
        PetscCall(MeshSetUp(mesh));
    }

    {
        PetscInt M, N, xs, ys, xm, ym;
        PetscReal **arrcx, **arrcy;
        PetscInt i, j;

        PetscCall(MeshCartesianGetInfo(mesh, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        PetscCall(MeshCartesianGetCorners(mesh, &xs, &ys, NULL, &xm, &ym, NULL));
        PetscCall(MeshCartesianFaceCoordinateGetArray(mesh, &arrcx, &arrcy, NULL));
        for (i = xs; i <= xs + xm; i++)
            arrcx[i][0] = 0.5 - 0.5 * cos(M_PI * i / M);
        for (j = ys; j <= ys + ym; j++)
            arrcy[j][0] = 0.5 - 0.5 * cos(M_PI * j / N);
        PetscCall(MeshCartesianFaceCoordinateRestoreArray(mesh, &arrcx, &arrcy, NULL));
    }

    {
        PetscCall(SolCreate(PETSC_COMM_WORLD, &sol));
        PetscCall(SolSetType(sol, SOLFSM));
        PetscCall(SolSetMesh(sol, mesh));
    }

    {
        PetscViewer viewer;

        PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)mesh), &viewer));
        PetscCall(PetscViewerSetType(viewer, PETSCVIEWERCGNS));
        PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
        PetscCall(PetscViewerFileSetName(viewer, "mesh-%d.cgns"));
        PetscCall(SolView(sol, viewer));
        PetscCall(PetscViewerDestroy(&viewer));
    }

    PetscCall(MeshDestroy(&mesh));
    PetscCall(SolDestroy(&sol));

    PetscCall(FlucaFinalize());
}
