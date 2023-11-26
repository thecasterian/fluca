#include <flucamesh.h>
#include <flucans.h>
#include <flucasol.h>
#include <flucasys.h>
#include <math.h>
#include <petscdmda.h>
#include <petscviewer.h>

const char *help = "mesh test\n";

int main(int argc, char **argv) {
    Mesh mesh;
    NS ns;

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
            arrcx[i][0] = (PetscReal)i / M;
        for (j = ys; j <= ys + ym; j++)
            arrcy[j][0] = (PetscReal)j / N;
        PetscCall(MeshCartesianFaceCoordinateRestoreArray(mesh, &arrcx, &arrcy, NULL));
    }

    {
        PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
        PetscCall(NSSetType(ns, NSFSM));
        PetscCall(NSSetMesh(ns, mesh));
        PetscCall(NSSetDensity(ns, 1000.0));
        PetscCall(NSSetViscosity(ns, 1.0));
        PetscCall(NSSetTimeStepSize(ns, 0.002));
        PetscCall(NSSetFromOptions(ns));
        PetscCall(NSSetUp(ns));
        PetscCall(NSSolve(ns, 100));
    }

    {
        PetscViewer viewer;

        PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)mesh), &viewer));
        PetscCall(PetscViewerSetType(viewer, PETSCVIEWERCGNS));
        PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
        PetscCall(PetscViewerFileSetName(viewer, "fluca-%d.cgns"));
        PetscCall(NSView(ns, viewer));
        PetscCall(PetscViewerDestroy(&viewer));
    }

    PetscCall(MeshDestroy(&mesh));
    PetscCall(NSDestroy(&ns));

    PetscCall(FlucaFinalize());
}
