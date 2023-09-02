#include <flucamesh.h>
#include <flucasys.h>
#include <math.h>
#include <petscviewer.h>

const char *help = "mesh test\n";

int main(int argc, char **argv) {
    PetscLogStage stage;
    Mesh mesh;

    PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

    PetscCall(PetscLogStageRegister("Assemble mesh", &stage));
    PetscCall(PetscLogStagePush(stage));
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
        PetscInt i, j, ileft, ibottom;

        PetscCall(MeshCartesianGetInfo(mesh, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        PetscCall(MeshCartesianGetCorners(mesh, &xs, &ys, NULL, &xm, &ym, NULL));
        PetscCall(MeshCartesianCoordinateVecGetArray(mesh, &arrcx, &arrcy, NULL));
        PetscCall(MeshCartesianCoordinateGetLocationSlot(mesh, MESHCARTESIAN_COORDINATE_LEFT, &ileft));
        PetscCall(MeshCartesianCoordinateGetLocationSlot(mesh, MESHCARTESIAN_COORDINATE_BOTTOM, &ibottom));
        for (i = xs; i <= xs + xm; i++)
            arrcx[i][ileft] = 0.5 - 0.5 * cos(M_PI * i / M);
        for (j = ys; j <= ys + ym; j++)
            arrcy[j][ibottom] = 0.5 - 0.5 * cos(M_PI * j / N);
        PetscCall(MeshCartesianCoordinateVecRestoreArray(mesh, &arrcx, &arrcy, NULL));
    }
    PetscCall(PetscLogStagePop());

    PetscCall(PetscLogStageRegister("View mesh", &stage));
    PetscCall(PetscLogStagePush(stage));
    {
        PetscViewer viewer;

        PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)mesh), &viewer));
        PetscCall(PetscViewerSetType(viewer, PETSCVIEWERCGNS));
        PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
        PetscCall(PetscViewerFileSetName(viewer, "mesh-%d.cgns"));
        PetscCall(MeshView(mesh, viewer));
        PetscCall(PetscViewerDestroy(&viewer));
    }
    PetscCall(PetscLogStagePop());

    PetscCall(MeshDestroy(&mesh));

    PetscCall(FlucaFinalize());
}
