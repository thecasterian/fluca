#include <flucamesh.h>
#include <flucasys.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

static const char help[] = "Test Mesh: MeshCartesian on a user-provided DMStag\n"
                           "Options:\n"
                           "  -dim <int> : spatial dimension, 1, 2 or 3 (default: 2)\n"
                           "  (DMStag options such as -stag_grid_x and -stag_boundary_type_y set the grid)\n"
                           "  -roundtrip <file> : write the stretched grid to a CGNS file, reload it into a new Mesh and print it\n";

static PetscErrorCode CreateDM_Private(PetscInt dim, DM *dm)
{
  PetscFunctionBegin;
  switch (dim) {
  case 1:
    PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 4, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, dm));
    break;
  case 2:
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, dm));
    break;
  case 3:
    PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, 4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, dm));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);
  }
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetUp(*dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm, 0., 1., 0., 2., 0., 3.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Stretch direction 0 to x_i = (i/N)^2 on every vertex slot the rank holds, element centers at midpoints */
static PetscErrorCode StretchX_Private(DM dm)
{
  PetscScalar **arr[3] = {NULL, NULL, NULL};
  PetscInt      gx, gm, N, iprev, ielem, i;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &N, NULL, NULL));
  PetscCall(DMStagGetGhostCorners(dm, &gx, NULL, NULL, &gm, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielem));
  for (i = gx; i < gx + gm; ++i) {
    arr[0][i][iprev] = ((PetscReal)i / N) * ((PetscReal)i / N);
    arr[0][i][ielem] = 0.5 * (((PetscReal)i / N) * ((PetscReal)i / N) + ((PetscReal)(i + 1) / N) * ((PetscReal)(i + 1) / N));
  }
  PetscCall(DMStagRestoreProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Vertex and element-center coordinates over the ghosted range of each direction */
static PetscErrorCode PrintCoordinates_Private(Mesh mesh)
{
  DM                  dm;
  const PetscScalar **arr[3] = {NULL, NULL, NULL};
  PetscInt            dim, gx[3], gm[3], iprev, ielem, d, i;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(MeshGetDimension(mesh, &dim));
  PetscCall(DMStagGetGhostCorners(dm, &gx[0], &gx[1], &gx[2], &gm[0], &gm[1], &gm[2]));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielem));
  for (d = 0; d < dim; ++d) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Direction %" PetscInt_FMT " vertices:", d));
    for (i = gx[d]; i < gx[d] + gm[d]; ++i) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %.6g", (double)PetscRealPart(arr[d][i][iprev])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nDirection %" PetscInt_FMT " centers:", d));
    for (i = gx[d]; i < gx[d] + gm[d]; ++i) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %.6g", (double)PetscRealPart(arr[d][i][ielem])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  Mesh        mesh, loaded;
  PetscInt    dim = 2;
  char        file[PETSC_MAX_PATH_LEN];
  PetscBool   roundtrip;
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-roundtrip", file, sizeof(file), &roundtrip));

  PetscCall(CreateDM_Private(dim, &dm));
  if (roundtrip) PetscCall(StretchX_Private(dm));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));
  if (!roundtrip) PetscCall(MeshView(mesh, PETSC_VIEWER_STDOUT_WORLD));
  else {
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer));
    PetscCall(MeshView(mesh, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MeshCreate(PETSC_COMM_WORLD, &loaded));
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MeshLoad(loaded, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(MeshSetUp(loaded));
    PetscCall(MeshView(loaded, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PrintCoordinates_Private(loaded));
    PetscCall(MeshDestroy(&loaded));
  }

  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: view_ascii_1d
    nsize: 1
    args: -dim 1 -stag_grid_x 8

  test:
    suffix: view_ascii_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 8 -stag_grid_y 4 -stag_boundary_type_y periodic

  test:
    suffix: view_ascii_3d
    nsize: 1
    args: -dim 3 -stag_grid_x 4 -stag_grid_y 3 -stag_grid_z 2 -stag_boundary_type_z periodic

  test:
    suffix: cgns_roundtrip_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 4 -stag_grid_y 3 -stag_boundary_type_y periodic -roundtrip mesh_ex1_2d.cgns

  test:
    suffix: cgns_roundtrip_3d
    nsize: 1
    args: -dim 3 -stag_grid_x 3 -stag_grid_y 2 -stag_grid_z 2 -stag_boundary_type_x periodic -roundtrip mesh_ex1_3d.cgns

TEST*/
