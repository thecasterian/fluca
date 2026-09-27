#include <flucamesh.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test Mesh: MeshCartesian on a user-provided DMStag\n"
                           "Options:\n"
                           "  -dim <int> : spatial dimension, 1, 2 or 3 (default: 2)\n"
                           "  (DMStag options such as -stag_grid_x and -stag_boundary_type_y set the grid)\n";

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

int main(int argc, char **argv)
{
  DM       dm;
  Mesh     mesh;
  PetscInt dim = 2;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));

  PetscCall(CreateDM_Private(dim, &dm));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));
  PetscCall(MeshView(mesh, PETSC_VIEWER_STDOUT_WORLD));

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

TEST*/
