#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

static const char help[] = "Test NSViewSolution and NSLoadSolution through a CGNS file\n"
                           "Options:\n"
                           "  -file <name> : CGNS file to write and read (default: ns_ex3.cgns)\n";

static PetscErrorCode CreateNS_Private(Mesh mesh, NS *ns)
{
  Phys     phys;
  PhysBC   wall = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL};
  PetscInt f;

  PetscFunctionBegin;
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(phys, mesh));
  for (f = 0; f < 4; ++f) PetscCall(PhysSetBoundaryCondition(phys, f, wall));
  PetscCall(NSCreate(PETSC_COMM_WORLD, ns));
  PetscCall(NSSetPhys(*ns, phys));
  PetscCall(NSSetTimeStepSize(*ns, 0.05));
  PetscCall(NSSetUp(*ns));
  PetscCall(PhysDestroy(&phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  Mesh        mesh;
  NS          ns, ns2;
  Vec         sol, sol2;
  PetscViewer viewer;
  PetscInt    step;
  PetscReal   time, nrm;
  PetscRandom rnd;
  char        file[PETSC_MAX_PATH_LEN] = "ns_ex3.cgns";

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-file", file, sizeof(file), NULL));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(CreateNS_Private(mesh, &ns));
  PetscCall(NSSetTimeStep(ns, 7));
  PetscCall(NSSetTime(ns, 0.35));
  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetSeed(rnd, 42));
  PetscCall(PetscRandomSeed(rnd));
  PetscCall(VecSetRandom(sol, rnd));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer));
  PetscCall(NSViewSolution(ns, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(CreateNS_Private(mesh, &ns2));
  PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
  PetscCall(NSLoadSolution(ns2, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(NSGetTimeStep(ns2, &step));
  PetscCall(NSGetTime(ns2, &time));
  PetscCall(NSGetSolution(ns2, &sol2));
  PetscCall(VecAXPY(sol2, -1., sol));
  PetscCall(VecNorm(sol2, NORM_INFINITY, &nrm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Loaded step %" PetscInt_FMT " time %g\n", step, (double)time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max |loaded - written|: %g\n", (double)nrm));

  PetscCall(NSDestroy(&ns2));
  PetscCall(NSDestroy(&ns));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: cgns_roundtrip
    nsize: 1
    args: -file ns_ex3.cgns

TEST*/
