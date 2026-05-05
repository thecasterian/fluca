#include <flucaib.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaIB lifecycle and FlucaIBNone subtype\n"
                           "Options:\n"
                           "  -case <full|shortcut|default_type|view_after_setup>\n";

static PetscErrorCode CreateDM(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMStagCreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, dm));
  PetscCall(DMSetUp(*dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Full lifecycle: explicit Create + SetType + SetDM + SetUp + ApplyBC + Destroy. */
static PetscErrorCode RunFull(MPI_Comm comm)
{
  DM      dm;
  FlucaIB ib;
  Vec     v;
  DM      dm_back;

  PetscFunctionBegin;
  PetscCall(CreateDM(comm, &dm));

  PetscCall(FlucaIBCreate(comm, &ib));
  PetscCall(FlucaIBSetType(ib, FLUCAIB_NONE));
  PetscCall(FlucaIBSetDM(ib, dm));
  PetscCall(FlucaIBSetUp(ib));

  PetscCall(FlucaIBGetDM(ib, &dm_back));
  PetscCall(PetscPrintf(comm, "[full] dm matches: %s\n", dm_back == dm ? "yes" : "no"));

  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecSet(v, 7.5));
  PetscCall(FlucaIBApplyBC(ib, v, FLUCAIB_VELOCITY));

  PetscCall(FlucaIBView(ib, PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(VecDestroy(&v));
  PetscCall(FlucaIBDestroy(&ib));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Convenience constructor FlucaIBCreateNone. */
static PetscErrorCode RunShortcut(MPI_Comm comm)
{
  DM          dm;
  FlucaIB     ib;
  FlucaIBType type;

  PetscFunctionBegin;
  PetscCall(CreateDM(comm, &dm));
  PetscCall(FlucaIBCreateNone(comm, dm, &ib));

  PetscCall(FlucaIBGetType(ib, &type));
  PetscCall(PetscPrintf(comm, "[shortcut] type: %s\n", type));

  PetscCall(FlucaIBDestroy(&ib));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* SetUp without explicit SetType should default to FLUCAIB_NONE. */
static PetscErrorCode RunDefaultType(MPI_Comm comm)
{
  DM          dm;
  FlucaIB     ib;
  FlucaIBType type;

  PetscFunctionBegin;
  PetscCall(CreateDM(comm, &dm));
  PetscCall(FlucaIBCreate(comm, &ib));
  PetscCall(FlucaIBSetDM(ib, dm));
  PetscCall(FlucaIBSetUp(ib));

  PetscCall(FlucaIBGetType(ib, &type));
  PetscCall(PetscPrintf(comm, "[default_type] type after SetUp: %s\n", type));

  PetscCall(FlucaIBDestroy(&ib));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  char      case_str[32] = "full";
  PetscBool flg;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-case", case_str, sizeof(case_str), NULL));

  PetscCall(PetscStrcmp(case_str, "full", &flg));
  if (flg) PetscCall(RunFull(comm));

  PetscCall(PetscStrcmp(case_str, "shortcut", &flg));
  if (flg) PetscCall(RunShortcut(comm));

  PetscCall(PetscStrcmp(case_str, "default_type", &flg));
  if (flg) PetscCall(RunDefaultType(comm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: full
    nsize: 1
    args: -case full

  test:
    suffix: shortcut
    nsize: 1
    args: -case shortcut

  test:
    suffix: default_type
    nsize: 1
    args: -case default_type

TEST*/
