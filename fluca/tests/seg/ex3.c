#include <flucaseg.h>
#include <flucasys.h>

static const char help[] = "Test SRK tableau set/get, default, and view\n";

int main(int argc, char **argv)
{
  Seg        seg;
  SegSRKType srktype;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetType(seg, SEGSRK));

  /* Default tableau should be ars343 */
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Default SRK type: %s\n", srktype));

  /* Set and get each registered tableau */
  PetscCall(SegSRKSetType(seg, SEGSRKARS111));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARS121));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARS222));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARS232));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARS343));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARS443));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARK324L2SA));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARK436L2SA));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKARK548L2SA));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKMARK324L2SA));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKMARS343));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegSRKSetType(seg, SEGSRKBHR553));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Set/get: %s\n", srktype));

  PetscCall(SegDestroy(&seg));

  /* SetFromOptions for SRK type */
  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetFromOptions(seg));
  PetscCall(SegSRKGetType(seg, &srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "From options SRK type: %s\n", srktype));

  /* View (pre-setup: only shows class name and SRK tableau info) */
  PetscCall(SegView(seg, NULL));

  PetscCall(SegDestroy(&seg));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: default
    nsize: 1

  test:
    suffix: from_options
    nsize: 1
    args: -seg_srk_type ark436l2sa

TEST*/
