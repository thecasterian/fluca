#include <flucaseg.h>
#include <flucasys.h>

static const char help[] = "Test Seg and SRK SetFromOptions\n";

int main(int argc, char **argv)
{
  Seg        seg;
  SegType    type;
  SegSRKType srktype;
  PetscReal  dt, max_time;
  PetscInt   max_steps;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetFromOptions(seg));

  PetscCall(SegGetType(seg, &type));
  PetscCall(SegGetTimeStepSize(seg, &dt));
  PetscCall(SegGetMaxTime(seg, &max_time));
  PetscCall(SegGetMaxSteps(seg, &max_steps));
  PetscCall(SegSRKGetType(seg, &srktype));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Type: %s\n", type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "SRK type: %s\n", srktype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dt: %g\n", (double)dt));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "max_time: %g\n", (double)max_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "max_steps: %" PetscInt_FMT "\n", max_steps));

  PetscCall(SegDestroy(&seg));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: defaults
    nsize: 1

  test:
    suffix: override
    nsize: 1
    args: -seg_dt 0.025 -seg_max_time 5.0 -seg_max_steps 500 -seg_srk_type ark436l2sa

TEST*/
