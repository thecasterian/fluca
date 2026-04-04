#include <flucaseg.h>
#include <flucasys.h>

static const char help[] = "Test Seg lifecycle and property set/get\n";

int main(int argc, char **argv)
{
  Seg         seg;
  SegType     type;
  PetscReal   dt, t, max_time;
  PetscInt    max_steps, step_number;
  const char *prefix;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create and destroy empty Seg (no type set) */
  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegDestroy(&seg));

  /* Create, set type, query type */
  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetType(seg, SEGSRK));
  PetscCall(SegGetType(seg, &type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Type: %s\n", type));

  /* Default time properties */
  PetscCall(SegGetTime(seg, &t));
  PetscCall(SegGetStepNumber(seg, &step_number));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Default time: %g\n", (double)t));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Default step number: %" PetscInt_FMT "\n", step_number));

  /* Set and get time step size */
  PetscCall(SegSetTimeStepSize(seg, 0.005));
  PetscCall(SegGetTimeStepSize(seg, &dt));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dt: %g\n", (double)dt));

  /* Set and get time */
  PetscCall(SegSetTime(seg, 1.5));
  PetscCall(SegGetTime(seg, &t));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "t: %g\n", (double)t));

  /* Set and get max time */
  PetscCall(SegSetMaxTime(seg, 10.0));
  PetscCall(SegGetMaxTime(seg, &max_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "max_time: %g\n", (double)max_time));

  /* Set and get max steps */
  PetscCall(SegSetMaxSteps(seg, 200));
  PetscCall(SegGetMaxSteps(seg, &max_steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "max_steps: %" PetscInt_FMT "\n", max_steps));

  /* Options prefix */
  PetscCall(SegSetOptionsPrefix(seg, "test_"));
  PetscCall(SegGetOptionsPrefix(seg, &prefix));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Prefix: %s\n", prefix));

  PetscCall(SegAppendOptionsPrefix(seg, "sub_"));
  PetscCall(SegGetOptionsPrefix(seg, &prefix));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Prefix after append: %s\n", prefix));

  PetscCall(SegDestroy(&seg));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: lifecycle
    nsize: 1

TEST*/
