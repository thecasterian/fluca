#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test Phys INS subtype\n"
                           "Options:\n"
                           "  -phys_ins_density <real>   : Density\n"
                           "  -phys_ins_viscosity <real> : Dynamic viscosity\n";

int main(int argc, char **argv)
{
  DM       dm, sol_dm;
  Phys     phys;
  PetscInt dof0, dof1, dof2, dof3;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create 2D base DMStag with 1 vertex DOF, 1 edge DOF, 1 element DOF */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  /* Create Phys, set type and base DM */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  /* Query solution DM DOF layout */
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMStagGetDOF(sol_dm, &dof0, &dof1, &dof2, &dof3));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solution DM DOFs: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", dof0, dof1, dof2, dof3));

  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  test:
    suffix: 2d
    nsize: 1
    args: -phys_view

  test:
    suffix: 2d_params
    nsize: 1
    args: -phys_ins_density 1000 -phys_ins_viscosity 0.001 -phys_view

TEST*/
