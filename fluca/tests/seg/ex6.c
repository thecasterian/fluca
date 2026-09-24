#include "segtest.h"

static const char help[] = "Test that PhysLaminarSetViscosity() after SegSetUp() is picked up at assembly\n"
                           "Assembling the momentum system after changing the viscosity must equal\n"
                           "assembling it with that viscosity from the start.\n";

/* Lid-driven cavity: u = 1 on the top wall */
static PetscErrorCode LidVelocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = (comp == 0 && x[1] > 1. - 1e-12) ? 1. : 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm1, dm2, sol_dm1, sol_dm2;
  Phys      phys1, phys2;
  Seg       seg1, seg2;
  Mat       M1, M2;
  Vec       f1, f2, X1, X2;
  PetscBool mat_eq, vec_eq;
  PetscReal rho = 1., dt = 0.1;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Case 1: mu = 1 at setup, changed to 0.01 after SegSetUp() */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 6, 5, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm1));
  PetscCall(DMSetFromOptions(dm1));
  PetscCall(DMSetUp(dm1));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm1, 0., 1., 0., 1., 0., 0.));
  PetscCall(SegTestSetUp(dm1, rho, 1., LidVelocity, &phys1, &seg1));
  PetscCall(PhysLaminarSetViscosity(phys1, 0.01));
  PetscCall(PhysGetSolutionDM(phys1, &sol_dm1));
  PetscCall(SegTestCreateSystem(phys1, &M1, &f1));
  PetscCall(DMCreateGlobalVector(sol_dm1, &X1));
  PetscCall(VecZeroEntries(X1));
  PetscCall(SegCNLinearComputeMomentumSystem_Internal(seg1, 0., dt, X1, M1, f1));
  PetscCall(MatAssemblyBegin(M1, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M1, MAT_FINAL_ASSEMBLY));

  /* Case 2: mu = 0.01 from the start, independent of case 1 */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 6, 5, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm2));
  PetscCall(DMSetFromOptions(dm2));
  PetscCall(DMSetUp(dm2));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm2, 0., 1., 0., 1., 0., 0.));
  PetscCall(SegTestSetUp(dm2, rho, 0.01, LidVelocity, &phys2, &seg2));
  PetscCall(PhysGetSolutionDM(phys2, &sol_dm2));
  PetscCall(SegTestCreateSystem(phys2, &M2, &f2));
  PetscCall(DMCreateGlobalVector(sol_dm2, &X2));
  PetscCall(VecZeroEntries(X2));
  PetscCall(SegCNLinearComputeMomentumSystem_Internal(seg2, 0., dt, X2, M2, f2));
  PetscCall(MatAssemblyBegin(M2, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M2, MAT_FINAL_ASSEMBLY));

  PetscCall(MatEqual(M1, M2, &mat_eq));
  PetscCall(VecEqual(f1, f2, &vec_eq));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Matrix equal: %s\n", mat_eq ? "true" : "false"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "RHS equal: %s\n", vec_eq ? "true" : "false"));

  PetscCall(VecDestroy(&X2));
  PetscCall(VecDestroy(&f2));
  PetscCall(MatDestroy(&M2));
  PetscCall(SegDestroy(&seg2));
  PetscCall(PhysDestroy(&phys2));
  PetscCall(DMDestroy(&dm2));
  PetscCall(VecDestroy(&X1));
  PetscCall(VecDestroy(&f1));
  PetscCall(MatDestroy(&M1));
  PetscCall(SegDestroy(&seg1));
  PetscCall(PhysDestroy(&phys1));
  PetscCall(DMDestroy(&dm1));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: viscosity_refresh
    nsize: 1

TEST*/
