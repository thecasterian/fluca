#include <flucaphys.h>
#include <flucaseg.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "Verify Seg SRK lifecycle: create, setup, step, destroy\n"
                           "Sets up a 2D TGV problem and runs one SRK step.\n";

/* TGV exact solution at t=0 */
static PetscErrorCode FillIC(DM dm, Vec Y)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem, i, j, c;
  PetscReal           x, y;
  PetscScalar         vals[3];
  DMStagStencil       stencils[3];

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      x = PetscRealPart(arrc[0][i][slot_elem]);
      y = PetscRealPart(arrc[1][j][slot_elem]);

      vals[0] = -PetscCosReal(x) * PetscSinReal(y);
      vals[1] = PetscSinReal(x) * PetscCosReal(y);
      vals[2] = -0.25 * (PetscCosReal(2. * x) + PetscCosReal(2. * y));

      for (c = 0; c < 3; c++) {
        stencils[c].i   = i;
        stencils[c].j   = j;
        stencils[c].k   = 0;
        stencils[c].loc = DMSTAG_ELEMENT;
        stencils[c].c   = c;
      }
      PetscCall(DMStagVecSetValuesStencil(dm, Y, 3, stencils, vals, INSERT_VALUES));
    }
  }

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  Seg       seg;
  Vec       Y;
  PetscReal norm_before, norm_after;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create small 2D periodic grid */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));

  /* Create PhysINS */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysINSSetDensity(phys, 1.));
  PetscCall(PhysINSSetViscosity(phys, 0.01));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));

  /* Create Seg SRK and wire operators */
  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetType(seg, SEGSRK));
  PetscCall(PhysSetUpSeg(phys, seg));

  /* Set solution and time parameters */
  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(FillIC(sol_dm, Y));
  PetscCall(SegSetSolution(seg, Y));
  PetscCall(SegSetTimeStepSize(seg, 0.01));
  PetscCall(SegSetMaxSteps(seg, 1));
  PetscCall(SegSetFromOptions(seg));
  PetscCall(SegSetUp(seg));

  /* Record norm before stepping */
  PetscCall(VecNorm(Y, NORM_2, &norm_before));
  PetscCheck(PetscIsNormalReal(norm_before), PETSC_COMM_WORLD, PETSC_ERR_FP, "Initial solution norm is not finite");

  /* Run one step */
  PetscCall(SegStep(seg));

  /* Verify solution is still finite after stepping */
  PetscCall(VecNorm(Y, NORM_2, &norm_after));
  PetscCheck(PetscIsNormalReal(norm_after), PETSC_COMM_WORLD, PETSC_ERR_FP, "Solution norm is not finite after step");

  /* Cleanup */
  PetscCall(VecDestroy(&Y));
  PetscCall(SegDestroy(&seg));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: ars111
    nsize: 1
    args: -seg_srk_type ars111
    output_file: output/empty.out

  test:
    suffix: ars121
    nsize: 1
    args: -seg_srk_type ars121
    output_file: output/empty.out

  test:
    suffix: ars222
    nsize: 1
    args: -seg_srk_type ars222
    output_file: output/empty.out

  test:
    suffix: ars232
    nsize: 1
    args: -seg_srk_type ars232
    output_file: output/empty.out

  test:
    suffix: ars343
    nsize: 1
    args: -seg_srk_type ars343
    output_file: output/empty.out

  test:
    suffix: ars443
    nsize: 1
    args: -seg_srk_type ars443
    output_file: output/empty.out

  test:
    suffix: ark324l2sa
    nsize: 1
    args: -seg_srk_type ark324l2sa
    output_file: output/empty.out

  test:
    suffix: ark436l2sa
    nsize: 1
    args: -seg_srk_type ark436l2sa
    output_file: output/empty.out

  test:
    suffix: ark548l2sa
    nsize: 1
    args: -seg_srk_type ark548l2sa
    output_file: output/empty.out

  test:
    suffix: mark324l2sa
    nsize: 1
    args: -seg_srk_type mark324l2sa
    output_file: output/empty.out

  test:
    suffix: mars343
    nsize: 1
    args: -seg_srk_type mars343
    output_file: output/empty.out

  test:
    suffix: bhr553
    nsize: 1
    args: -seg_srk_type bhr553
    output_file: output/empty.out

TEST*/
