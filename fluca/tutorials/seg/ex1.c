#include <flucaphys.h>
#include <flucaseg.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "2D Taylor-Green vortex with Seg SRK\n"
                           "Exact solution on periodic [0, 2*pi]^2:\n"
                           "  u = -cos(x)*sin(y)*exp(-2*nu*t)\n"
                           "  v =  sin(x)*cos(y)*exp(-2*nu*t)\n"
                           "  p = -(cos(2x)+cos(2y))/4 * exp(-4*nu*t)\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : Grid cells per direction (default: 32)\n"
                           "  -seg_dt <real>       : Time step size (default: 0.02)\n"
                           "  -seg_max_time <real> : Final time (default: 0.1)\n"
                           "  -rho <real>          : Density (default: 1.)\n"
                           "  -mu <real>           : Dynamic viscosity (default: 0.01)\n";

static PetscErrorCode FillExactSolution(DM dm, PetscReal nu, PetscReal t, Vec Y)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem, i, j, c;
  PetscReal           decay_v, decay_p;

  PetscFunctionBegin;
  decay_v = PetscExpReal(-2. * nu * t);
  decay_p = PetscExpReal(-4. * nu * t);

  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil stencils[3];

      vals[0] = -PetscCosReal(x) * PetscSinReal(y) * decay_v;
      vals[1] = PetscSinReal(x) * PetscCosReal(y) * decay_v;
      vals[2] = -0.25 * (PetscCosReal(2. * x) + PetscCosReal(2. * y)) * decay_p;

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

static PetscErrorCode ComputeL2Error(DM dm, PetscReal nu, PetscReal t, Vec Y, PetscReal *err_u, PetscReal *err_v, PetscReal *err_p)
{
  Vec           Y_exact, diff, sub;
  PetscReal     h2, norm;
  PetscInt      N;
  DMStagStencil stencil;
  IS            is;

  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(dm, &Y_exact));
  PetscCall(FillExactSolution(dm, nu, t, Y_exact));
  PetscCall(VecDuplicate(Y, &diff));
  PetscCall(VecCopy(Y, diff));
  PetscCall(VecAXPY(diff, -1., Y_exact));

  PetscCall(DMStagGetGlobalSizes(dm, &N, NULL, NULL));
  h2 = (2. * PETSC_PI / N) * (2. * PETSC_PI / N);

  stencil.i   = 0;
  stencil.j   = 0;
  stencil.k   = 0;
  stencil.loc = DMSTAG_ELEMENT;

  /* u error */
  stencil.c = 0;
  PetscCall(DMStagCreateISFromStencils(dm, 1, &stencil, &is));
  PetscCall(VecGetSubVector(diff, is, &sub));
  PetscCall(VecNorm(sub, NORM_2, &norm));
  *err_u = norm * PetscSqrtReal(h2);
  PetscCall(VecRestoreSubVector(diff, is, &sub));
  PetscCall(ISDestroy(&is));

  /* v error */
  stencil.c = 1;
  PetscCall(DMStagCreateISFromStencils(dm, 1, &stencil, &is));
  PetscCall(VecGetSubVector(diff, is, &sub));
  PetscCall(VecNorm(sub, NORM_2, &norm));
  *err_v = norm * PetscSqrtReal(h2);
  PetscCall(VecRestoreSubVector(diff, is, &sub));
  PetscCall(ISDestroy(&is));

  /* p error (subtract mean) */
  stencil.c = 2;
  PetscCall(DMStagCreateISFromStencils(dm, 1, &stencil, &is));
  PetscCall(VecGetSubVector(diff, is, &sub));
  {
    PetscScalar mean;
    PetscInt    np;

    PetscCall(VecGetSize(sub, &np));
    PetscCall(VecSum(sub, &mean));
    mean /= np;
    PetscCall(VecShift(sub, -mean));
  }
  PetscCall(VecNorm(sub, NORM_2, &norm));
  *err_p = norm * PetscSqrtReal(h2);
  PetscCall(VecRestoreSubVector(diff, is, &sub));
  PetscCall(ISDestroy(&is));

  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&Y_exact));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  Seg       seg;
  Vec       Y;
  PetscReal rho = 1., mu = 0.01, nu;
  PetscReal err_u, err_v, err_p, t_final, dt;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho", &rho, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &mu, NULL));
  nu = mu / rho;

  /* Create 2D periodic base DMStag.
     Grid size is set via -stag_grid_x / -stag_grid_y through DMSetFromOptions. */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, 32, 32, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));

  /* Create PhysINS */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysINSSetDensity(phys, rho));
  PetscCall(PhysINSSetViscosity(phys, mu));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));

  /* Create Seg SRK.
     Time step and max time are set via -seg_dt / -seg_max_time through SegSetFromOptions. */
  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetType(seg, SEGSRK));
  PetscCall(PhysSetUpSeg(phys, seg));

  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(FillExactSolution(sol_dm, nu, 0., Y));
  PetscCall(SegSetSolution(seg, Y));
  PetscCall(SegSetTimeStepSize(seg, 0.02));
  PetscCall(SegSetMaxTime(seg, 0.1));
  PetscCall(SegSetFromOptions(seg));
  PetscCall(SegSetUp(seg));

  PetscCall(SegSolve(seg));
  PetscCall(SegGetTime(seg, &t_final));
  PetscCall(SegGetTimeStepSize(seg, &dt));

  PetscCall(ComputeL2Error(sol_dm, nu, t_final, Y, &err_u, &err_v, &err_p));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dt = %.6e  err_u = %.6e  err_v = %.6e  err_p = %.6e\n", (double)dt, (double)err_u, (double)err_v, (double)err_p));

  PetscCall(VecDestroy(&Y));
  PetscCall(SegDestroy(&seg));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: tgv
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -seg_dt 0.02 -seg_max_time 0.1

TEST*/
