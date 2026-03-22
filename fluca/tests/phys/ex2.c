#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "Test IFunction and RHSFunction with 2D Taylor-Green vortex\n"
                           "Exact solution on periodic [0, 2*pi]^2:\n"
                           "  u = -cos(x)*sin(y)*exp(-2*nu*t)\n"
                           "  v =  sin(x)*cos(y)*exp(-2*nu*t)\n"
                           "  p = -(cos(2x)+cos(2y))/4 * exp(-4*nu*t)\n";

/* Fill solution vector with exact TGV at t=0 */
static PetscErrorCode FillExactSolution(DM sol_dm, PetscInt dim, Vec Y)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil stencils[3];

      vals[0] = -PetscCosReal(x) * PetscSinReal(y);
      vals[1] = PetscSinReal(x) * PetscCosReal(y);
      vals[2] = -0.25 * (PetscCosReal(2 * x) + PetscCosReal(2 * y));

      for (PetscInt c = 0; c < dim + 1; c++) {
        stencils[c].i   = i;
        stencils[c].j   = j;
        stencils[c].k   = 0;
        stencils[c].loc = DMSTAG_ELEMENT;
        stencils[c].c   = c;
      }
      PetscCall(DMStagVecSetValuesStencil(sol_dm, Y, dim + 1, stencils, vals, INSERT_VALUES));
    }
  }

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fill y_dot with exact time derivative of TGV at t=0.
   du/dt = 2*nu*cos(x)*sin(y), dv/dt = -2*nu*sin(x)*cos(y),
   dp/dt = nu*(cos(2x)+cos(2y)) */
static PetscErrorCode FillExactYDot(DM sol_dm, PetscInt dim, PetscReal nu, Vec Ydot)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem;
  PetscInt            i, j;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil stencils[3];

      vals[0] = 2 * nu * PetscCosReal(x) * PetscSinReal(y);
      vals[1] = -2 * nu * PetscSinReal(x) * PetscCosReal(y);
      vals[2] = nu * (PetscCosReal(2 * x) + PetscCosReal(2 * y));

      for (PetscInt c = 0; c < dim + 1; c++) {
        stencils[c].i   = i;
        stencils[c].j   = j;
        stencils[c].k   = 0;
        stencils[c].loc = DMSTAG_ELEMENT;
        stencils[c].c   = c;
      }
      PetscCall(DMStagVecSetValuesStencil(sol_dm, Ydot, dim + 1, stencils, vals, INSERT_VALUES));
    }
  }

  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Ydot));
  PetscCall(VecAssemblyEnd(Ydot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  Vec       Y, Ydot, F, G;
  PetscInt  N   = 8;
  PetscReal rho = 1., mu = 1., nu, norm, h, tol;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));

  /* Create 2D periodic base DMStag */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, N, N, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2 * PETSC_PI, 0., 2 * PETSC_PI, 0., 0.));

  /* Create Phys INS (periodic — no explicit BCs needed) */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysINSSetDensity(phys, rho));
  PetscCall(PhysINSSetViscosity(phys, mu));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  nu = mu / rho;

  /* Create and fill solution and time derivative vectors */
  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(DMCreateGlobalVector(sol_dm, &Ydot));
  PetscCall(DMCreateGlobalVector(sol_dm, &F));
  PetscCall(DMCreateGlobalVector(sol_dm, &G));

  PetscCall(FillExactSolution(sol_dm, 2, Y));
  PetscCall(FillExactYDot(sol_dm, 2, nu, Ydot));

  /* Compute IFunction and RHSFunction at t=0.
     With alpha=0 (no TS), IFunction momentum = rho*u_dot - mu*L*u + grad(p).
     For the exact TGV solution this equals -(u.grad)u analytically.
     RHSFunction momentum = -C(u) = -(u.grad)u analytically.
     So F - G should be zero (up to discretization error between centered FD and TVD). */
  PetscCall(PhysComputeIFunction(phys, 0., Y, Ydot, F));
  PetscCall(PhysComputeRHSFunction(phys, 0., Y, G));

  /* F - G should be zero for the exact solution (up to discretization error).
     On 8x8 this is O(h^2) since centered FD and TVD have different truncation. */
  PetscCall(VecAXPY(F, -1., G));
  PetscCall(VecNorm(F, NORM_INFINITY, &norm));
  /* The TVD limiter degrades to first order at extrema, so the infinity-norm
     error between centered-FD IFunction and TVD RHSFunction is O(h). */
  h   = 2 * PETSC_PI / N;
  tol = 2 * h;
  PetscCheck(norm < tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "IFunction - RHSFunction norm %g exceeds tolerance %g (h = %g)", (double)norm, (double)tol, (double)h);

  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&F));
  PetscCall(VecDestroy(&Ydot));
  PetscCall(VecDestroy(&Y));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: tgv_8
    nsize: 1
    args: -N 8
    output_file: output/empty.out

  test:
    suffix: tgv_16
    nsize: 1
    args: -N 16
    output_file: output/empty.out

  test:
    suffix: tgv_32
    nsize: 1
    args: -N 32
    output_file: output/empty.out

  test:
    suffix: tgv_64
    nsize: 1
    args: -N 64
    output_file: output/empty.out

  test:
    suffix: tgv_128
    nsize: 1
    args: -N 128
    output_file: output/empty.out

TEST*/
