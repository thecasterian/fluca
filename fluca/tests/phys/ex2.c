#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Verify PhysComputeIFunction vanishes near the manufactured exact solution\n"
                           "Exact solution: Taylor-Green-like (mu=rho=1)\n"
                           "  u = -cos(pi*x)*sin(pi*y)\n"
                           "  v = sin(pi*x)*cos(pi*y)\n"
                           "  p = -(cos(2*pi*x) + cos(2*pi*y))/4\n"
                           "Full NS body force: f = rho*(u.grad)u - mu*nabla^2(u) + grad(p)\n"
                           "  Convection and pressure gradient cancel for this solution.\n"
                           "  f_x = -2*pi^2*cos(pi*x)*sin(pi*y)\n"
                           "  f_y =  2*pi^2*sin(pi*x)*cos(pi*y)\n"
                           "Options:\n"
                           "  -stag_grid_x <int> : Grid size in x (default: 16)\n"
                           "  -stag_grid_y <int> : Grid size in y (default: 16)\n";

/* Exact solution: u[0]=u, u[1]=v, u[2]=p */
static PetscErrorCode ExactSolution(PetscInt dim, const PetscReal x[], PetscScalar u[])
{
  PetscReal pi = PETSC_PI;

  PetscFunctionBeginUser;
  u[0] = -PetscCosReal(pi * x[0]) * PetscSinReal(pi * x[1]);
  u[1] = PetscSinReal(pi * x[0]) * PetscCosReal(pi * x[1]);
  u[2] = -(PetscCosReal(2.0 * pi * x[0]) + PetscCosReal(2.0 * pi * x[1])) / 4.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* BC callback: exact velocity component at boundary coordinates */
static PetscErrorCode BCVelocity(PetscInt dim, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscScalar u[3];

  PetscFunctionBeginUser;
  PetscCall(ExactSolution(dim, x, u));
  *val = u[comp];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Full NS body force: f = rho*(u.grad)u - mu*nabla^2(u) + grad(p) with mu=rho=1.
   For this Taylor-Green solution, the convection term rho*(u.grad)u and the pressure gradient
   grad(p) cancel exactly, leaving only the viscous contribution. */
static PetscErrorCode BodyForce(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx)
{
  PetscReal pi = PETSC_PI;
  PetscReal cx = PetscCosReal(pi * x[0]);
  PetscReal cy = PetscCosReal(pi * x[1]), sy = PetscSinReal(pi * x[1]);
  PetscReal sx = PetscSinReal(pi * x[0]);

  PetscFunctionBeginUser;
  f[0] = -2.0 * pi * pi * cx * sy;
  f[1] = 2.0 * pi * pi * sx * cy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                  dm, sol_dm;
  Phys                phys;
  Vec                 x_exact, u_t, F;
  PetscInt            Nx, Ny;
  PetscInt            f;
  PhysINSBC           bc;
  PetscReal           fnorm, tol;
  const PetscScalar **arrc[2] = {NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_e;
  PetscInt            i, j, d;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create 2D base DMStag; size overridden via -stag_grid_x/-stag_grid_y */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 16, 16, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, NULL));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));

  /* Create and configure Phys INS */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysSetBodyForce(phys, BodyForce, NULL));

  /* Velocity Dirichlet BCs on all 4 faces */
  bc.type = PHYS_INS_BC_VELOCITY;
  bc.fn   = BCVelocity;
  bc.ctx  = NULL;
  for (f = 0; f < 4; f++) PetscCall(PhysINSSetBoundaryCondition(phys, f, bc));

  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));

  /* Create solution and residual vectors */
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMCreateGlobalVector(sol_dm, &x_exact));
  PetscCall(DMCreateGlobalVector(sol_dm, &u_t));
  PetscCall(DMCreateGlobalVector(sol_dm, &F));
  PetscCall(VecZeroEntries(u_t));

  /* Fill x_exact with the manufactured solution at cell centers */
  PetscCall(DMStagGetProductCoordinateLocationSlot(sol_dm, DMSTAG_ELEMENT, &slot_e));
  PetscCall(DMStagGetProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], NULL));
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     coords[2];
      PetscScalar   exact[3];
      DMStagStencil st;

      coords[0] = PetscRealPart(arrc[0][i][slot_e]);
      coords[1] = PetscRealPart(arrc[1][j][slot_e]);
      PetscCall(ExactSolution(2, coords, exact));

      st.i   = i;
      st.j   = j;
      st.k   = 0;
      st.loc = DMSTAG_ELEMENT;
      for (d = 0; d < 3; d++) {
        st.c = d;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, x_exact, 1, &st, &exact[d], INSERT_VALUES));
      }
    }
  }
  PetscCall(VecAssemblyBegin(x_exact));
  PetscCall(VecAssemblyEnd(x_exact));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(sol_dm, &arrc[0], &arrc[1], NULL));

  /* Evaluate IFunction at exact solution with zero time derivative.
     The TVD limiter degenerates to first-order at velocity extrema, so
     the inf-norm error scales as O(h) = O(1/N) rather than O(h^2). */
  PetscCall(PhysComputeIFunction(phys, 0.0, x_exact, u_t, F));
  PetscCall(VecNorm(F, NORM_INFINITY, &fnorm));
  tol = 4.0 / (PetscReal)PetscMax(Nx, Ny);
  PetscCheck(fnorm < tol, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "IFunction norm too large: %g (tol %g)", (double)fnorm, (double)tol);

  /* Cleanup */
  PetscCall(VecDestroy(&x_exact));
  PetscCall(VecDestroy(&u_t));
  PetscCall(VecDestroy(&F));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  test:
    suffix: 16
    nsize: 1
    args: -phys_ins_flucafd_limiter superbee
    output_file: output/empty.out

  test:
    suffix: 32
    nsize: 1
    args: -stag_grid_x 32 -stag_grid_y 32 -phys_ins_flucafd_limiter superbee
    output_file: output/empty.out

  test:
    suffix: 64
    nsize: 1
    args: -stag_grid_x 64 -stag_grid_y 64 -phys_ins_flucafd_limiter superbee
    output_file: output/empty.out

TEST*/
