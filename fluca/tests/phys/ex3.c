#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Verify convection operator with manufactured solution\n"
                           "Exact solution: u = (y, -x), p = 0 on [0,1]x[0,1]\n"
                           "  div(u) = 0\n"
                           "  (u.grad)u = (-x, -y)\n"
                           "  nabla^2(u) = (0, 0)  (linear field)\n"
                           "Body force: f = (-x, -y) to balance full NS residual\n"
                           "Options:\n"
                           "  -stag_grid_x <int> : Grid size in x (default: 8)\n"
                           "  -stag_grid_y <int> : Grid size in y (default: 8)\n";

/* Exact solution: u[0]=y, u[1]=-x, u[2]=p=0 */
static PetscErrorCode ExactSolution(PetscInt dim, const PetscReal x[], PetscScalar u[])
{
  PetscFunctionBeginUser;
  u[0] = x[1];
  u[1] = -x[0];
  u[2] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* BC callback: exact velocity component at boundary */
static PetscErrorCode BCVelocity(PetscInt dim, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscScalar u[3];

  PetscFunctionBeginUser;
  PetscCall(ExactSolution(dim, x, u));
  *val = u[comp];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Full NS body force: f = (u.grad)u - mu*nabla^2(u) + (1/rho)*grad(p)
   For u=(y,-x), p=0, mu=rho=1: f = (-x, -y) + 0 + 0 = (-x, -y) */
static PetscErrorCode BodyForce(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx)
{
  PetscFunctionBeginUser;
  f[0] = -x[0];
  f[1] = -x[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                  dm, sol_dm;
  Phys                phys;
  Vec                 x_exact, u_t, F;
  PetscInt            f;
  PhysINSBC           bc;
  PetscReal           fnorm, tol;
  const PetscScalar **arrc[2] = {NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_e;
  PetscInt            i, j, d;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Create 2D base DMStag; size overridden via -stag_grid_x/-stag_grid_y */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
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
     The velocity field is linear, so TVD interpolation should be exact
     (no limiter truncation). Use a tight tolerance. */
  PetscCall(PhysComputeIFunction(phys, 0.0, x_exact, u_t, F));
  PetscCall(VecNorm(F, NORM_INFINITY, &fnorm));
  tol = 1e-10;
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
    suffix: 8
    nsize: 1
    args: -phys_ins_flucafd_limiter superbee
    output_file: output/empty.out

  test:
    suffix: 16
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -phys_ins_flucafd_limiter superbee
    output_file: output/empty.out

TEST*/
