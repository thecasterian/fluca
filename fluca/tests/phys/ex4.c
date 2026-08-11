#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test that the discrete gradient and divergence blocks are negative\n"
                           "transposes on interior rows: D = -B, where B^T is the (0,1) block and\n"
                           "D the (1,0) block of the Jacobian. This is discrete integration by\n"
                           "parts; it pins the density out of D, so it fails if the divergence\n"
                           "operator carries a rho factor. Only interior rows are compared: on\n"
                           "boundary rows the velocity Dirichlet conditions imposed on D and the\n"
                           "pressure Neumann conditions imposed on B^T are not adjoint.\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : Grid cells per direction\n"
                           "  -stag_boundary_type_x/y <type>         : periodic leaves no boundary rows\n"
                           "  -phys_ins_density <real>               : Density; use a non-unit value\n";

static PetscErrorCode BCVelocityZero(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM             dm, sol_dm;
  Phys           phys;
  Mat            J, Gblk, Dblk, DblkT;
  Vec            Y, Ydot;
  IS             is_vel, is_p;
  PhysINSBC      bc;
  DMBoundaryType bndx, bndy;
  PetscBool      periodic;
  PetscInt       dim = 2, Nx, Ny, e, c, f, n, nv, np, *vi, *pi;
  PetscReal      nrm_g, nrm_diff;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  PetscCall(DMStagGetBoundaryTypes(dm, &bndx, &bndy, NULL));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, NULL));
  periodic = (PetscBool)(bndx == DM_BOUNDARY_PERIODIC && bndy == DM_BOUNDARY_PERIODIC);
  PetscCheck(periodic || (bndx != DM_BOUNDARY_PERIODIC && bndy != DM_BOUNDARY_PERIODIC), PETSC_COMM_WORLD, PETSC_ERR_SUP, "Mixed periodicity is not covered by this test");

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  if (!periodic) {
    bc.type       = PHYS_INS_BC_VELOCITY;
    bc.fn         = BCVelocityZero;
    bc.ctx        = NULL;
    bc.fn_dot     = NULL;
    bc.fn_dot_ctx = NULL;
    for (f = 0; f < 2 * dim; f++) PetscCall(PhysINSSetBoundaryCondition(phys, f, bc));
  }
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));

  /* The implicit residual is linear, so the Jacobian does not depend on the state */
  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(DMCreateGlobalVector(sol_dm, &Ydot));
  PetscCall(VecSet(Y, 1.));
  PetscCall(VecSet(Ydot, 1.));

  /* The stabilization stencil is wider than the DM stencil width, and the velocity
     Dirichlet stencils widen further on boundary rows, so let PETSc grow the pattern */
  PetscCall(DMCreateMatrix(sol_dm, &J));
  PetscCall(MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(PhysComputeIJacobian(phys, 0., Y, Ydot, 1., J, J));

  /* Index sets for the interior elements only. Elements are numbered row-major with
     dim + 1 contiguous DOFs each (velocity components, then pressure); this holds for
     the single-rank runs these tests use. */
  PetscCall(VecGetSize(Y, &n));
  PetscCheck(n == Nx * Ny * (dim + 1), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected DOF layout");
  PetscCall(PetscMalloc1(Nx * Ny * dim, &vi));
  PetscCall(PetscMalloc1(Nx * Ny, &pi));
  nv = np = 0;
  for (e = 0; e < Nx * Ny; e++) {
    PetscInt ix = e % Nx, iy = e / Nx;

    if (!periodic && (ix == 0 || ix == Nx - 1 || iy == 0 || iy == Ny - 1)) continue;
    for (c = 0; c < dim; c++) vi[nv++] = e * (dim + 1) + c;
    pi[np++] = e * (dim + 1) + dim;
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nv, vi, PETSC_USE_POINTER, &is_vel));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, np, pi, PETSC_USE_POINTER, &is_p));

  PetscCall(MatCreateSubMatrix(J, is_vel, is_p, MAT_INITIAL_MATRIX, &Gblk));
  PetscCall(MatCreateSubMatrix(J, is_p, is_vel, MAT_INITIAL_MATRIX, &Dblk));
  PetscCall(MatTranspose(Dblk, MAT_INITIAL_MATRIX, &DblkT));

  /* DblkT <- B^T + D^T, which vanishes exactly when D = -B */
  PetscCall(MatNorm(Gblk, NORM_INFINITY, &nrm_g));
  PetscCall(MatAXPY(DblkT, 1., Gblk, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(DblkT, NORM_INFINITY, &nrm_diff));

  PetscCheck(nrm_g > 0., PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Gradient block is empty; the test would pass vacuously");
  PetscCheck(nrm_diff <= 1e-12 * nrm_g, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "D != -B on interior rows: ||B^T + D^T||/||B^T|| = %g", (double)(nrm_diff / nrm_g));

  PetscCall(MatDestroy(&DblkT));
  PetscCall(MatDestroy(&Dblk));
  PetscCall(MatDestroy(&Gblk));
  PetscCall(ISDestroy(&is_p));
  PetscCall(ISDestroy(&is_vel));
  PetscCall(PetscFree(pi));
  PetscCall(PetscFree(vi));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&Ydot));
  PetscCall(VecDestroy(&Y));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));

  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: wall
    nsize: 1
    args: -stag_grid_x 8 -stag_grid_y 8 -phys_ins_density 2.5
    output_file: output/empty.out

  test:
    suffix: periodic
    nsize: 1
    args: -stag_grid_x 8 -stag_grid_y 8 -stag_boundary_type_x periodic -stag_boundary_type_y periodic -phys_ins_density 7.3
    output_file: output/empty.out

  test:
    suffix: wall_unit_density
    nsize: 1
    args: -stag_grid_x 6 -stag_grid_y 10 -phys_ins_density 1.0
    output_file: output/empty.out

TEST*/
