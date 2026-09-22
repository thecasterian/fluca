#include "phystest.h"

static const char help[] = "Test the boundary terms of PhysComputeCouplingSystem with time-dependent wall velocity\n"
                           "Boundary-face rows must read U = u_b(t) . n; all other right-hand-side entries vanish.\n";

static PetscErrorCode WallVelocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = comp == 0 ? t * (1. + x[1]) : t * (2. + x[0]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Boundary-face row of M must be the unit row */
static PetscErrorCode CheckUnitRow(DM sol_dm, Mat M, DMStagStencil st)
{
  ISLocalToGlobalMapping ltog;
  PetscInt               il, ig, ncols, c;
  const PetscInt        *cols;
  const PetscScalar     *vals;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(sol_dm, &ltog));
  PetscCall(DMStagStencilToIndexLocal(sol_dm, 2, 1, &st, &il));
  PetscCall(ISLocalToGlobalMappingApply(ltog, 1, &il, &ig));
  PetscCall(MatGetRow(M, ig, &ncols, &cols, &vals));
  for (c = 0; c < ncols; ++c) {
    PetscScalar expect = cols[c] == ig ? 1. : 0.;

    PetscCheck(PetscAbsScalar(vals[c] - expect) == 0., PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Boundary face (%" PetscInt_FMT ", %" PetscInt_FMT ") row has entry %g at column %" PetscInt_FMT, st.i, st.j, (double)PetscRealPart(vals[c]), cols[c]);
  }
  PetscCall(MatRestoreRow(M, ig, &ncols, &cols, &vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                dm, sol_dm;
  Phys              phys;
  Mat               M;
  Vec               f, E;
  PhysFieldLocation loc;
  PetscInt          c_U, Nx, Ny, i, j, b;
  PetscReal         t = 0.5, hx, hy;
  DMStagStencil     st;
  PetscScalar       v;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 6, 5, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(PhysTestCreateLaminar(dm, 1., 1., WallVelocity, &phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, &loc, &c_U, NULL));
  PetscCall(DMStagGetGlobalSizes(sol_dm, &Nx, &Ny, NULL));
  hx = 1. / Nx;
  hy = 1. / Ny;

  PetscCall(PhysTestCreateSystem(phys, &M, &f));
  PetscCall(PhysComputeCouplingSystem(phys, t, 0.1, M, f));

  /* Expected right-hand side: u_b . n on boundary faces, zero elsewhere (single rank) */
  PetscCall(DMCreateGlobalVector(sol_dm, &E));
  PetscCall(VecZeroEntries(E));
  st.k = 0;
  st.c = c_U;
  for (j = 0; j < Ny; ++j) {
    for (b = 0; b < 2; ++b) {
      st.i   = b == 0 ? 0 : Nx;
      st.j   = j;
      st.loc = DMSTAG_LEFT;
      v      = t * (1. + (j + 0.5) * hy);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, E, 1, &st, &v, INSERT_VALUES));
    }
  }
  for (i = 0; i < Nx; ++i) {
    for (b = 0; b < 2; ++b) {
      st.i   = i;
      st.j   = b == 0 ? 0 : Ny;
      st.loc = DMSTAG_DOWN;
      v      = t * (2. + (i + 0.5) * hx);
      PetscCall(DMStagVecSetValuesStencil(sol_dm, E, 1, &st, &v, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(E));
  PetscCall(VecAssemblyEnd(E));
  PetscCall(PhysTestCheckField(phys, PHYS_FIELD_FACE_VELOCITY, f, E, 1e-14));
  PetscCall(PhysTestCheckField(phys, PHYS_FIELD_PRESSURE, f, E, 0.));
  PetscCall(PhysTestCheckField(phys, PHYS_FIELD_VELOCITY, f, E, 0.));

  /* Boundary-face rows of M are unit rows */
  for (j = 0; j < Ny; ++j) {
    for (b = 0; b < 2; ++b) {
      st.i   = b == 0 ? 0 : Nx;
      st.j   = j;
      st.loc = DMSTAG_LEFT;
      PetscCall(CheckUnitRow(sol_dm, M, st));
    }
  }
  for (i = 0; i < Nx; ++i) {
    for (b = 0; b < 2; ++b) {
      st.i   = i;
      st.j   = b == 0 ? 0 : Ny;
      st.loc = DMSTAG_DOWN;
      PetscCall(CheckUnitRow(sol_dm, M, st));
    }
  }

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&M));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: wall_velocity
    nsize: 1
    output_file: output/empty.out

TEST*/
