#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test the nonzero pattern of the NS Jacobian\n"
                           "Options:\n"
                           "  -dim <2|3> : dimension\n";

/* Valid only at nsize 1: MatGetRow() reads the row on the owning rank */
static PetscErrorCode RowLength_Private(DM dm, Mat A, DMStagStencil row, PetscInt *n)
{
  ISLocalToGlobalMapping ltog;
  PetscInt               dim, ir, gr, nc;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagStencilToIndexLocal(dm, dim, 1, &row, &ir));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(ISLocalToGlobalMappingApply(ltog, 1, &ir, &gr));
  PetscCall(MatGetRow(A, gr, &nc, NULL, NULL));
  *n = nc;
  PetscCall(MatRestoreRow(A, gr, &nc, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM            dm, sdm;
  Mesh          mesh;
  Phys          phys;
  NS            ns;
  SNES          snes;
  Mat           J;
  MatInfo       info;
  PhysBC        bc  = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL};
  PetscInt      dim = 2, d, r, n, c_vel, c_U, c_p, mid;
  DMStagStencil rows[6];
  const char   *labels[6] = {"u_x corner", "u_x wall", "u_x interior", "U_x boundary face", "U_x interior face", "p interior"};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  /* The grid DM carries one element dof; Phys lays out the solution fields on its own compatible DM */
  if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 6, 6, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  else PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 6, 6, 6, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(phys, mesh));
  for (d = 0; d < 2 * dim; ++d) PetscCall(PhysSetBoundaryCondition(phys, d, bc));
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));
  PetscCall(NSGetSNES(ns, &snes));
  PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));

  PetscCall(PhysGetSolutionDM(phys, &sdm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  mid = 3;
  for (r = 0; r < 6; ++r) {
    rows[r].i = mid;
    rows[r].j = mid;
    rows[r].k = dim == 3 ? mid : 0;
  }
  rows[0].loc = DMSTAG_ELEMENT, rows[0].c = c_vel, rows[0].i = 0, rows[0].j = 0, rows[0].k = 0;
  rows[1].loc = DMSTAG_ELEMENT, rows[1].c = c_vel, rows[1].i = 0;
  rows[2].loc = DMSTAG_ELEMENT, rows[2].c = c_vel;
  rows[3].loc = DMSTAG_LEFT, rows[3].c = c_U, rows[3].i = 0;
  rows[4].loc = DMSTAG_LEFT, rows[4].c = c_U;
  rows[5].loc = DMSTAG_ELEMENT, rows[5].c = c_p;
  for (r = 0; r < 6; ++r) {
    PetscCall(RowLength_Private(sdm, J, rows[r], &n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: %" PetscInt_FMT "\n", labels[r], n));
  }
  PetscCall(MatGetInfo(J, MAT_GLOBAL_SUM, &info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "total: %.0f\n", info.nz_used));

  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d
    nsize: 1
    args: -dim 2

  test:
    suffix: 3d
    nsize: 1
    args: -dim 3

TEST*/
