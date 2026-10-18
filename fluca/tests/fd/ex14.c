#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaFDGetOperatorPattern: the structural pattern survives zero coefficients\n"
                           "Options:\n"
                           "  -test_case <constant|vector> : scale the operator by the constant 0 or by a zero vector\n";

/* Preallocate an AIJ matrix from what fill inserted into a MATPREALLOCATOR */
static PetscErrorCode CreateFromPreallocator_Private(DM dm, FlucaFD fd, PetscBool pattern, Mat *A)
{
  Mat                    prealloc;
  ISLocalToGlobalMapping ltog;
  PetscInt               m;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetEntries(dm, &m));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &prealloc));
  PetscCall(MatSetType(prealloc, MATPREALLOCATOR));
  PetscCall(MatSetSizes(prealloc, m, m, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetLocalToGlobalMapping(prealloc, ltog, ltog));
  PetscCall(MatSetUp(prealloc));
  if (pattern) PetscCall(FlucaFDGetOperatorPattern(fd, dm, dm, prealloc));
  else PetscCall(FlucaFDGetOperator(fd, dm, dm, prealloc));
  PetscCall(MatAssemblyBegin(prealloc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(prealloc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, m, m, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*A, MATAIJ));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltog, ltog));
  PetscCall(MatPreallocatorPreallocate(prealloc, PETSC_TRUE, *A));
  PetscCall(MatDestroy(&prealloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Number of stored entries in the row of element (i, j) */
static PetscErrorCode RowLength_Private(DM dm, Mat A, PetscInt i, PetscInt j, PetscInt *n)
{
  DMStagStencil          row = {DMSTAG_ELEMENT, i, j, 0, 0};
  ISLocalToGlobalMapping ltog;
  PetscInt               ir, gr, ncols;

  PetscFunctionBeginUser;
  PetscCall(DMStagStencilToIndexLocal(dm, 2, 1, &row, &ir));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(ISLocalToGlobalMappingApply(ltog, 1, &ir, &gr));
  PetscCall(MatGetRow(A, gr, &ncols, NULL, NULL));
  *n = ncols;
  PetscCall(MatRestoreRow(A, gr, &ncols, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  Mesh      mesh;
  FlucaFD   d2, scaled;
  Vec       zero;
  Mat       Avalue, Apattern;
  char      test_case[16] = "constant";
  PetscBool isvector;
  PetscInt  nv, np, r;
  PetscInt  rows[3][2] = {
    {0, 0},
    {0, 2},
    {2, 2}
  };
  const char *labels[3] = {"corner", "x wall", "interior"};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-test_case", test_case, sizeof(test_case), NULL));
  PetscCall(PetscStrcmp(test_case, "vector", &isvector));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetUp(mesh));

  /* d2/dx2 without BCs: one-sided at the walls; its stencil has 3 (interior) or 4 (wall) points */
  PetscCall(FlucaFDDerivativeCreate(mesh, FLUCAFD_X, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &d2));
  PetscCall(FlucaFDSetUp(d2));
  PetscCall(DMCreateGlobalVector(dm, &zero));
  PetscCall(VecZeroEntries(zero));
  if (isvector) PetscCall(FlucaFDScaleCreateVector(d2, zero, 0, &scaled));
  else PetscCall(FlucaFDScaleCreateConstant(d2, 0., &scaled));
  PetscCall(FlucaFDSetUp(scaled));

  PetscCall(CreateFromPreallocator_Private(dm, scaled, PETSC_FALSE, &Avalue));
  PetscCall(CreateFromPreallocator_Private(dm, scaled, PETSC_TRUE, &Apattern));
  for (r = 0; r < 3; ++r) {
    PetscCall(RowLength_Private(dm, Avalue, rows[r][0], rows[r][1], &nv));
    PetscCall(RowLength_Private(dm, Apattern, rows[r][0], rows[r][1], &np));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s (%" PetscInt_FMT ", %" PetscInt_FMT "): operator %" PetscInt_FMT ", pattern %" PetscInt_FMT "\n", labels[r], rows[r][0], rows[r][1], nv, np));
  }

  PetscCall(MatDestroy(&Apattern));
  PetscCall(MatDestroy(&Avalue));
  PetscCall(VecDestroy(&zero));
  PetscCall(FlucaFDDestroy(&scaled));
  PetscCall(FlucaFDDestroy(&d2));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: constant
    nsize: 1
    args: -test_case constant

  test:
    suffix: vector
    nsize: 1
    args: -test_case vector

TEST*/
