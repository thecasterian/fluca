#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaFDGetOperatorPattern: the structural pattern survives zero coefficients\n"
                           "Options:\n"
                           "  -test_case <constant|vector|convection3d> : scale a 2D operator by the constant 0 or by a zero vector,\n"
                           "                                              or compare the pattern of a 3D convection-like sum with its values\n";

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

/* Number of stored entries in the row of element (i, j, k) (k is ignored in 2D). Valid only at nsize 1 */
static PetscErrorCode RowLength_Private(DM dm, Mat A, PetscInt i, PetscInt j, PetscInt k, PetscInt *n)
{
  DMStagStencil          row = {DMSTAG_ELEMENT, i, j, k, 0};
  PetscInt               dim;
  ISLocalToGlobalMapping ltog;
  PetscInt               ir, gr, ncols;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagStencilToIndexLocal(dm, dim, 1, &row, &ir));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(ISLocalToGlobalMappingApply(ltog, 1, &ir, &gr));
  PetscCall(MatGetRow(A, gr, &ncols, NULL, NULL));
  *n = ncols;
  PetscCall(MatRestoreRow(A, gr, &ncols, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A 3D sum of six scaled terms, as in the NS convection: d/dx_e (interp_e(ubar) interp_e) for e = x, y, z, each
   term scaled by a vector (so the terms cannot merge) and by a second vector. Its raw stencil in pattern mode,
   with the zero-weight points of the one-sided corner stencils kept, has 36 points. */
static PetscErrorCode TestConvection3d_Private(void)
{
  DM                          dm;
  Mesh                        mesh;
  Vec                         ones;
  FlucaFD                     terms[6], sum;
  Mat                         Avalue, Apattern;
  MatInfo                     info;
  PetscReal                   nz_value, nz_pattern, nz_filled;
  PetscInt                    e, nv, np;
  const DMStagStencilLocation face_loc[3] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

  PetscFunctionBeginUser;
  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, 4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetUp(mesh));
  PetscCall(DMCreateGlobalVector(dm, &ones));
  PetscCall(VecSet(ones, 1.));

  for (e = 0; e < 3; ++e) {
    FlucaFD interp_u, interp_v, outer, scaled_u, scaled_v;

    PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, 0, face_loc[e], 0, &interp_u));
    PetscCall(FlucaFDSetUp(interp_u));
    PetscCall(FlucaFDScaleCreateVector(interp_u, ones, 0, &scaled_u));
    PetscCall(FlucaFDSetUp(scaled_u));
    PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 0, 2, DMSTAG_ELEMENT, 0, face_loc[e], 0, &interp_v));
    PetscCall(FlucaFDSetUp(interp_v));
    PetscCall(FlucaFDScaleCreateVector(interp_v, ones, 0, &scaled_v));
    PetscCall(FlucaFDSetUp(scaled_v));
    PetscCall(FlucaFDDerivativeCreate(mesh, (FlucaFDDirection)e, 1, 2, face_loc[e], 0, DMSTAG_ELEMENT, 0, &outer));
    PetscCall(FlucaFDSetUp(outer));
    PetscCall(FlucaFDCompositionCreate(scaled_u, outer, &terms[2 * e]));
    PetscCall(FlucaFDSetUp(terms[2 * e]));
    PetscCall(FlucaFDCompositionCreate(scaled_v, outer, &terms[2 * e + 1]));
    PetscCall(FlucaFDSetUp(terms[2 * e + 1]));
    PetscCall(FlucaFDDestroy(&outer));
    PetscCall(FlucaFDDestroy(&scaled_v));
    PetscCall(FlucaFDDestroy(&interp_v));
    PetscCall(FlucaFDDestroy(&scaled_u));
    PetscCall(FlucaFDDestroy(&interp_u));
  }
  PetscCall(FlucaFDSumCreate(6, terms, &sum));
  PetscCall(FlucaFDSetUp(sum));

  PetscCall(CreateFromPreallocator_Private(dm, sum, PETSC_FALSE, &Avalue));
  PetscCall(CreateFromPreallocator_Private(dm, sum, PETSC_TRUE, &Apattern));
  PetscCall(MatGetInfo(Avalue, MAT_GLOBAL_SUM, &info));
  nz_value = info.nz_used;
  PetscCall(MatGetInfo(Apattern, MAT_GLOBAL_SUM, &info));
  nz_pattern = info.nz_used;
  PetscCall(RowLength_Private(dm, Avalue, 0, 0, 0, &nv));
  PetscCall(RowLength_Private(dm, Apattern, 0, 0, 0, &np));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "corner (0, 0, 0): operator %" PetscInt_FMT ", pattern %" PetscInt_FMT "\n", nv, np));
  PetscCall(RowLength_Private(dm, Avalue, 1, 1, 1, &nv));
  PetscCall(RowLength_Private(dm, Apattern, 1, 1, 1, &np));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "interior (1, 1, 1): operator %" PetscInt_FMT ", pattern %" PetscInt_FMT "\n", nv, np));

  /* Every value must land inside the pattern: filling the values into the pattern matrix adds no entry */
  PetscCall(MatSetOption(Apattern, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(FlucaFDGetOperator(sum, dm, dm, Apattern));
  PetscCall(MatAssemblyBegin(Apattern, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Apattern, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetInfo(Apattern, MAT_GLOBAL_SUM, &info));
  nz_filled = info.nz_used;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "stored entries: operator %.0f, pattern %.0f, pattern after inserting the operator %.0f\n", (double)nz_value, (double)nz_pattern, (double)nz_filled));

  PetscCall(MatDestroy(&Apattern));
  PetscCall(MatDestroy(&Avalue));
  PetscCall(FlucaFDDestroy(&sum));
  for (e = 0; e < 6; ++e) PetscCall(FlucaFDDestroy(&terms[e]));
  PetscCall(VecDestroy(&ones));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
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
  PetscBool isvector, isconv3d;
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
  PetscCall(PetscStrcmp(test_case, "convection3d", &isconv3d));
  if (isconv3d) {
    PetscCall(TestConvection3d_Private());
    PetscCall(FlucaFinalize());
    return 0;
  }

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
    PetscCall(RowLength_Private(dm, Avalue, rows[r][0], rows[r][1], 0, &nv));
    PetscCall(RowLength_Private(dm, Apattern, rows[r][0], rows[r][1], 0, &np));
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

  test:
    suffix: convection3d
    nsize: 1
    args: -test_case convection3d

TEST*/
