#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaFDGetOperatorPattern: the pattern survives zero scales\n"
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
   term scaled by a vector (so the terms cannot merge) and by a second vector. Its pattern covers every
   location a vector scale may fill; zero weights from the mesh are left out. */
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

/* A 1D second-order TVD operator (element phi -> face phi) taken with a positive mass flux. Its pattern keeps
   both upwind candidates of a face, so filling the values of the reversed flux adds no entry. */
static PetscErrorCode TestTVD_Private(void)
{
  DM                     input_dm, output_dm;
  Mesh                   mesh;
  Vec                    phi, mass_flux;
  FlucaFD                tvd;
  Mat                    prealloc, Avalue, Apattern;
  ISLocalToGlobalMapping rltog, cltog;
  MatInfo                info;
  PetscReal              nz_before, nz_after;
  PetscInt               m, n, nv, np, ir, gr, ncols;
  DMStagStencil          row = {DMSTAG_LEFT, 4, 0, 0, 0};

  PetscFunctionBeginUser;
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, &input_dm));
  PetscCall(DMSetUp(input_dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(input_dm, 0., 1., 0., 0., 0., 0.));
  PetscCall(MeshCartesianCreate(input_dm, &mesh));
  PetscCall(MeshSetUp(mesh));
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 1, 0, DMSTAG_STENCIL_BOX, 1, NULL, &output_dm));
  PetscCall(DMSetUp(output_dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(output_dm, 0., 1., 0., 0., 0., 0.));
  PetscCall(DMCreateGlobalVector(input_dm, &phi));
  PetscCall(VecGetSize(phi, &n));
  for (ir = 0; ir < n; ++ir) PetscCall(VecSetValue(phi, ir, (PetscScalar)(ir * ir), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(phi));
  PetscCall(VecAssemblyEnd(phi));
  PetscCall(DMCreateGlobalVector(output_dm, &mass_flux));
  PetscCall(VecSet(mass_flux, 1.));

  PetscCall(FlucaFDSecondOrderTVDCreate(mesh, FLUCAFD_X, 0, 0, &tvd));
  PetscCall(FlucaFDSetFromOptions(tvd));
  PetscCall(FlucaFDSetUp(tvd));
  PetscCall(FlucaFDSecondOrderTVDSetMassFlux(tvd, mass_flux, 0));
  PetscCall(FlucaFDSecondOrderTVDSetCurrentSolution(tvd, phi));

  PetscCall(DMStagGetEntries(output_dm, &m));
  PetscCall(DMStagGetEntries(input_dm, &n));
  PetscCall(DMGetLocalToGlobalMapping(output_dm, &rltog));
  PetscCall(DMGetLocalToGlobalMapping(input_dm, &cltog));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &prealloc));
  PetscCall(MatSetType(prealloc, MATPREALLOCATOR));
  PetscCall(MatSetSizes(prealloc, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetLocalToGlobalMapping(prealloc, rltog, cltog));
  PetscCall(MatSetUp(prealloc));
  PetscCall(FlucaFDGetOperatorPattern(tvd, input_dm, output_dm, prealloc));
  PetscCall(MatAssemblyBegin(prealloc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(prealloc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Apattern));
  PetscCall(MatSetSizes(Apattern, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(Apattern, MATAIJ));
  PetscCall(MatSetLocalToGlobalMapping(Apattern, rltog, cltog));
  PetscCall(MatPreallocatorPreallocate(prealloc, PETSC_TRUE, Apattern));
  PetscCall(MatDestroy(&prealloc));

  /* The values with the positive flux: only the upwind point is a grid point */
  PetscCall(MatDuplicate(Apattern, MAT_DO_NOT_COPY_VALUES, &Avalue));
  PetscCall(MatSetOption(Avalue, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(FlucaFDGetOperator(tvd, input_dm, output_dm, Avalue));
  PetscCall(MatAssemblyBegin(Avalue, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Avalue, MAT_FINAL_ASSEMBLY));
  PetscCall(DMStagStencilToIndexLocal(output_dm, 1, 1, &row, &ir));
  PetscCall(ISLocalToGlobalMappingApply(rltog, 1, &ir, &gr));
  PetscCall(MatGetRow(Avalue, gr, &ncols, NULL, NULL));
  nv = ncols;
  PetscCall(MatRestoreRow(Avalue, gr, &ncols, NULL, NULL));
  PetscCall(MatGetRow(Apattern, gr, &ncols, NULL, NULL));
  np = ncols;
  PetscCall(MatRestoreRow(Apattern, gr, &ncols, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "interior face 4, positive flux: operator %" PetscInt_FMT ", pattern %" PetscInt_FMT "\n", nv, np));

  /* Reverse the flux and fill the values into the pattern: no entry may be added */
  PetscCall(MatSetOption(Apattern, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatGetInfo(Apattern, MAT_GLOBAL_SUM, &info));
  nz_before = info.nz_used;
  PetscCall(VecScale(mass_flux, -1.));
  PetscCall(FlucaFDSecondOrderTVDSetMassFlux(tvd, mass_flux, 0));
  PetscCall(FlucaFDGetOperator(tvd, input_dm, output_dm, Apattern));
  PetscCall(MatAssemblyBegin(Apattern, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Apattern, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetInfo(Apattern, MAT_GLOBAL_SUM, &info));
  nz_after = info.nz_used;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "pattern entries: before %.0f, after inserting the reversed-flux operator %.0f\n", (double)nz_before, (double)nz_after));

  PetscCall(MatDestroy(&Avalue));
  PetscCall(MatDestroy(&Apattern));
  PetscCall(FlucaFDDestroy(&tvd));
  PetscCall(VecDestroy(&mass_flux));
  PetscCall(VecDestroy(&phi));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&output_dm));
  PetscCall(DMDestroy(&input_dm));
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
  PetscBool isvector, isconv3d, istvd;
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
  PetscCall(PetscStrcmp(test_case, "tvd", &istvd));
  if (istvd) {
    PetscCall(TestTVD_Private());
    PetscCall(FlucaFinalize());
    return 0;
  }
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

  test:
    suffix: tvd
    nsize: 1
    args: -test_case tvd

TEST*/
