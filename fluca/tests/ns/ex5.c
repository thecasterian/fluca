#include <flucans.h>
#include <flucasys.h>

static const char help[] = "Test PCABF with a pressure-pressure block and a null vector that vanishes on a decoupled pressure entry\n"
                           "Unknowns: velocity 0-1, face velocity 2-3, pressure 4-6; pressure 6 is decoupled (identity row)\n";

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

int main(int argc, char **argv)
{
  Mat                M, S;
  Vec                nv;
  MatNullSpace       nsp, snsp;
  KSP                ksp, kspA, kspS;
  PC                 pc;
  IS                 isv, isV, isp;
  PetscBool          isnull, has_cnst;
  PetscInt           nvec, r, c, ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;
  const Vec         *vecs;
  /* (row, column, value) of [A 0 G; -T I -R; 0 D C] */
  const PetscInt    ent_r[] = {0, 1, 0, 0, 1, 1, 2, 2, 3, 2, 3, 2, 2, 4, 4, 5, 5, 6};
  const PetscInt    ent_c[] = {0, 1, 4, 5, 4, 5, 0, 1, 1, 2, 3, 4, 5, 2, 3, 2, 3, 6};
  const PetscScalar ent_v[] = {1., 1., -1., 1., -1., 1., -0.5, -0.5, -1., 1., 1., 0.1, -0.1, 1., 1., -1., -1., 1.};
  const PetscScalar half    = 1. / PetscSqrtReal(2.);

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(NSInitializePackage());
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 7, 7, 7, NULL, 7, NULL, &M));
  for (r = 0; r < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(ent_r); ++r) PetscCall(MatSetValue(M, ent_r[r], ent_c[r], ent_v[r], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  /* Null vector: constant on pressures 4 and 5, zero on the decoupled pressure 6 */
  PetscCall(MatCreateVecs(M, &nv, NULL));
  PetscCall(VecZeroEntries(nv));
  PetscCall(VecSetValue(nv, 4, half, INSERT_VALUES));
  PetscCall(VecSetValue(nv, 5, half, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(nv));
  PetscCall(VecAssemblyEnd(nv));
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nv, &nsp));
  PetscCall(MatSetNullSpace(M, nsp));

  PetscCall(ISCreateStride(PETSC_COMM_WORLD, 2, 0, 1, &isv));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, 2, 2, 1, &isV));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, 3, 4, 1, &isp));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCABF));
  PetscCall(PCABFSetFieldIS(pc, isv, isV, isp));
  PetscCall(KSPSetUp(ksp));

  PetscCall(PCABFGetSubKSPs(pc, &kspA, &kspS));
  PetscCall(KSPGetOperators(kspS, &S, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Schur complement entries:\n"));
  for (r = 0; r < 3; ++r) {
    PetscCall(MatGetRow(S, r, &ncols, &cols, &vals));
    for (c = 0; c < ncols; ++c)
      if (Clean_Private(PetscRealPart(vals[c])) != 0.) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  S[%" PetscInt_FMT "][%" PetscInt_FMT "] = %.6f\n", r, cols[c], (double)PetscRealPart(vals[c])));
    PetscCall(MatRestoreRow(S, r, &ncols, &cols, &vals));
  }
  PetscCall(MatGetNullSpace(S, &snsp));
  PetscCall(MatNullSpaceGetVecs(snsp, &has_cnst, &nvec, &vecs));
  PetscCall(MatNullSpaceTest(snsp, S, &isnull));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Schur null space: constant %s, %" PetscInt_FMT " vector(s), S n = 0: %s\n", has_cnst ? "yes" : "no", nvec, isnull ? "yes" : "no"));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(ISDestroy(&isp));
  PetscCall(ISDestroy(&isV));
  PetscCall(ISDestroy(&isv));
  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(VecDestroy(&nv));
  PetscCall(MatDestroy(&M));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: pressure_block
    nsize: 1

TEST*/
