#include <petsc/private/pcimpl.h>
#include <flucans.h>

const char *const PCABFAinvTypes[] = {"ID", "DIAG", "ROWSUM", "PCABFAinvType", "", NULL};

typedef struct {
  PetscInt      vidx; /* index of velocity field */
  PetscInt      Vidx; /* index of face-normal velocity field */
  PetscInt      pidx; /* index of pressure field */
  PCABFAinvType schurainv;
  PCABFAinvType upperainv;

  KSP kspA;
  KSP kspS;

  Mat A;    /* momentum equation operator */
  Mat negT; /* negative face-normal velocity interpolation operator */
  Mat G;    /* pressure gradient operator */
  Mat D;    /* face-normal velocity divergence operator */
  Mat negR; /* Rhie-Chow interpolation correction operator */
  Mat S;    /* approximate Schur complement */

  MatNullSpace nullspace;

  Vec Adiag;
  Vec vstar;
  Vec Vstar;
  Vec Srhs;
  Vec invA2Gp;
  Vec negRp;
} PC_ABF;

static PetscErrorCode PCABFCreateKSP_Private(PC pc, const char prefix[], KSP *ksp)
{
  KSP  k;
  char kprefix[256];

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &k));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)k, (PetscObject)pc, 1));
  PetscCall(PetscObjectSetOptions((PetscObject)k, ((PetscObject)pc)->options));
  PetscCall(PetscSNPrintf(kprefix, sizeof(kprefix), "%s%s", ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", prefix));
  PetscCall(KSPSetOptionsPrefix(k, kprefix));
  *ksp = k;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_ABF(PC pc, Vec b, Vec x)
{
  PC_ABF  *abf = (PC_ABF *)pc->data;
  PetscInt m, n;
  IS      *rowis, *colis;
  Vec      momrhs, interprhs, contrhs, v, V, p;

  PetscFunctionBegin;
  PetscCall(MatNestGetSize(pc->pmat, &m, &n));
  PetscCall(PetscMalloc2(m, &rowis, n, &colis));
  PetscCall(MatNestGetISs(pc->pmat, rowis, colis));
  PetscCall(VecGetSubVector(b, rowis[abf->vidx], &momrhs));
  PetscCall(VecGetSubVector(b, rowis[abf->Vidx], &interprhs));
  PetscCall(VecGetSubVector(b, rowis[abf->pidx], &contrhs));
  PetscCall(VecGetSubVector(x, colis[abf->vidx], &v));
  PetscCall(VecGetSubVector(x, colis[abf->Vidx], &V));
  PetscCall(VecGetSubVector(x, colis[abf->pidx], &p));

  if (!abf->vstar) PetscCall(MatCreateVecs(abf->A, &abf->vstar, NULL));
  if (!abf->Vstar) PetscCall(MatCreateVecs(abf->negT, NULL, &abf->Vstar));
  if (!abf->Srhs) PetscCall(MatCreateVecs(abf->S, NULL, &abf->Srhs));
  if (!abf->invA2Gp) PetscCall(MatCreateVecs(abf->A, &abf->invA2Gp, NULL));

  /* Stage 1: solve the lower triangular matrix */
  PetscCall(KSPSolve(abf->kspA, momrhs, abf->vstar));
  PetscCall(MatMult(abf->negT, abf->vstar, abf->Vstar));
  PetscCall(VecAYPX(abf->Vstar, -1., interprhs));
  PetscCall(MatMult(abf->D, abf->Vstar, abf->Srhs));
  PetscCall(VecAYPX(abf->Srhs, -1., contrhs));
  PetscCall(KSPSolve(abf->kspS, abf->Srhs, p));

  /* Stage 2: solve the upper triangular matrix */
  PetscCall(MatMult(abf->G, p, abf->invA2Gp));
  switch (abf->upperainv) {
  case PC_ABF_AINV_ID:
    break;
  case PC_ABF_AINV_DIAG:
  case PC_ABF_AINV_ROWSUM:
    if (!abf->Adiag) PetscCall(MatCreateVecs(abf->A, &abf->Adiag, NULL));
    if (abf->upperainv == PC_ABF_AINV_DIAG) PetscCall(MatGetDiagonal(abf->A, abf->Adiag));
    else PetscCall(MatGetRowSum(abf->A, abf->Adiag));
    PetscCall(VecReciprocal(abf->Adiag));
    PetscCall(VecPointwiseMult(abf->invA2Gp, abf->Adiag, abf->invA2Gp));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Unsupported Ainv type");
  }
  PetscCall(VecWAXPY(v, -1., abf->invA2Gp, abf->vstar));
  PetscCall(MatMultAdd(abf->negT, abf->invA2Gp, abf->Vstar, V));
  if (abf->negR) {
    if (!abf->negRp) PetscCall(MatCreateVecs(abf->negR, NULL, &abf->negRp));
    PetscCall(MatMult(abf->negR, p, abf->negRp));
    PetscCall(VecAXPY(V, -1., abf->negRp));
  }

  PetscCall(VecRestoreSubVector(b, rowis[abf->vidx], &momrhs));
  PetscCall(VecRestoreSubVector(b, rowis[abf->Vidx], &interprhs));
  PetscCall(VecRestoreSubVector(b, rowis[abf->pidx], &contrhs));
  PetscCall(VecRestoreSubVector(x, colis[abf->vidx], &v));
  PetscCall(VecRestoreSubVector(x, colis[abf->Vidx], &V));
  PetscCall(VecRestoreSubVector(x, colis[abf->pidx], &p));
  PetscCall(PetscFree2(rowis, colis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_ABF(PC pc)
{
  PC_ABF      *abf = (PC_ABF *)pc->data;
  PetscBool    isnest;
  PetscInt     m, n;
  IS          *rowis, *colis;
  Mat          Gdup, tmp;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat, MATNEST, &isnest));
  PetscCheck(isnest, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Only Pmat of MATNEST type is supported");

  PetscCall(MatDestroy(&abf->A));
  PetscCall(MatDestroy(&abf->negT));
  PetscCall(MatDestroy(&abf->G));
  PetscCall(MatDestroy(&abf->D));
  PetscCall(MatDestroy(&abf->negR));
  PetscCall(MatDestroy(&abf->S));
  PetscCall(MatNullSpaceDestroy(&abf->nullspace));
  PetscCall(VecDestroy(&abf->Adiag));
  PetscCall(VecDestroy(&abf->vstar));
  PetscCall(VecDestroy(&abf->Vstar));
  PetscCall(VecDestroy(&abf->Srhs));
  PetscCall(VecDestroy(&abf->invA2Gp));
  PetscCall(VecDestroy(&abf->negRp));

  PetscCall(MatNestGetSize(pc->pmat, &m, &n));
  PetscCall(PetscMalloc2(m, &rowis, n, &colis));
  PetscCall(MatNestGetISs(pc->pmat, rowis, colis));
  PetscCall(MatCreateSubMatrix(pc->mat, rowis[abf->vidx], colis[abf->vidx], MAT_INITIAL_MATRIX, &abf->A));
  PetscCall(MatCreateSubMatrix(pc->mat, rowis[abf->Vidx], colis[abf->vidx], MAT_INITIAL_MATRIX, &abf->negT));
  PetscCall(MatCreateSubMatrix(pc->mat, rowis[abf->vidx], colis[abf->pidx], MAT_INITIAL_MATRIX, &abf->G));
  PetscCall(MatCreateSubMatrix(pc->mat, rowis[abf->pidx], colis[abf->Vidx], MAT_INITIAL_MATRIX, &abf->D));
  PetscCall(MatCreateSubMatrix(pc->mat, rowis[abf->Vidx], colis[abf->pidx], MAT_INITIAL_MATRIX, &abf->negR));
  PetscCall(PetscFree2(rowis, colis));

  /* S = D ((-T) A^-1 G - (-R)) */
  switch (abf->schurainv) {
  case PC_ABF_AINV_ID:
    PetscCall(MatMatMult(abf->negT, abf->G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp));
    break;
  case PC_ABF_AINV_DIAG:
  case PC_ABF_AINV_ROWSUM:
    if (!abf->Adiag) PetscCall(MatCreateVecs(abf->A, &abf->Adiag, NULL));
    if (abf->schurainv == PC_ABF_AINV_DIAG) PetscCall(MatGetDiagonal(abf->A, abf->Adiag));
    else PetscCall(MatGetRowSum(abf->A, abf->Adiag));
    PetscCall(VecReciprocal(abf->Adiag));
    PetscCall(MatDuplicate(abf->G, MAT_COPY_VALUES, &Gdup));
    PetscCall(MatDiagonalScale(Gdup, abf->Adiag, NULL));
    PetscCall(MatMatMult(abf->negT, Gdup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp));
    PetscCall(MatDestroy(&Gdup));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Unsupported Ainv type");
  }
  if (abf->negR) PetscCall(MatAXPY(tmp, -1., abf->negR, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatMatMult(abf->D, tmp, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &abf->S));
  PetscCall(MatDestroy(&tmp));

  PetscCall(MatGetNullSpace(pc->mat, &nullspace));
  if (nullspace) {
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)pc->mat), PETSC_TRUE, 0, NULL, &abf->nullspace));
    PetscCall(MatSetNullSpace(abf->S, abf->nullspace));
  }

  PetscCall(KSPSetOperators(abf->kspA, abf->A, abf->A));
  PetscCall(KSPSetOperators(abf->kspS, abf->S, abf->S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_ABF(PC pc)
{
  PC_ABF *abf = (PC_ABF *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&abf->kspA));
  PetscCall(KSPDestroy(&abf->kspS));
  PetscCall(MatDestroy(&abf->A));
  PetscCall(MatDestroy(&abf->negT));
  PetscCall(MatDestroy(&abf->G));
  PetscCall(MatDestroy(&abf->D));
  PetscCall(MatDestroy(&abf->S));
  PetscCall(MatDestroy(&abf->negR));
  PetscCall(MatNullSpaceDestroy(&abf->nullspace));
  PetscCall(VecDestroy(&abf->Adiag));
  PetscCall(VecDestroy(&abf->vstar));
  PetscCall(VecDestroy(&abf->Vstar));
  PetscCall(VecDestroy(&abf->Srhs));
  PetscCall(VecDestroy(&abf->invA2Gp));
  PetscCall(VecDestroy(&abf->negRp));

  PetscCall(PCABFCreateKSP_Private(pc, "abf_momentum_", &abf->kspA));
  PetscCall(PCABFCreateKSP_Private(pc, "abf_schur_", &abf->kspS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_ABF(PC pc)
{
  PC_ABF *abf = (PC_ABF *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&abf->kspA));
  PetscCall(KSPDestroy(&abf->kspS));
  PetscCall(MatDestroy(&abf->A));
  PetscCall(MatDestroy(&abf->negT));
  PetscCall(MatDestroy(&abf->G));
  PetscCall(MatDestroy(&abf->D));
  PetscCall(MatDestroy(&abf->S));
  PetscCall(MatDestroy(&abf->negR));
  PetscCall(MatNullSpaceDestroy(&abf->nullspace));
  PetscCall(VecDestroy(&abf->Adiag));
  PetscCall(VecDestroy(&abf->vstar));
  PetscCall(VecDestroy(&abf->Vstar));
  PetscCall(VecDestroy(&abf->Srhs));
  PetscCall(VecDestroy(&abf->invA2Gp));
  PetscCall(VecDestroy(&abf->negRp));

  PetscCall(PetscFree(abf));

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFSetFields_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFGetSubKSPs_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFSetSchurComplementAinvType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFSetUpperTriangularAinvType_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCSetFromOptions_ABF(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_ABF *abf = (PC_ABF *)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "ABF options");
  PetscCall(PetscOptionsEnum("-pc_abf_schur_ainv_type", "Type of approximation used in Schur complement", "PCABFSetSchurComplementAinvType", PCABFAinvTypes, (PetscEnum)abf->schurainv, (PetscEnum *)&abf->schurainv, NULL));
  PetscCall(PetscOptionsEnum("-pc_abf_upper_ainv_type", "Type of approximation used in upper triangular matrix", "PCABFSetUpperTriangularAinvType", PCABFAinvTypes, (PetscEnum)abf->upperainv, (PetscEnum *)&abf->upperainv, NULL));
  PetscCall(KSPSetFromOptions(abf->kspA));
  PetscCall(KSPSetFromOptions(abf->kspS));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCView_ABF(PC pc, PetscViewer viewer)
{
  PC_ABF   *abf = (PC_ABF *)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Approximate Block Factorization (ABF) preconditioner\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Schur complement A inverse type: %s\n", PCABFAinvTypes[abf->schurainv]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Upper triangular A inverse type: %s\n", PCABFAinvTypes[abf->upperainv]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "KSP solver for momentum equation operator\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(KSPView(abf->kspA, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "KSP solver for Schur complement\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(KSPView(abf->kspS, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFSetFields_ABF(PC pc, PetscInt vidx, PetscInt Vidx, PetscInt pidx)
{
  PC_ABF *abf = (PC_ABF *)pc->data;

  PetscFunctionBegin;
  abf->vidx = vidx;
  abf->Vidx = Vidx;
  abf->pidx = pidx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFGetSubKSPs_ABF(PC pc, KSP *kspA, KSP *kspS)
{
  PC_ABF *abf = (PC_ABF *)pc->data;

  PetscFunctionBegin;
  if (kspA) *kspA = abf->kspA;
  if (kspS) *kspS = abf->kspS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFSetSchurComplementAinvType_ABF(PC pc, PCABFAinvType type)
{
  PC_ABF *abf = (PC_ABF *)pc->data;

  PetscFunctionBegin;
  abf->schurainv = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFSetUpperTriangularAinvType_ABF(PC pc, PCABFAinvType type)
{
  PC_ABF *abf = (PC_ABF *)pc->data;

  PetscFunctionBegin;
  abf->upperainv = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_ABF(PC pc)
{
  PC_ABF *abf;

  PetscFunctionBegin;
  PetscCall(PetscNew(&abf));

  abf->vidx      = 0;
  abf->Vidx      = 1;
  abf->pidx      = 2;
  abf->schurainv = PC_ABF_AINV_ID;
  abf->upperainv = PC_ABF_AINV_ID;
  PetscCall(PCABFCreateKSP_Private(pc, "abf_momentum_", &abf->kspA));
  PetscCall(PCABFCreateKSP_Private(pc, "abf_schur_", &abf->kspS));
  abf->A         = NULL;
  abf->negT      = NULL;
  abf->G         = NULL;
  abf->D         = NULL;
  abf->negR      = NULL;
  abf->S         = NULL;
  abf->nullspace = NULL;
  abf->Adiag     = NULL;
  abf->vstar     = NULL;
  abf->Vstar     = NULL;
  abf->Srhs      = NULL;
  abf->invA2Gp   = NULL;
  abf->negRp     = NULL;

  pc->data = abf;

  pc->ops->apply          = PCApply_ABF;
  pc->ops->setup          = PCSetUp_ABF;
  pc->ops->reset          = PCReset_ABF;
  pc->ops->destroy        = PCDestroy_ABF;
  pc->ops->setfromoptions = PCSetFromOptions_ABF;
  pc->ops->view           = PCView_ABF;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFSetFields_C", PCABFSetFields_ABF));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFSetSchurComplementAinvType_C", PCABFSetSchurComplementAinvType_ABF));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFSetUpperTriangularAinvType_C", PCABFSetUpperTriangularAinvType_ABF));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCABFGetSubKSPs_C", PCABFGetSubKSPs_ABF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFSetFields(PC pc, PetscInt vidx, PetscInt Vidx, PetscInt pidx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCABFSetFields_C", (PC, PetscInt, PetscInt, PetscInt), (pc, vidx, Vidx, pidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFGetSubKSPs(PC pc, KSP *kspA, KSP *kspS)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCABFGetSubKSPs_C", (PC, KSP *, KSP *), (pc, kspA, kspS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFSetSchurComplementAinvType(PC pc, PCABFAinvType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCABFSetSchurComplementAinvType_C", (PC, PCABFAinvType), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCABFSetUpperTriangularAinvType(PC pc, PCABFAinvType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCABFSetUpperTriangularAinvType_C", (PC, PCABFAinvType), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}
