#include <fluca/private/segsrkimpl.h>

/* --- ARKIMEX L2 tableau -------------------------------------------------- */

static PetscErrorCode SegSRKSetTableau_L2(Seg_SRK *srk)
{
  /* ARKIMEX L2: s=2, 2nd order, L-stable, stiffly accurate, FSAL
     gamma = 1 - 1/sqrt(2)
     Implicit At:  [[0,       0    ],     Explicit A:  [[0, 0],
                    [1-gamma, gamma]]                   [1, 0]]
     bt = [1-gamma, gamma],  b = [1-gamma, gamma]
     ct = [0, 1],            c = [0, 1]                            */
  const PetscReal gamma = 1. - 1. / PetscSqrtReal(2.);

  PetscFunctionBegin;
  srk->s     = 2;
  srk->order = 2;

  PetscCall(PetscMalloc1(4, &srk->At));
  PetscCall(PetscMalloc1(4, &srk->A));
  PetscCall(PetscMalloc1(2, &srk->bt));
  PetscCall(PetscMalloc1(2, &srk->b));
  PetscCall(PetscMalloc1(2, &srk->ct));
  PetscCall(PetscMalloc1(2, &srk->c));

  /* Implicit tableau */
  srk->At[0] = 0.;
  srk->At[1] = 0.;
  srk->At[2] = 1. - gamma;
  srk->At[3] = gamma;

  /* Explicit tableau */
  srk->A[0] = 0.;
  srk->A[1] = 0.;
  srk->A[2] = 1.;
  srk->A[3] = 0.;

  srk->bt[0] = 1. - gamma;
  srk->bt[1] = gamma;
  srk->b[0]  = 1. - gamma;
  srk->b[1]  = gamma;
  srk->ct[0] = 0.;
  srk->ct[1] = 1.;
  srk->c[0]  = 0.;
  srk->c[1]  = 1.;

  srk->stiffly_accurate     = PETSC_TRUE;
  srk->fsal                 = PETSC_TRUE;
  srk->explicit_first_stage = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Setup: build matrices and KSPs -------------------------------------- */

static PetscErrorCode AssembleLaplacianSubMatrices_SRK(Seg seg)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;
  DM       dm  = seg->dm;
  Mat      M_full;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(DMCreateMatrix(dm, &M_full));
  PetscCall(MatSetOption(M_full, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatZeroEntries(M_full));

  for (d = 0; d < srk->dim; d++) PetscCall(FlucaFDGetOperator(srk->fd_laplacian[d], dm, dm, M_full));
  PetscCall(MatAssemblyBegin(M_full, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M_full, MAT_FINAL_ASSEMBLY));

  for (d = 0; d < srk->dim; d++) PetscCall(MatCreateSubMatrix(M_full, srk->is_comp[d], srk->is_comp[d], MAT_INITIAL_MATRIX, &srk->L_helm[d]));
  PetscCall(MatDestroy(&M_full));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssemblePressureMatrix_SRK(Seg seg)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;
  DM       dm  = seg->dm;
  Mat      M_full;
  PetscInt d;

  PetscFunctionBegin;
  /* Build compact Laplacian: sum_d d^2p/dx_d^2 */
  {
    FlucaFD compact_dir[SEG_SRK_MAX_DIM];

    for (d = 0; d < srk->dim; d++) {
      PetscCall(FlucaFDDerivativeCreate(dm, (FlucaFDDirection)d, 2, 2, DMSTAG_ELEMENT, srk->dim, DMSTAG_ELEMENT, srk->dim, &compact_dir[d]));
      PetscCall(FlucaFDSetUp(compact_dir[d]));
    }
    PetscCall(FlucaFDSumCreate(srk->dim, compact_dir, &srk->fd_pres_lap));
    PetscCall(FlucaFDSetUp(srk->fd_pres_lap));
    for (d = 0; d < srk->dim; d++) PetscCall(FlucaFDDestroy(&compact_dir[d]));
  }

  /* Assemble into full matrix and extract pressure block */
  PetscCall(DMCreateMatrix(dm, &M_full));
  PetscCall(MatSetOption(M_full, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatZeroEntries(M_full));
  PetscCall(FlucaFDGetOperator(srk->fd_pres_lap, dm, dm, M_full));
  PetscCall(MatAssemblyBegin(M_full, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M_full, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateSubMatrix(M_full, srk->is_p, srk->is_p, MAT_INITIAL_MATRIX, &srk->A_pres));
  PetscCall(MatDestroy(&M_full));

  /* Negate: the compact Laplacian has negative eigenvalues; CG needs positive definite.
     A_pres = -L is positive semi-definite. */
  PetscCall(MatScale(srk->A_pres, -1.));

  /* Pressure null space (constant vector) */
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)seg), PETSC_TRUE, 0, NULL, &srk->pres_nullspace));
  PetscCall(MatSetNullSpace(srk->A_pres, srk->pres_nullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKAssembleHelmholtz(Seg seg)
{
  Seg_SRK  *srk = (Seg_SRK *)seg->data;
  PetscInt  s   = srk->s, d;
  PetscReal gamma_diag, helm_shift;

  PetscFunctionBegin;
  PetscCheck(seg->dt > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Time step size must be positive for Helmholtz assembly");
  gamma_diag = srk->At[(s - 1) * s + (s - 1)]; /* At[s-1][s-1] */
  helm_shift = srk->rho / (gamma_diag * seg->dt);

  for (d = 0; d < srk->dim; d++) {
    if (srk->A_helm[d]) PetscCall(MatDestroy(&srk->A_helm[d]));
    PetscCall(MatDuplicate(srk->L_helm[d], MAT_COPY_VALUES, &srk->A_helm[d]));
    PetscCall(MatShift(srk->A_helm[d], helm_shift));
    PetscCall(KSPSetOperators(srk->ksp_helm[d], srk->A_helm[d], srk->A_helm[d]));
  }
  srk->dt_assembled = seg->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegSetUp_SRK(Seg seg)
{
  Seg_SRK    *srk = (Seg_SRK *)seg->data;
  PetscInt    d, i;
  MPI_Comm    comm;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)seg, &comm));

  /* Validate required inputs */
  PetscCheck(srk->dim > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "Field IS not set. Call SegSRKSetFieldIS() first");
  PetscCheck(srk->rho > 0., comm, PETSC_ERR_ARG_WRONGSTATE, "Density not set. Call SegSRKSetDensity() first");
  PetscCheck(srk->fd_div, comm, PETSC_ERR_ARG_WRONGSTATE, "Divergence operator not set. Call SegSRKSetDivergence() first");
  PetscCheck(seg->rhsfn, comm, PETSC_ERR_ARG_WRONGSTATE, "RHS function not set. Call SegSetRHSFunction() first");
  for (d = 0; d < srk->dim; d++) {
    PetscCheck(srk->fd_laplacian[d], comm, PETSC_ERR_ARG_WRONGSTATE, "Laplacian operator [%" PetscInt_FMT "] not set", d);
    PetscCheck(srk->fd_grad_p[d], comm, PETSC_ERR_ARG_WRONGSTATE, "Gradient operator [%" PetscInt_FMT "] not set", d);
  }

  /* Allocate stage vectors */
  PetscCall(PetscMalloc1(srk->s, &srk->Y));
  PetscCall(PetscMalloc1(srk->s, &srk->K_u));
  PetscCall(PetscMalloc1(srk->s, &srk->K_hat_u));
  for (i = 0; i < srk->s; i++) {
    PetscCall(VecDuplicate(seg->sol, &srk->Y[i]));
    PetscCall(VecDuplicate(seg->sol, &srk->K_u[i]));
    PetscCall(VecDuplicate(seg->sol, &srk->K_hat_u[i]));
  }
  PetscCall(VecDuplicate(seg->sol, &srk->K_hat_prev));
  PetscCall(VecDuplicate(seg->sol, &srk->Z));
  PetscCall(VecDuplicate(seg->sol, &srk->U_prev));
  PetscCall(VecDuplicate(seg->sol, &srk->work1));
  PetscCall(VecDuplicate(seg->sol, &srk->work2));
  PetscCall(VecDuplicate(seg->sol, &srk->work3));

  /* Assemble operator sub-matrices */
  PetscCall(AssembleLaplacianSubMatrices_SRK(seg));
  PetscCall(AssemblePressureMatrix_SRK(seg));

  /* Create KSPs */
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)seg, &prefix));

  for (d = 0; d < srk->dim; d++) {
    PetscCall(KSPCreate(comm, &srk->ksp_helm[d]));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)srk->ksp_helm[d], (PetscObject)seg, 1));
    PetscCall(KSPSetOptionsPrefix(srk->ksp_helm[d], prefix));
    PetscCall(KSPAppendOptionsPrefix(srk->ksp_helm[d], "seg_helm_"));
    PetscCall(KSPSetType(srk->ksp_helm[d], KSPCG));
    PetscCall(KSPSetTolerances(srk->ksp_helm[d], 1.e-10, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
    {
      PC pc;
      PetscCall(KSPGetPC(srk->ksp_helm[d], &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
    }
  }

  PetscCall(KSPCreate(comm, &srk->ksp_pres));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)srk->ksp_pres, (PetscObject)seg, 1));
  PetscCall(KSPSetOptionsPrefix(srk->ksp_pres, prefix));
  PetscCall(KSPAppendOptionsPrefix(srk->ksp_pres, "seg_pres_"));
  PetscCall(KSPSetType(srk->ksp_pres, KSPCG));
  PetscCall(KSPSetTolerances(srk->ksp_pres, 1.e-10, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(KSPSetOperators(srk->ksp_pres, srk->A_pres, srk->A_pres));
  {
    PC pc;
    PetscCall(KSPGetPC(srk->ksp_pres, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
  }

  srk->first_step   = PETSC_TRUE;
  srk->dt_assembled = -1.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- SetFromOptions ------------------------------------------------------ */

static PetscErrorCode SegSetFromOptions_SRK(Seg seg, PetscOptionItems PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Seg SRK options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Reset / Destroy ----------------------------------------------------- */

static PetscErrorCode SegReset_SRK(Seg seg)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;
  PetscInt d, i;

  PetscFunctionBegin;
  if (srk->Y) {
    for (i = 0; i < srk->s; i++) {
      PetscCall(VecDestroy(&srk->Y[i]));
      PetscCall(VecDestroy(&srk->K_u[i]));
      PetscCall(VecDestroy(&srk->K_hat_u[i]));
    }
    PetscCall(PetscFree(srk->Y));
    PetscCall(PetscFree(srk->K_u));
    PetscCall(PetscFree(srk->K_hat_u));
  }
  PetscCall(VecDestroy(&srk->K_hat_prev));
  PetscCall(VecDestroy(&srk->Z));
  PetscCall(VecDestroy(&srk->U_prev));
  PetscCall(VecDestroy(&srk->work1));
  PetscCall(VecDestroy(&srk->work2));
  PetscCall(VecDestroy(&srk->work3));

  for (d = 0; d < srk->dim; d++) {
    PetscCall(MatDestroy(&srk->L_helm[d]));
    PetscCall(MatDestroy(&srk->A_helm[d]));
    PetscCall(KSPDestroy(&srk->ksp_helm[d]));
  }
  PetscCall(MatDestroy(&srk->A_pres));
  PetscCall(KSPDestroy(&srk->ksp_pres));
  PetscCall(FlucaFDDestroy(&srk->fd_pres_lap));
  PetscCall(MatNullSpaceDestroy(&srk->pres_nullspace));

  srk->dt_assembled = -1.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegDestroy_SRK(Seg seg)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(srk->At));
  PetscCall(PetscFree(srk->A));
  PetscCall(PetscFree(srk->bt));
  PetscCall(PetscFree(srk->b));
  PetscCall(PetscFree(srk->ct));
  PetscCall(PetscFree(srk->c));
  PetscCall(PetscFree(srk));
  seg->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- View ---------------------------------------------------------------- */

static PetscErrorCode SegView_SRK(Seg seg, PetscViewer viewer)
{
  Seg_SRK  *srk = (Seg_SRK *)seg->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stages: %" PetscInt_FMT "\n", srk->s));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Order: %" PetscInt_FMT "\n", srk->order));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stiffly accurate: %s\n", srk->stiffly_accurate ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "FSAL: %s\n", srk->fsal ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Density: %g\n", (double)srk->rho));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Dimension: %" PetscInt_FMT "\n", srk->dim));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Operator setters ---------------------------------------------------- */

PetscErrorCode SegSRKSetLaplacian(Seg seg, PetscInt d, FlucaFD fd)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  PetscCheck(d >= 0 && d < SEG_SRK_MAX_DIM, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Component %" PetscInt_FMT " out of range [0, %d)", d, SEG_SRK_MAX_DIM);
  srk->fd_laplacian[d] = fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKSetGradient(Seg seg, PetscInt d, FlucaFD fd)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  PetscCheck(d >= 0 && d < SEG_SRK_MAX_DIM, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Component %" PetscInt_FMT " out of range [0, %d)", d, SEG_SRK_MAX_DIM);
  srk->fd_grad_p[d] = fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKSetDivergence(Seg seg, FlucaFD fd)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  srk->fd_div = fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKSetFieldIS(Seg seg, PetscInt dim, IS is_vel, IS is_p, IS is_comp[])
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;
  PetscInt d;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  PetscCheck(dim > 0 && dim <= SEG_SRK_MAX_DIM, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " out of range [1, %d]", dim, SEG_SRK_MAX_DIM);
  srk->dim    = dim;
  srk->is_vel = is_vel;
  srk->is_p   = is_p;
  for (d = 0; d < dim; d++) srk->is_comp[d] = is_comp[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKSetStabilization(Seg seg, FlucaFD fd)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  srk->fd_pstab = fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKSetDensity(Seg seg, PetscReal rho)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  PetscCheck(rho > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Density must be positive, got %g", (double)rho);
  srk->rho = rho;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Factory ------------------------------------------------------------- */

PetscErrorCode SegCreate_SRK(Seg seg)
{
  Seg_SRK *srk;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(PetscNew(&srk));

  /* Initialize tableau pointers before loading (safe on partial allocation failure) */
  srk->At = NULL;
  srk->A  = NULL;
  srk->bt = NULL;
  srk->b  = NULL;
  srk->ct = NULL;
  srk->c  = NULL;
  PetscCall(SegSRKSetTableau_L2(srk));

  /* Initialize all pointers to NULL */
  srk->Y              = NULL;
  srk->K_u            = NULL;
  srk->K_hat_u        = NULL;
  srk->K_hat_prev     = NULL;
  srk->Z              = NULL;
  srk->U_prev         = NULL;
  srk->work1          = NULL;
  srk->work2          = NULL;
  srk->work3          = NULL;
  srk->fd_div         = NULL;
  srk->fd_pstab       = NULL;
  srk->fd_pres_lap    = NULL;
  srk->is_vel         = NULL;
  srk->is_p           = NULL;
  srk->ksp_pres       = NULL;
  srk->A_pres         = NULL;
  srk->pres_nullspace = NULL;
  srk->rho            = 0.;
  srk->dim            = 0;
  srk->dt_assembled   = -1.;
  srk->first_step     = PETSC_TRUE;

  for (d = 0; d < SEG_SRK_MAX_DIM; d++) {
    srk->fd_laplacian[d] = NULL;
    srk->fd_grad_p[d]    = NULL;
    srk->is_comp[d]      = NULL;
    srk->L_helm[d]       = NULL;
    srk->A_helm[d]       = NULL;
    srk->ksp_helm[d]     = NULL;
  }

  seg->data = srk;

  seg->ops->setfromoptions = SegSetFromOptions_SRK;
  seg->ops->setup          = SegSetUp_SRK;
  seg->ops->step           = SegStep_SRK;
  seg->ops->reset          = SegReset_SRK;
  seg->ops->destroy        = SegDestroy_SRK;
  seg->ops->view           = SegView_SRK;
  PetscFunctionReturn(PETSC_SUCCESS);
}
