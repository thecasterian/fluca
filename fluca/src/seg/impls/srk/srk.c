#include <fluca/private/segsrkimpl.h>
#include "srktab.h"

/* --- SegSRKSetType / GetType --------------------------------------------- */

PetscErrorCode SegSRKSetType(Seg seg, SegSRKType srktype)
{
  Seg_SRK   *srk = (Seg_SRK *)seg->data;
  SRKTableau tab;
  PetscBool  match;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  PetscAssertPointer(srktype, 2);

  /* Quick exit if already set to this type */
  if (srk->tableau) {
    PetscCall(PetscStrcmp(srk->tableau->name, srktype, &match));
    if (match) PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(SegSRKInitializePackage());
  PetscCall(SegSRKLookupTableau_Internal(srktype, &tab));

  /* Reset stage vectors if already set up (stage count may change) */
  if (seg->setupcalled) PetscCall(seg->ops->reset(seg));

  srk->tableau = tab;

  if (seg->setupcalled) PetscCall(seg->ops->setup(seg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKGetType(Seg seg, SegSRKType *srktype)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  PetscAssertPointer(srktype, 2);
  *srktype = srk->tableau ? srk->tableau->name : NULL;
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

  for (d = 0; d < srk->dim; d++) PetscCall(FlucaFDGetOperator(srk->fd_diff[d], dm, dm, M_full));
  PetscCall(MatAssemblyBegin(M_full, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M_full, MAT_FINAL_ASSEMBLY));

  /* fd_diff = +nu*nabla^2 (negative semi-definite); negate to get L_helm = -F_diff (positive semi-definite) */
  for (d = 0; d < srk->dim; d++) {
    PetscCall(MatCreateSubMatrix(M_full, srk->is_comp[d], srk->is_comp[d], MAT_INITIAL_MATRIX, &srk->L_helm[d]));
    PetscCall(MatScale(srk->L_helm[d], -1.));
  }
  PetscCall(MatDestroy(&M_full));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssemblePressureMatrix_SRK(Seg seg)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;
  DM       dm  = seg->dm;
  Mat      M_full;

  PetscFunctionBegin;
  /* Assemble compact Laplacian (provided by PhysINS) into full matrix and extract pressure block */
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
  Seg_SRK   *srk = (Seg_SRK *)seg->data;
  SRKTableau tab = srk->tableau;
  PetscInt   s   = tab->s, d;
  PetscReal  gamma_diag, helm_shift;

  PetscFunctionBegin;
  PetscCheck(seg->dt > 0., PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_WRONGSTATE, "Time step size must be positive for Helmholtz assembly");
  gamma_diag = tab->A[(s - 1) * s + (s - 1)]; /* a_ss */
  helm_shift = 1. / (gamma_diag * seg->dt);

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
  SRKTableau  tab = srk->tableau;
  PetscInt    s   = tab->s, d, i, k;
  PetscReal   gamma_diag;
  MPI_Comm    comm;
  const char *prefix;
  Vec         tmp_p;
  PC          pc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)seg, &comm));

  /* Validate required inputs */
  PetscCheck(srk->tableau, comm, PETSC_ERR_ARG_WRONGSTATE, "SRK tableau not set. Call SegSRKSetType() first");
  /* SRK uses a single Helmholtz shift for all implicit stages — require SDIRK (constant diagonal) */
  gamma_diag = tab->A[(s - 1) * s + (s - 1)];
  for (i = 1; i < s; i++) {
    PetscCheck(tab->A[i * s + i] == gamma_diag, comm, PETSC_ERR_SUP, "SRK tableau \"%s\" is not SDIRK: A[%" PetscInt_FMT "][%" PetscInt_FMT "] = %g differs from gamma = %g", tab->name, i, i, (double)tab->A[i * s + i], (double)gamma_diag);
  }
  PetscCheck(srk->dim > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "Field IS not set. Call SegSRKSetFieldIS() first");
  PetscCheck(srk->rho > 0., comm, PETSC_ERR_ARG_WRONGSTATE, "Density not set. Call SegSRKSetDensity() first");
  PetscCheck(srk->fd_div, comm, PETSC_ERR_ARG_WRONGSTATE, "Divergence operator not set. Call SegSRKSetDivergence() first");
  PetscCheck(seg->rhsfn, comm, PETSC_ERR_ARG_WRONGSTATE, "RHS function not set. Call SegSetRHSFunction() first");
  PetscCheck(srk->fd_pres_lap, comm, PETSC_ERR_ARG_WRONGSTATE, "Pressure Laplacian not set. Call SegSRKSetPressureLaplacian() first");
  for (d = 0; d < srk->dim; d++) {
    PetscCheck(srk->fd_diff[d], comm, PETSC_ERR_ARG_WRONGSTATE, "Diffusion operator [%" PetscInt_FMT "] not set", d);
    PetscCheck(srk->fd_grad_p[d], comm, PETSC_ERR_ARG_WRONGSTATE, "Gradient operator [%" PetscInt_FMT "] not set", d);
  }

  /* Allocate stage vectors */
  PetscCall(PetscMalloc1(s, &srk->Y));
  PetscCall(PetscMalloc1(s, &srk->K_u));
  PetscCall(PetscMalloc1(s, &srk->K_hat_u));
  for (i = 0; i < s; i++) {
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

  /* Allocate mu/mu_tilde pressure vectors */
  PetscCall(MatCreateVecs(srk->A_pres, NULL, &tmp_p));
  PetscCall(PetscMalloc1(s, &srk->mu_tilde));
  for (i = 0; i < s; i++) PetscCall(VecDuplicate(tmp_p, &srk->mu_tilde[i]));
  PetscCall(VecDuplicate(tmp_p, &srk->mu_work));

  /* CK-type correction: precompute d_j and allocate nu_tilde_1 */
  if (!tab->ars_type) {
    PetscCall(VecDuplicate(tmp_p, &srk->nu_tilde_1));
    PetscCall(PetscMalloc1(s, &srk->d_j));
    srk->d_j[0] = 1.;
    for (i = 1; i < s; i++) {
      srk->d_j[i] = 0.;
      for (k = 0; k < i; k++) srk->d_j[i] -= tab->A[i * s + k] * srk->d_j[k];
      srk->d_j[i] /= gamma_diag;
    }
  }

  PetscCall(VecDestroy(&tmp_p));

  /* Create KSPs */
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)seg, &prefix));

  for (d = 0; d < srk->dim; d++) {
    PetscCall(KSPCreate(comm, &srk->ksp_helm[d]));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)srk->ksp_helm[d], (PetscObject)seg, 1));
    PetscCall(KSPSetOptionsPrefix(srk->ksp_helm[d], prefix));
    PetscCall(KSPAppendOptionsPrefix(srk->ksp_helm[d], "seg_helm_"));
    PetscCall(KSPSetType(srk->ksp_helm[d], KSPCG));
    PetscCall(KSPSetTolerances(srk->ksp_helm[d], 1.e-10, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(KSPGetPC(srk->ksp_helm[d], &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
  }

  PetscCall(KSPCreate(comm, &srk->ksp_pres));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)srk->ksp_pres, (PetscObject)seg, 1));
  PetscCall(KSPSetOptionsPrefix(srk->ksp_pres, prefix));
  PetscCall(KSPAppendOptionsPrefix(srk->ksp_pres, "seg_pres_"));
  PetscCall(KSPSetType(srk->ksp_pres, KSPCG));
  PetscCall(KSPSetTolerances(srk->ksp_pres, 1.e-10, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(KSPSetOperators(srk->ksp_pres, srk->A_pres, srk->A_pres));
  PetscCall(KSPGetPC(srk->ksp_pres, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));

  srk->first_step   = PETSC_TRUE;
  srk->dt_assembled = -1.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- SetFromOptions ------------------------------------------------------ */

static PetscErrorCode SegSetFromOptions_SRK(Seg seg, PetscOptionItems PetscOptionsObject)
{
  Seg_SRK       *srk = (Seg_SRK *)seg->data;
  SRKTableauLink link, l;
  PetscInt       count, choice;
  const char   **namelist;
  const char    *default_name;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Seg SRK options");

  /* Build dynamic list of registered schemes */
  PetscCall(SegSRKInitializePackage());
  PetscCall(SegSRKGetTableauList_Internal(&link));
  count = 0;
  for (l = link; l; l = l->next) count++;

  PetscCall(PetscMalloc1(count, &namelist));
  count = 0;
  for (l = link; l; l = l->next) namelist[count++] = l->tab.name;

  default_name = srk->tableau ? srk->tableau->name : namelist[0];
  PetscCall(PetscOptionsEList("-seg_srk_type", "SRK tableau type", "SegSRKSetType", namelist, count, default_name, &choice, &flg));
  if (flg) PetscCall(SegSRKSetType(seg, namelist[choice]));

  PetscCall(PetscFree(namelist));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Reset / Destroy ----------------------------------------------------- */

static PetscErrorCode SegReset_SRK(Seg seg)
{
  Seg_SRK   *srk = (Seg_SRK *)seg->data;
  SRKTableau tab = srk->tableau;
  PetscInt   s, d, i;

  PetscFunctionBegin;
  s = tab ? tab->s : 0;

  if (srk->Y) {
    for (i = 0; i < s; i++) {
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

  if (srk->mu_tilde) {
    for (i = 0; i < s; i++) PetscCall(VecDestroy(&srk->mu_tilde[i]));
    PetscCall(PetscFree(srk->mu_tilde));
  }
  PetscCall(VecDestroy(&srk->mu_work));
  PetscCall(VecDestroy(&srk->nu_tilde_1));
  PetscCall(PetscFree(srk->d_j));

  for (d = 0; d < srk->dim; d++) {
    PetscCall(MatDestroy(&srk->L_helm[d]));
    PetscCall(MatDestroy(&srk->A_helm[d]));
    PetscCall(KSPDestroy(&srk->ksp_helm[d]));
  }
  PetscCall(MatDestroy(&srk->A_pres));
  PetscCall(KSPDestroy(&srk->ksp_pres));
  PetscCall(MatNullSpaceDestroy(&srk->pres_nullspace));

  srk->dt_assembled = -1.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SegDestroy_SRK(Seg seg)
{
  PetscFunctionBegin;
  PetscCall(SegReset_SRK(seg));
  /* Tableau is owned by the registry, not by us */
  PetscCall(PetscFree(seg->data));
  seg->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- View ---------------------------------------------------------------- */

static PetscErrorCode SegView_SRK(Seg seg, PetscViewer viewer)
{
  Seg_SRK   *srk = (Seg_SRK *)seg->data;
  SRKTableau tab = srk->tableau;
  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii && tab) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Tableau: %s\n", tab->name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stages: %" PetscInt_FMT "\n", tab->s));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Order: %" PetscInt_FMT "\n", tab->order));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stiffly accurate: %s\n", tab->stiffly_accurate ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "FSAL: %s\n", tab->fsal ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ARS type: %s\n", tab->ars_type ? "yes" : "no"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "(alpha*tau)_max: %g\n", (double)tab->alpha_tau_max));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Density: %g\n", (double)srk->rho));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Dimension: %" PetscInt_FMT "\n", srk->dim));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Operator setters ---------------------------------------------------- */

PetscErrorCode SegSRKSetDiffusion(Seg seg, PetscInt d, FlucaFD fd)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  PetscCheck(d >= 0 && d < SEG_SRK_MAX_DIM, PetscObjectComm((PetscObject)seg), PETSC_ERR_ARG_OUTOFRANGE, "Component %" PetscInt_FMT " out of range [0, %d)", d, SEG_SRK_MAX_DIM);
  srk->fd_diff[d] = fd;
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

PetscErrorCode SegSRKSetPressureLaplacian(Seg seg, FlucaFD fd)
{
  Seg_SRK *srk = (Seg_SRK *)seg->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(seg, SEG_CLASSID, 1, SEGSRK);
  srk->fd_pres_lap = fd;
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

  PetscFunctionBegin;
  PetscCall(PetscNew(&srk)); /* zero-initialized by PetscNew */

  /* Non-zero defaults only */
  srk->dt_assembled = -1.;
  srk->first_step   = PETSC_TRUE;

  seg->data = srk;

  seg->ops->setfromoptions = SegSetFromOptions_SRK;
  seg->ops->setup          = SegSetUp_SRK;
  seg->ops->step           = SegStep_SRK;
  seg->ops->reset          = SegReset_SRK;
  seg->ops->destroy        = SegDestroy_SRK;
  seg->ops->view           = SegView_SRK;

  /* Set default tableau */
  PetscCall(SegSRKSetType(seg, SEGSRKARS343));
  PetscFunctionReturn(PETSC_SUCCESS);
}
