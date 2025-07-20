#include <fluca/private/flucaviewercgnsimpl.h>
#include <fluca/private/nsfsmimpl.h>
#include <flucameshcart.h>
#include <petscdmstag.h>

static PetscErrorCode ComputeRHSUStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSVStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSPprime2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeOperatorPprime2d_Private(KSP, Mat, Mat, void *);

PetscErrorCode NSFSMCalculateConvection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  Vec u_global, v_global, UV_global, u_interp, v_interp, UV_u_interp, UV_v_interp, Nu_global, Nv_global;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMGetGlobalVector(dm, &u_global));
  PetscCall(DMGetGlobalVector(dm, &v_global));
  PetscCall(DMGetGlobalVector(fdm, &UV_global));
  PetscCall(DMGetGlobalVector(fdm, &u_interp));
  PetscCall(DMGetGlobalVector(fdm, &v_interp));
  PetscCall(DMGetGlobalVector(fdm, &UV_u_interp));
  PetscCall(DMGetGlobalVector(fdm, &UV_v_interp));
  PetscCall(DMGetGlobalVector(dm, &Nu_global));
  PetscCall(DMGetGlobalVector(dm, &Nv_global));

  PetscCall(DMLocalToGlobal(dm, fsm->v[0], INSERT_VALUES, u_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v[1], INSERT_VALUES, v_global));
  PetscCall(DMLocalToGlobal(fdm, fsm->fv, INSERT_VALUES, UV_global));

  PetscCall(MatMult(fsm->interp_v[0], u_global, u_interp));
  PetscCall(MatMultAdd(fsm->interp_v[1], u_global, u_interp, u_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 0, ns->bcs, ns->t + ns->dt, ADD_VALUES, u_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 0, ns->bcs, ns->t + ns->dt, ADD_VALUES, u_interp));
  PetscCall(MatMult(fsm->interp_v[0], v_global, v_interp));
  PetscCall(MatMultAdd(fsm->interp_v[1], v_global, v_interp, v_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 1, ns->bcs, ns->t + ns->dt, ADD_VALUES, v_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 1, ns->bcs, ns->t + ns->dt, ADD_VALUES, v_interp));

  PetscCall(VecPointwiseMult(UV_u_interp, UV_global, u_interp));
  PetscCall(VecPointwiseMult(UV_v_interp, UV_global, v_interp));
  PetscCall(MatMult(fsm->div_fv, UV_u_interp, Nu_global));
  PetscCall(MatMult(fsm->div_fv, UV_v_interp, Nv_global));

  PetscCall(DMGlobalToLocal(dm, Nu_global, INSERT_VALUES, fsm->N[0]));
  PetscCall(DMGlobalToLocal(dm, Nv_global, INSERT_VALUES, fsm->N[1]));

  PetscCall(DMRestoreGlobalVector(dm, &u_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_global));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_global));
  PetscCall(DMRestoreGlobalVector(fdm, &u_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &v_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_u_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_v_interp));
  PetscCall(DMRestoreGlobalVector(dm, &Nu_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nv_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  PetscErrorCode (*rhs[2])(KSP, Vec, void *) = {ComputeRHSUStar2d_Private, ComputeRHSVStar2d_Private};
  PetscInt d;

  Vec p_global, grad_p[2], grad_p_f, u_star_global, v_star_global, u_tilde, v_tilde, UV_tilde, UV_star_global;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    Vec s;

    PetscCall(KSPSetComputeRHS(fsm->kspv[d], rhs[d], ns));
    PetscCall(KSPSetComputeOperators(fsm->kspv[d], ComputeOperatorsUVstar2d_Private, ns));
    PetscCall(KSPSolve(fsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(fsm->kspv[d], &s));
    PetscCall(DMGlobalToLocal(dm, s, INSERT_VALUES, fsm->v_star[d]));
  }

  /* Calculate face-centered intermediate velocity. */
  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &grad_p[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p[1]));
  PetscCall(DMGetGlobalVector(fdm, &grad_p_f));
  PetscCall(DMGetGlobalVector(dm, &u_star_global));
  PetscCall(DMGetGlobalVector(dm, &v_star_global));
  PetscCall(DMGetGlobalVector(dm, &u_tilde));
  PetscCall(DMGetGlobalVector(dm, &v_tilde));
  PetscCall(DMGetGlobalVector(fdm, &UV_tilde));
  PetscCall(DMGetGlobalVector(fdm, &UV_star_global));

  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v_star[0], INSERT_VALUES, u_star_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v_star[1], INSERT_VALUES, v_star_global));

  PetscCall(MatMult(fsm->grad_p[0], p_global, grad_p[0]));
  PetscCall(MatMult(fsm->grad_p[1], p_global, grad_p[1]));
  PetscCall(MatMult(fsm->grad_p_f, p_global, grad_p_f));

  PetscCall(VecWAXPY(u_tilde, ns->dt / ns->rho, grad_p[0], u_star_global));
  PetscCall(VecWAXPY(v_tilde, ns->dt / ns->rho, grad_p[1], v_star_global));

  PetscCall(MatMult(fsm->interp_v[0], u_tilde, UV_tilde));
  PetscCall(MatMultAdd(fsm->interp_v[1], v_tilde, UV_tilde, UV_tilde));

  PetscCall(VecWAXPY(UV_star_global, -ns->dt / ns->rho, grad_p_f, UV_tilde));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 0, ns->bcs, ns->t + ns->dt, INSERT_VALUES, UV_star_global));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 1, ns->bcs, ns->t + ns->dt, INSERT_VALUES, UV_star_global));
  PetscCall(DMGlobalToLocal(fdm, UV_star_global, INSERT_VALUES, fsm->fv_star));

  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p[0]));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p[1]));
  PetscCall(DMRestoreGlobalVector(fdm, &grad_p_f));
  PetscCall(DMRestoreGlobalVector(dm, &u_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &u_tilde));
  PetscCall(DMRestoreGlobalVector(dm, &v_tilde));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_tilde));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_star_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculatePressureCorrection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Vec     s;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(KSPSetComputeRHS(fsm->kspp, ComputeRHSPprime2d_Private, ns));
  PetscCall(KSPSetComputeOperators(fsm->kspp, ComputeOperatorPprime2d_Private, ns));
  PetscCall(KSPSolve(fsm->kspp, NULL, NULL));
  PetscCall(KSPGetSolution(fsm->kspp, &s));
  PetscCall(DMGlobalToLocal(dm, s, INSERT_VALUES, fsm->p_prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMUpdate2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  Vec u_star_global, v_star_global, fv_star_global, p_prime_global, p_global, u_global, v_global, fv_global;
  Vec grad_p_prime[2], grad_p_prime_f, lap_p_prime;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMGetGlobalVector(dm, &u_star_global));
  PetscCall(DMGetGlobalVector(dm, &v_star_global));
  PetscCall(DMGetGlobalVector(fdm, &fv_star_global));
  PetscCall(DMGetGlobalVector(dm, &p_prime_global));
  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &u_global));
  PetscCall(DMGetGlobalVector(dm, &v_global));
  PetscCall(DMGetGlobalVector(fdm, &fv_global));
  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[1]));
  PetscCall(DMGetGlobalVector(fdm, &grad_p_prime_f));
  PetscCall(DMGetGlobalVector(dm, &lap_p_prime));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecCopy(fsm->N[1], fsm->N_prev[1]));

  PetscCall(DMLocalToGlobal(dm, fsm->v_star[0], INSERT_VALUES, u_star_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v_star[1], INSERT_VALUES, v_star_global));
  PetscCall(DMLocalToGlobal(fdm, fsm->fv_star, INSERT_VALUES, fv_star_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_prime, INSERT_VALUES, p_prime_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));

  PetscCall(MatMult(fsm->grad_p_prime[0], p_prime_global, grad_p_prime[0]));
  PetscCall(MatMult(fsm->grad_p_prime[1], p_prime_global, grad_p_prime[1]));
  PetscCall(MatMult(fsm->grad_p_prime_f, p_prime_global, grad_p_prime_f));
  PetscCall(MatMult(fsm->lap_p_prime, p_prime_global, lap_p_prime));

  PetscCall(VecWAXPY(u_global, -ns->dt / ns->rho, grad_p_prime[0], u_star_global));
  PetscCall(VecWAXPY(v_global, -ns->dt / ns->rho, grad_p_prime[1], v_star_global));
  PetscCall(VecWAXPY(fv_global, -ns->dt / ns->rho, grad_p_prime_f, fv_star_global));

  PetscCall(VecAXPY(p_global, 1., p_prime_global));
  PetscCall(VecAXPY(p_global, -0.5 * ns->mu * ns->dt / ns->rho, lap_p_prime));

  PetscCall(DMGlobalToLocal(dm, u_global, INSERT_VALUES, fsm->v[0]));
  PetscCall(DMGlobalToLocal(dm, v_global, INSERT_VALUES, fsm->v[1]));
  PetscCall(DMGlobalToLocal(fdm, fv_global, INSERT_VALUES, fsm->fv));
  PetscCall(DMGlobalToLocal(dm, p_global, INSERT_VALUES, fsm->p_half));

  PetscCall(VecAXPBYPCZ(fsm->p, 1.5, -0.5, 0, fsm->p_half, fsm->p_half_prev));

  PetscCall(DMRestoreGlobalVector(dm, &u_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_star_global));
  PetscCall(DMRestoreGlobalVector(fdm, &fv_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_prime_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &u_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_global));
  PetscCall(DMRestoreGlobalVector(fdm, &fv_global));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p_prime[0]));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p_prime[1]));
  PetscCall(DMRestoreGlobalVector(fdm, &grad_p_prime_f));
  PetscCall(DMRestoreGlobalVector(dm, &lap_p_prime));

  PetscCall(NSFSMCalculateConvection2d_Cart_Internal(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSUStar2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;

  Vec u_global, p_global, Nu_global, Nu_prev_global, helm_u, dpdx, valbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &u_global));
  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &Nu_global));
  PetscCall(DMGetGlobalVector(dm, &Nu_prev_global));
  PetscCall(DMGetGlobalVector(dm, &helm_u));
  PetscCall(DMGetGlobalVector(dm, &dpdx));
  PetscCall(DMGetGlobalVector(dm, &valbc));

  PetscCall(DMLocalToGlobal(dm, fsm->v[0], INSERT_VALUES, u_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N[0], INSERT_VALUES, Nu_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N_prev[0], INSERT_VALUES, Nu_prev_global));

  PetscCall(MatMult(fsm->helm_v, u_global, helm_u));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, helm_u));

  PetscCall(MatMult(fsm->grad_p[0], p_global, dpdx));

  PetscCall(VecSet(valbc, 0.));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, valbc));

  PetscCall(VecCopy(helm_u, b));
  PetscCall(VecAXPY(b, -1.5 * ns->dt, Nu_global));
  PetscCall(VecAXPY(b, 0.5 * ns->dt, Nu_prev_global));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, dpdx));
  PetscCall(VecAXPY(b, 1.0, valbc));

  PetscCall(DMRestoreGlobalVector(dm, &u_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nu_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nu_prev_global));
  PetscCall(DMRestoreGlobalVector(dm, &helm_u));
  PetscCall(DMRestoreGlobalVector(dm, &dpdx));
  PetscCall(DMRestoreGlobalVector(dm, &valbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSVStar2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;

  Vec v_global, p_global, Nv_global, Nv_prev_global, helm_v, dpdy, valbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &v_global));
  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &Nv_global));
  PetscCall(DMGetGlobalVector(dm, &Nv_prev_global));
  PetscCall(DMGetGlobalVector(dm, &helm_v));
  PetscCall(DMGetGlobalVector(dm, &dpdy));
  PetscCall(DMGetGlobalVector(dm, &valbc));

  PetscCall(DMLocalToGlobal(dm, fsm->v[1], INSERT_VALUES, v_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N[1], INSERT_VALUES, Nv_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N_prev[1], INSERT_VALUES, Nv_prev_global));

  PetscCall(MatMult(fsm->helm_v, v_global, helm_v));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, helm_v));

  PetscCall(MatMult(fsm->grad_p[1], p_global, dpdy));

  PetscCall(VecSet(valbc, 0.));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, valbc));

  PetscCall(VecCopy(helm_v, b));
  PetscCall(VecAXPY(b, -1.5 * ns->dt, Nv_global));
  PetscCall(VecAXPY(b, 0.5 * ns->dt, Nv_prev_global));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, dpdy));
  PetscCall(VecAXPY(b, 1.0, valbc));

  PetscCall(DMRestoreGlobalVector(dm, &v_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nv_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nv_prev_global));
  PetscCall(DMRestoreGlobalVector(dm, &helm_v));
  PetscCall(DMRestoreGlobalVector(dm, &dpdy));
  PetscCall(DMRestoreGlobalVector(dm, &valbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSPprime2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  MPI_Comm     comm;
  MatNullSpace nullspace;

  Vec UV_star_global;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(DMGetGlobalVector(fdm, &UV_star_global));
  PetscCall(DMLocalToGlobal(fdm, fsm->fv_star, INSERT_VALUES, UV_star_global));

  PetscCall(MatMult(fsm->div_fv, UV_star_global, b));
  PetscCall(VecScale(b, ns->rho / ns->dt));

  // TODO: below is only for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  PetscCall(DMRestoreGlobalVector(fdm, &UV_star_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  (void)J;

  NS ns = (NS)ctx;
  DM dm;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(dm, ns->bcs, 1., -0.5 * ns->mu * ns->dt / ns->rho, Jpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorPprime2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS           ns = (NS)ctx;
  MPI_Comm     comm;
  DM           dm;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(dm, ns->bcs, Jpre));

  // TODO: below is temporary for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(J, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS ns, PetscViewer v)
{
  PetscBool iscgns;
  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) PetscCall(NSViewSolution_FSM_Cart_CGNS_Internal(ns, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
