#include <fluca/private/flucaviewercgnsimpl.h>
#include <fluca/private/nsfsmimpl.h>
#include <flucameshcart.h>
#include <petscdmstag.h>

static PetscErrorCode ComputeRHSUStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSVStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSPprime2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeOperatorPprime2d_Private(KSP, Mat, Mat, void *);

static PetscErrorCode NSFSMCalculateConvection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;
  Vec     u_interp, v_interp, UV_u_interp, UV_v_interp;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMGetGlobalVector(fdm, &u_interp));
  PetscCall(DMGetGlobalVector(fdm, &v_interp));
  PetscCall(DMGetGlobalVector(fdm, &UV_u_interp));
  PetscCall(DMGetGlobalVector(fdm, &UV_v_interp));

  PetscCall(MatMult(fsm->interp_v[0], fsm->v[0], u_interp));
  PetscCall(MatMultAdd(fsm->interp_v[1], fsm->v[0], u_interp, u_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 0, ns->bcs, ns->t + ns->dt, ADD_VALUES, u_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 0, ns->bcs, ns->t + ns->dt, ADD_VALUES, u_interp));
  PetscCall(MatMult(fsm->interp_v[0], fsm->v[1], v_interp));
  PetscCall(MatMultAdd(fsm->interp_v[1], fsm->v[1], v_interp, v_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 1, ns->bcs, ns->t + ns->dt, ADD_VALUES, v_interp));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 1, ns->bcs, ns->t + ns->dt, ADD_VALUES, v_interp));

  PetscCall(VecPointwiseMult(UV_u_interp, fsm->fv, u_interp));
  PetscCall(VecPointwiseMult(UV_v_interp, fsm->fv, v_interp));
  PetscCall(MatMult(fsm->div_f, UV_u_interp, fsm->N[0]));
  PetscCall(MatMult(fsm->div_f, UV_v_interp, fsm->N[1]));

  PetscCall(DMRestoreGlobalVector(fdm, &u_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &v_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_u_interp));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_v_interp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMSetKSPComputeFunctions2d_Cart_Internal(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  PetscInt d;

  PetscErrorCode (*rhs[2])(KSP, Vec, void *) = {ComputeRHSUStar2d_Private, ComputeRHSVStar2d_Private};

  PetscFunctionBegin;
  for (d = 0; d < 2; ++d) {
    PetscCall(KSPSetComputeOperators(fsm->kspv[d], ComputeOperatorsUVstar2d_Private, ns));
    PetscCall(KSPSetComputeRHS(fsm->kspv[d], rhs[d], ns));
  }
  PetscCall(KSPSetComputeOperators(fsm->kspp, ComputeOperatorPprime2d_Private, ns));
  PetscCall(KSPSetComputeRHS(fsm->kspp, ComputeRHSPprime2d_Private, ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS ns)
{
  NS_FSM  *fsm = (NS_FSM *)ns->data;
  DM       dm, fdm;
  Vec      grad_p[2], grad_p_f, u_tilde, v_tilde, UV_tilde, s;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    PetscCall(KSPSolve(fsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(fsm->kspv[d], &s));
    PetscCall(VecCopy(s, fsm->v_star[d]));
  }

  /* Calculate face-centered intermediate velocity. */
  PetscCall(DMGetGlobalVector(dm, &grad_p[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p[1]));
  PetscCall(DMGetGlobalVector(fdm, &grad_p_f));
  PetscCall(DMGetGlobalVector(dm, &u_tilde));
  PetscCall(DMGetGlobalVector(dm, &v_tilde));
  PetscCall(DMGetGlobalVector(fdm, &UV_tilde));

  PetscCall(MatMult(fsm->grad_p[0], fsm->p_half, grad_p[0]));
  PetscCall(MatMult(fsm->grad_p[1], fsm->p_half, grad_p[1]));
  PetscCall(MatMult(fsm->grad_f, fsm->p_half, grad_p_f));

  PetscCall(VecWAXPY(u_tilde, ns->dt / ns->rho, grad_p[0], fsm->v_star[0]));
  PetscCall(VecWAXPY(v_tilde, ns->dt / ns->rho, grad_p[1], fsm->v_star[1]));

  PetscCall(MatMult(fsm->interp_v[0], u_tilde, UV_tilde));
  PetscCall(MatMultAdd(fsm->interp_v[1], v_tilde, UV_tilde, UV_tilde));

  PetscCall(VecWAXPY(fsm->fv_star, -ns->dt / ns->rho, grad_p_f, UV_tilde));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 0, ns->bcs, ns->t + ns->dt, INSERT_VALUES, fsm->fv_star));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 1, ns->bcs, ns->t + ns->dt, INSERT_VALUES, fsm->fv_star));

  PetscCall(DMRestoreGlobalVector(dm, &grad_p[0]));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p[1]));
  PetscCall(DMRestoreGlobalVector(fdm, &grad_p_f));
  PetscCall(DMRestoreGlobalVector(dm, &u_tilde));
  PetscCall(DMRestoreGlobalVector(dm, &v_tilde));
  PetscCall(DMRestoreGlobalVector(fdm, &UV_tilde));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculatePressureCorrection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Vec     s;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(KSPSolve(fsm->kspp, NULL, NULL));
  PetscCall(KSPGetSolution(fsm->kspp, &s));
  PetscCall(VecCopy(s, fsm->p_prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMUpdate2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;
  Vec     grad_p_prime[2], grad_p_prime_f, lap_p_prime;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[1]));
  PetscCall(DMGetGlobalVector(fdm, &grad_p_prime_f));
  PetscCall(DMGetGlobalVector(dm, &lap_p_prime));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecCopy(fsm->N[1], fsm->N_prev[1]));

  PetscCall(MatMult(fsm->grad_p_prime[0], fsm->p_prime, grad_p_prime[0]));
  PetscCall(MatMult(fsm->grad_p_prime[1], fsm->p_prime, grad_p_prime[1]));
  PetscCall(MatMult(fsm->grad_f, fsm->p_prime, grad_p_prime_f));
  PetscCall(MatMult(fsm->lap_p_prime, fsm->p_prime, lap_p_prime));

  PetscCall(VecWAXPY(fsm->v[0], -ns->dt / ns->rho, grad_p_prime[0], fsm->v_star[0]));
  PetscCall(VecWAXPY(fsm->v[1], -ns->dt / ns->rho, grad_p_prime[1], fsm->v_star[1]));
  PetscCall(VecWAXPY(fsm->fv, -ns->dt / ns->rho, grad_p_prime_f, fsm->fv_star));

  PetscCall(VecAXPBYPCZ(fsm->p_half, 1., -0.5 * ns->mu * ns->dt / ns->rho, 1., fsm->p_prime, lap_p_prime));
  PetscCall(VecAXPBYPCZ(fsm->p, 1.5, -0.5, 0., fsm->p_half, fsm->p_half_prev));

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
  Vec     grad_p_x;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &grad_p_x));

  PetscCall(MatMult(fsm->grad_p[0], fsm->p_half, grad_p_x));

  PetscCall(MatMult(fsm->helm_v, fsm->v[0], b));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, b));

  PetscCall(VecAXPBYPCZ(b, -1.5 * ns->dt, 0.5 * ns->dt, 1., fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, grad_p_x));

  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, b));

  PetscCall(DMRestoreGlobalVector(dm, &grad_p_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSVStar2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;

  Vec grad_p_y;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &grad_p_y));

  PetscCall(MatMult(fsm->grad_p[1], fsm->p_half, grad_p_y));

  PetscCall(MatMult(fsm->helm_v, fsm->v[1], b));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, b));

  PetscCall(VecAXPBYPCZ(b, -1.5 * ns->dt, 0.5 * ns->dt, 1., fsm->N[1], fsm->N_prev[1]));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, grad_p_y));

  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, b));

  PetscCall(DMRestoreGlobalVector(dm, &grad_p_y));
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

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(MatMult(fsm->div_f, fsm->fv_star, b));
  PetscCall(VecScale(b, ns->rho / ns->dt));

  // TODO: below is only for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));
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
