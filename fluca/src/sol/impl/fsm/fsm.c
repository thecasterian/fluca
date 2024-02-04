#include <fluca/private/meshimpl.h>
#include <fluca/private/sol_fsm.h>

extern PetscErrorCode SolView_FSMCGNS(Sol, PetscViewer);

PetscErrorCode SolSetMesh_FSM(Sol sol, Mesh mesh)
{
  Sol_FSM *fsm = (Sol_FSM *)sol->data;
  DM       dm, fdm;
  PetscInt dim, d;

  PetscFunctionBegin;
  for (d = 0; d < 3; ++d) {
    PetscCall(VecDestroy(&fsm->v_star[d]));
    PetscCall(VecDestroy(&fsm->v_tilde[d]));
    PetscCall(VecDestroy(&fsm->N[d]));
    PetscCall(VecDestroy(&fsm->N_prev[d]));
  }
  PetscCall(VecDestroy(&fsm->fv));
  PetscCall(VecDestroy(&fsm->fv_star));
  PetscCall(VecDestroy(&fsm->p_half));
  PetscCall(VecDestroy(&fsm->p_prime));
  PetscCall(VecDestroy(&fsm->p_half_prev));

  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(MeshGetFaceDM(mesh, &fdm));
  PetscCall(MeshGetDim(mesh, &dim));

  for (d = 0; d < dim; ++d) {
    PetscCall(DMCreateLocalVector(dm, &fsm->v_star[d]));
    PetscCall(DMCreateLocalVector(dm, &fsm->v_tilde[d]));
    PetscCall(DMCreateLocalVector(dm, &fsm->N[d]));
    PetscCall(DMCreateLocalVector(dm, &fsm->N_prev[d]));
  }
  PetscCall(DMCreateLocalVector(fdm, &fsm->fv));
  PetscCall(DMCreateLocalVector(fdm, &fsm->fv_star));
  PetscCall(DMCreateLocalVector(dm, &fsm->p_half));
  PetscCall(DMCreateLocalVector(dm, &fsm->p_prime));
  PetscCall(DMCreateLocalVector(dm, &fsm->p_half_prev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolDestroy_FSM(Sol sol)
{
  Sol_FSM *fsm = (Sol_FSM *)sol->data;
  PetscInt d;

  PetscFunctionBegin;
  for (d = 0; d < 3; ++d) {
    PetscCall(VecDestroy(&fsm->v_star[d]));
    PetscCall(VecDestroy(&fsm->v_tilde[d]));
    PetscCall(VecDestroy(&fsm->N[d]));
    PetscCall(VecDestroy(&fsm->N_prev[d]));
  }
  PetscCall(VecDestroy(&fsm->fv));
  PetscCall(VecDestroy(&fsm->fv_star));
  PetscCall(VecDestroy(&fsm->p_half));
  PetscCall(VecDestroy(&fsm->p_prime));
  PetscCall(VecDestroy(&fsm->p_half_prev));

  PetscCall(PetscFree(sol->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolView_FSM(Sol sol, PetscViewer v)
{
  PetscBool iscgns;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERCGNS, &iscgns));
  if (iscgns) { PetscCall(SolView_FSMCGNS(sol, v)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolCreate_FSM(Sol sol)
{
  Sol_FSM *fsm;
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fsm));
  sol->data = (void *)fsm;

  for (d = 0; d < 3; ++d) {
    fsm->v_star[d]  = NULL;
    fsm->v_tilde[d] = NULL;
    fsm->N[d]       = NULL;
    fsm->N_prev[d]  = NULL;
  }
  fsm->fv          = NULL;
  fsm->fv_star     = NULL;
  fsm->p_half      = NULL;
  fsm->p_prime     = NULL;
  fsm->p_half_prev = NULL;

  sol->ops->setmesh = SolSetMesh_FSM;
  sol->ops->destroy = SolDestroy_FSM;
  sol->ops->view    = SolView_FSM;
  PetscFunctionReturn(PETSC_SUCCESS);
}
