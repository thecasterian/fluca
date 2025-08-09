#include <flucansfsm.h>
#include <fluca/private/nsfsmimpl.h>

PetscErrorCode NSFSMGetVelocity(NS ns, Vec *u, Vec *v, Vec *w)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (u) *u = fsm->v[0];
  if (v) *v = fsm->v[1];
  if (w) *w = fsm->v[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetIntermediateVelocity(NS ns, Vec *u_star, Vec *v_star, Vec *w_star)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (u_star) *u_star = fsm->v_star[0];
  if (v_star) *v_star = fsm->v_star[1];
  if (w_star) *w_star = fsm->v_star[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetConvection(NS ns, Vec *Nu, Vec *Nv, Vec *Nw)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (Nu) *Nu = fsm->N[0];
  if (Nv) *Nv = fsm->N[1];
  if (Nw) *Nw = fsm->N[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetPreviousConvection(NS ns, Vec *Nu_prev, Vec *Nv_prev, Vec *Nw_prev)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (Nu_prev) *Nu_prev = fsm->N_prev[0];
  if (Nv_prev) *Nv_prev = fsm->N_prev[1];
  if (Nw_prev) *Nw_prev = fsm->N_prev[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetPressure(NS ns, Vec *p)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (p) *p = fsm->p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetHalfStepPressure(NS ns, Vec *p_half)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (p_half) *p_half = fsm->p_half;
  PetscFunctionReturn(PETSC_SUCCESS);
}
