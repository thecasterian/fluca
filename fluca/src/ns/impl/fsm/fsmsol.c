#include <flucansfsm.h>
#include <fluca/private/nsfsmimpl.h>

PetscErrorCode NSFSMGetVelocity(NS ns, Vec *v)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (v) *v = fsm->v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetIntermediateVelocity(NS ns, Vec *v_star)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (v_star) *v_star = fsm->v_star;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetConvection(NS ns, Vec *N)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (N) *N = fsm->N;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMGetPreviousConvection(NS ns, Vec *N_prev)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (N_prev) *N_prev = fsm->N_prev;
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
