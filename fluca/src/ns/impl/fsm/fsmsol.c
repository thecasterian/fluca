#include <flucansfsm.h>
#include <fluca/private/nsfsmimpl.h>

PetscErrorCode NSFSMGetHalfStepPressure(NS ns, Vec *p_half)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(ns, NS_CLASSID, 1, NSFSM);
  if (p_half) *p_half = fsm->p_half;
  PetscFunctionReturn(PETSC_SUCCESS);
}
