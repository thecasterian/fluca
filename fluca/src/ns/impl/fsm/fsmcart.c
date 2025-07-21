#include <fluca/private/flucaviewercgnsimpl.h>
#include <fluca/private/nsfsmimpl.h>

PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS ns, PetscViewer v)
{
  PetscBool iscgns;
  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) PetscCall(NSViewSolution_FSM_Cart_CGNS_Internal(ns, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
