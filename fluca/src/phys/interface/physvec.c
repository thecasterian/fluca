#include <fluca/private/physimpl.h>
#include <fluca/private/meshimpl.h>

PetscErrorCode PhysCreateSolutionVector(Phys phys, Vec *v)
{
  MeshField         fields[PHYS_MAX_FIELDS];
  PhysFieldLocation loc;
  PetscInt          f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscAssertPointer(v, 2);
  PetscCheck(phys->setupcalled, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "Must call PhysSetUp() before PhysCreateSolutionVector()");
  for (f = 0; f < phys->nfields; ++f) {
    PetscCall(PhysGetField(phys, phys->fields[f].name, &loc, &fields[f].c0, &fields[f].ncomp));
    fields[f].name = phys->fields[f].name;
    fields[f].loc  = loc == PHYS_FIELD_FACE ? DMSTAG_LEFT : DMSTAG_ELEMENT;
  }
  PetscCall(DMCreateGlobalVector(phys->sol_dm, v));
  PetscCall(MeshVecSetFields_Internal(phys->mesh, *v, phys->nfields, fields));
  PetscFunctionReturn(PETSC_SUCCESS);
}
